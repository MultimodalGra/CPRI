import torch
from torch import nn
import torch.nn.functional as F
from .vilt import ViltModel
from .modules import FMMG, CAPG


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class cpri(torch.nn.Module):
    def __init__(self,
                 vilt: ViltModel,
                 dataset_name: str,
                 max_text_len: int,
                 max_image_len: int,
                 missing_type: str,
                 device: str,
                 prompt_pos: int,
                 prompt_len: int,
                 dropout_rate: float,
                 hs=768,
                 **kargs):
        super(cpri, self).__init__()
        self.device = device
        self.max_text_len = max_text_len
        self.missing_type = missing_type
        self.dataset_name = dataset_name
        self.embedding_layer = vilt.embeddings
        self.encoder_layer = vilt.encoder.layer
        self.layernorm = vilt.layernorm
        self.prompt_len = prompt_len
        self.prompt_pos = prompt_pos
        self.hs = hs

        if dataset_name == "hatememes":
            cls_num = 2
        elif dataset_name == "food101":
            cls_num = 101
        elif dataset_name == "mmimdb":
            cls_num = 23

        self.freeze()

        self.pooler = vilt.pooler

        if missing_type == "text":
            self.FMMG = FMMG(n=max_text_len, d=hs, dropout_rate=dropout_rate)
        elif missing_type == "image":
            self.FMMG = FMMG(n=max_image_len, d=hs, dropout_rate=dropout_rate)
        elif missing_type == "both":
            self.FMMG_t = FMMG(n=max_text_len, d=hs, dropout_rate=dropout_rate)
            self.FMMG_i = FMMG(n=max_image_len, d=hs, dropout_rate=dropout_rate)

        self.dynamic_prompt = CAPG(prompt_len=prompt_len)

        self.label_enhanced = nn.Parameter(torch.randn(cls_num, hs))
        self.classifier = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, hs),
        )
        self.classifier.apply(init_weights)

        self.num_attn_res_layers = 6

        self.t_query_proj = nn.ModuleList([nn.Linear(hs, hs) for _ in range(self.num_attn_res_layers)])
        self.i_query_proj = nn.ModuleList([nn.Linear(hs, hs) for _ in range(self.num_attn_res_layers)])
        self.l_query_proj = nn.ModuleList(
            [nn.Linear(hs, hs) for _ in range(self.num_attn_res_layers)])

        self.t_key_proj = nn.Linear(hs, hs)
        self.i_key_proj = nn.Linear(hs, hs)
        self.l_key_proj = nn.Linear(hs, hs)

        try:
            self.res_norm_t = nn.RMSNorm(hs)
            self.res_norm_i = nn.RMSNorm(hs)
            self.res_norm_l = nn.RMSNorm(hs)
        except AttributeError:
            self.res_norm_t = nn.LayerNorm(hs, elementwise_affine=False)
            self.res_norm_i = nn.LayerNorm(hs, elementwise_affine=False)
            self.res_norm_l = nn.LayerNorm(hs, elementwise_affine=False)

        proj_layers = [self.t_key_proj, self.i_key_proj, self.l_key_proj] + \
                      list(self.t_query_proj) + list(self.i_query_proj) + list(self.l_query_proj)
        for proj in proj_layers:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float32

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self,
                input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                pixel_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                r_t_list: torch.Tensor,
                r_i_list: torch.Tensor,
                r_l_list: torch.Tensor,
                missing_mask=None,
                image_token_type_idx=1):
        embedding, attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)
        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:, :]

        if self.missing_type == "text":
            recovered_t = self.FMMG(r_t_list)
            missing_mask_t = missing_mask.view(-1, 1, 1).expand(-1, self.max_text_len, self.hs)
            text_emb = text_emb * missing_mask_t + recovered_t * (1 - missing_mask_t)

        elif self.missing_type == "image":
            recovered_i = self.FMMG(r_i_list)
            missing_mask_i = missing_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
            image_emb = image_emb * missing_mask_i + recovered_i * (1 - missing_mask_i)

        elif self.missing_type == "both":
            recovered_t = self.FMMG_t(r_t_list)
            recovered_i = self.FMMG_i(r_i_list)
            t_missing_mask = [0 if i == 0 else 1 for i in missing_mask]
            i_missing_mask = [0 if i == 1 else 1 for i in missing_mask]
            t_missing_mask = torch.tensor(t_missing_mask).to(self.device)
            i_missing_mask = torch.tensor(i_missing_mask).to(self.device)
            missing_mask_t = t_missing_mask.view(-1, 1, 1).expand(-1, self.max_text_len, self.hs)
            missing_mask_i = i_missing_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
            text_emb = text_emb * missing_mask_t + recovered_t * (1 - missing_mask_t)
            image_emb = image_emb * missing_mask_i + recovered_i * (1 - missing_mask_i)

        t_prompt, i_prompt = self.dynamic_prompt(r_i=r_i_list, r_t=r_t_list, T=text_emb, V=image_emb)

        if self.dataset_name == "hatememes" or self.dataset_name == "food101":
            label_emb = self.label_enhanced[r_l_list]
            label_cls = self.label_enhanced
            label_emb = torch.mean(label_emb, dim=1)
            label_emb = label_emb.view(-1, 1, self.hs)

        elif self.dataset_name == "mmimdb":
            label_tmp = self.label_enhanced.repeat(r_l_list.shape[0], 1, 1)
            label_cls = self.label_enhanced
            label_emb = torch.matmul(r_l_list.float(), label_tmp)
            label_emb = torch.mean(label_emb, dim=1)
            label_emb = label_emb.view(-1, 1, self.hs)

        output = torch.cat([text_emb, image_emb], dim=1)
        history_t_prompts = [t_prompt]
        history_i_prompts = [i_prompt]
        history_label_embs = [label_emb]

        N = embedding.shape[0]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, dtype=embedding.dtype)

        for i, layer_module in enumerate(self.encoder_layer):
            if i == 0:
                cls_token = output[:, :1, :]
                other_tokens = output[:, 1:, :]
                output = torch.cat([cls_token, label_emb, t_prompt, i_prompt, other_tokens], dim=1)

                attention_mask = torch.cat(
                    [attention_mask[:, :1], torch.ones(N, 1 + self.prompt_len * 2).to(self.device),
                     attention_mask[:, 1:]], dim=1)
                extended_attention_mask = self.get_extended_attention_mask(attention_mask, dtype=output.dtype)

                layer_outputs = layer_module(output, attention_mask=extended_attention_mask)
                output = layer_outputs[0]

                history_label_embs.append(output[:, 1:2, :])
                history_t_prompts.append(output[:, 2: 2 + self.prompt_len, :])
                history_i_prompts.append(output[:, 2 + self.prompt_len: 2 + self.prompt_len * 2, :])

            elif i >= 1 and i <= 6:
                cls_token = output[:, :1, :]
                other_tokens = output[:, 2 + self.prompt_len * 2:, :]

                q_t = self.t_query_proj[i - 1](cls_token).squeeze(1)
                q_i = self.i_query_proj[i - 1](cls_token).squeeze(1)
                q_l = self.l_query_proj[i - 1](cls_token).squeeze(1)

                V_t = torch.stack(history_t_prompts, dim=1)
                V_i = torch.stack(history_i_prompts, dim=1)
                V_l = torch.stack(history_label_embs, dim=1)

                K_t = self.t_key_proj(self.res_norm_t(V_t.mean(dim=2)))
                K_i = self.i_key_proj(self.res_norm_i(V_i.mean(dim=2)))
                K_l = self.l_key_proj(self.res_norm_l(V_l.mean(dim=2)))

                attn_score_t = torch.bmm(q_t.unsqueeze(1), K_t.transpose(1, 2)) / (self.hs ** 0.5)
                attn_weights_t = F.softmax(attn_score_t, dim=-1).squeeze(1)

                attn_score_i = torch.bmm(q_i.unsqueeze(1), K_i.transpose(1, 2)) / (self.hs ** 0.5)
                attn_weights_i = F.softmax(attn_score_i, dim=-1).squeeze(1)

                attn_score_l = torch.bmm(q_l.unsqueeze(1), K_l.transpose(1, 2)) / (self.hs ** 0.5)
                attn_weights_l = F.softmax(attn_score_l, dim=-1).squeeze(1)

                new_t_prompt = torch.einsum('bn, bnpd -> bpd', attn_weights_t, V_t)
                new_i_prompt = torch.einsum('bn, bnpd -> bpd', attn_weights_i, V_i)
                new_label_emb = torch.einsum('bn, bnpd -> bpd', attn_weights_l, V_l)

                output = torch.cat([cls_token, new_label_emb, new_t_prompt, new_i_prompt, other_tokens], dim=1)

                layer_outputs = layer_module(output, attention_mask=extended_attention_mask)
                output = layer_outputs[0]

                history_label_embs.append(output[:, 1:2, :])
                history_t_prompts.append(output[:, 2: 2 + self.prompt_len, :])
                history_i_prompts.append(output[:, 2 + self.prompt_len: 2 + self.prompt_len * 2, :])

            else:
                layer_outputs = layer_module(output, attention_mask=extended_attention_mask)
                output = layer_outputs[0]

        output = self.layernorm(output)
        output = self.pooler(output)
        output = torch.cat([output, label_emb.squeeze(1)], dim=1)
        output = self.classifier(output)
        label_cls = label_cls.repeat(N, 1, 1)
        label_cls = label_cls.transpose(-1, -2)
        output = output.unsqueeze(1)
        output = torch.matmul(output, label_cls)
        output = output.squeeze(1)
        return {
            "output": output
        }
