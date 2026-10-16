## Data Preparation

### MM-IMDb

First, download the dataset from this link: https://archive.org/download/mmimdb/mmimdb.tar.gz

Then, place the raw images in folder **dataset/mmimdb/image** and put the json files in folder **dataset/mmimdb/meta_data**.

### HateMemes

First, download the dataset from this link: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

Then, place the raw images in folder **dataset/hatememes/image** and put the json files in folder **dataset/hatememes/metadata**.

Next, replace the **test.json** in metadata with **test_seen.json** downloaded from this link: https://www.kaggle.com/datasets/williamberrios/hateful-memes as the test.json downloaded from the prior website has no label information for evaluation. (Do not change other files, only replace the test.json with test_seen.json)

### Food101

First, download the dataset from this link: https://www.kaggle.com/datasets/gianmarco96/upmcfood101

Then, place the raw images in folder **dataset/food101/image** and put the csv files in folder **dataset/food101/meta_data**.

Note: The image folder follows the structure dataset/xxx/image/xxx.jpg|png|jpeg

## Code Running

Run all commands from the repository root.

### Dataset Initiation

Take the HateMemes dataset for an example, run the following command to build the split files, memory bank, and retrieval results:

```shell
python src/init_data.py --config-name=cpri_config data_para.dataset_name=hatememes
```

Use `data_para.dataset_name=mmimdb` or `data_para.dataset_name=food101` to initialize the other datasets.

### Training & Evaluation

Run the following command to train the model and evaluate the results:

```shell
python src/main.py --config-name=cpri_config data_para.dataset_name=hatememes data_para.missing_type=both data_para.missing_rate=0.7
```

`missing_type` can be `text`, `image`, or `both`. All parameters can be changed in **src/config/cpri_config.yaml** or overridden on the command line.

