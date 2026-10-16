## Data Preparation

### MM-IMDb

First, download the dataset from this link: https://archive.org/download/mmimdb/mmimdb.tar.gz

Then, place each image at **dataset/mmimdb/image/{id}.jpeg** and put the json files, including **split.json** and **{id}.json**, in folder **dataset/mmimdb/meta_data**.

### HateMemes

First, download the dataset from this link: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

Then, place each image at **dataset/hatememes/image/{id}.png** and put **train.jsonl**, **dev.jsonl**, and **test_seen.jsonl** in folder **dataset/hatememes/meta_data**. The code formats each jsonl `id` as a 5-digit name, so `1` must be stored as **00001.png**.

The initialization code reads **test_seen.jsonl** directly and saves it as **test.pkl**. It saves **dev.jsonl** as **valid.pkl**. Download **test_seen.jsonl** from https://www.kaggle.com/datasets/williamberrios/hateful-memes because the test file from the first source has no labels for evaluation.

### Food101

First, download the dataset from this link: https://www.kaggle.com/datasets/gianmarco96/upmcfood101

Then, place each image at **dataset/food101/image/{id}.jpg** and put **train_titles.csv**, **test_titles.csv**, and **class_idx.json** in folder **dataset/food101/meta_data**. `{id}` is the image file name in the csv with its extension removed.

Images stay in a flat **image** folder. The extension is fixed for each dataset: **.jpeg** for MM-IMDb, **.png** for HateMemes, and **.jpg** for Food101.

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

