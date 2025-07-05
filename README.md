# RecSys Challenge 2025 by Synerise

2nd place solution for [RecSys Challenge 2025](https://recsys.synerise.com) by Synerise (team ai_lab_recsys).

## Preliminaries

- Define environment variable `CHALLENGE_DATA_PATH` with path where all data will be placed.
- Download the dataset from the [official page](https://recsys.synerise.com/summary#download) and put it inside `$CHALLENGE_DATA_PATH/raw`.
- Install requirements (`requirements.txt`)
- Add root of the repository to PYTHONPATH and run scripts from it: `export PYTHONPATH="./"`.

> [!WARNING]
> The Implicit library must be installed with GPU support. This is usually done via conda, for this first install cudatoolkit
>
> `conda install -c conda-forge cudatoolkit`
>
> `conda install -c conda-forge implicit implicit-proc=*=gpu`

Embeddings will be placed in `$CHALLENGE_DATA_PATH/embeddings_submit`.

By default, the GPU with index 0 is selected, to select another GPU, change the parameter `cuda_visible_devices` in the config files or override this parameter in the command line (e.g. `python gru_autoencoder/gru_ae_day_event_embeddings.py cuda_visible_devices=1`).

For validation on open tasks:
- Split the data with [script from the competition repository](https://github.com/Synerise/recsys2025?tab=readme-ov-file#split-data-script).
- Set `submit: False` in config files or override this parameter in the command line (e.g. `python gru_autoencoder/gru_ae_day_event_embeddings.py submit=False`).
- Embeddings will be placed in `$CHALLENGE_DATA_PATH/embeddings_local`.

## Get embeddings from separate models

Script to get all embeddings:
```sh
sh get_all_embeddings.sh
```
The scripts for each model are described below separetely.

Get embeddings from GRU autoencoder
```sh
# Embeddings for day and event type
python gru_autoencoder/gru_ae_day_event_embeddings.py
# Embeddings for quantized text representation of sku
python gru_autoencoder/gru_ae_sku_text_embeddings.py 
# Embeddings for event type, category, price, sku and url
python gru_autoencoder/gru_ae_embeddings.py --config-name=gru_ae_event_cat_price_sku_url
# Embeddings for week, event type, category, price, sku and url
python gru_autoencoder/gru_ae_embeddings.py --config-name=gru_ae_week_event_cat_price_sku_url
```

Get embeddings from ALS
```sh
# Embeddings for category, add and buy
python ALS/als_embeddings.py --config-name=als_buy_add_categories
# Embeddings for visit and url
python ALS/als_embeddings.py --config-name=als_visits_relevant
```

Get statistic features
```sh
python stat_features/stat_features.py 
```

Get embeddings from next sku prediction with transformer model:
```sh
python next_sku/next_sku_embeddings.py 
```

Get embeddings from next url prediction with transformer model:
```sh
python next_url/next_url_embeddings.py 
```

Get LLM's embeddings:
```sh
# Preprocess search queries only
python LLM/preprocessing_search.py

# Preprocess all interactions
python LLM/convert_to_text.py

# Run LLM inference for search queries
python LLM/get_embeddings.py  

# Run LLM inference for all interactions
python LLM/get_embeddings.py \
    data_files=all_interactions_64.parquet \
    save_dir_name=smollm2_all_interactions \
    n_components=128
```

Get embeddings from LightFM model:
```sh
python LightFM/lightfm_train.py 
```

## Combine separate embeddings into final embedding

Best solution (4.7504 on Leaderboard)
```sh
python scripts/merge.py --config-name=best_solution
```

Other top performing solutions with less models in the ensemble
```sh
# LB score 4.7499
python scripts/merge.py --config-name=score_4_7499
# LB score 4.7484
python scripts/merge.py --config-name=score_4_7484
# LB score 4.7418
python scripts/merge.py --config-name=score_4_7418
# LB score 4.7406
python scripts/merge.py --config-name=score_4_7406
```
