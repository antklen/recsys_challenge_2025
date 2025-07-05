python gru_autoencoder/gru_ae_day_event_embeddings.py
python gru_autoencoder/gru_ae_sku_text_embeddings.py 
python gru_autoencoder/gru_ae_embeddings.py --config-name=gru_ae_event_cat_price_sku_url
python gru_autoencoder/gru_ae_embeddings.py --config-name=gru_ae_week_event_cat_price_sku_url

python ALS/als_embeddings.py --config-name=als_buy_add_categories
python ALS/als_embeddings.py --config-name=als_visits_relevant


python stat_features/stat_features.py

python next_sku/next_sku_embeddings.py 

python next_url/next_url_embeddings.py

python LLM/preprocessing_search.py 
python LLM/get_embeddings.py 

python LLM/convert_to_text.py
python LLM/get_embeddings.py \
    data_files=all_interactions_64.parquet \
    save_dir_name=smollm2_all_interactions \
    n_components=128

python LightFM/lightfm_train.py