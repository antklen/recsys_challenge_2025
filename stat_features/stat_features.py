import os
import pandas as pd
import numpy as np
import hydra

@hydra.main(version_base=None, config_path="configs", config_name="stat_features")
def main(config):

    SUFFIX = 'submit' if config.submit else 'local'
    product_buy, add_to_cart, page_visit, remove_from_cart, product_properties, relevant_clients = load_data(
        config.data_path, config.submit)

    all_interactions = [(product_buy, 'buy'), (add_to_cart, 'add'),
                        (page_visit, 'visit'), (remove_from_cart, 'remove')]

    action_interecation = [(product_buy, 'buy'), (add_to_cart, 'add'),
                           (remove_from_cart, 'remove')]
    all_features = []

    print('1')
    for df, prefix in all_interactions:
        all_features.append(calculate_interaction(df, prefix=prefix))
        all_features.append(mean_interaction_interval(df, prefix=prefix))
        all_features.append(calculate_interaction_diff(df, prefix=prefix))
        all_features.append(calculate_days_to_start(df, prefix=prefix))

    print('2')
    for df, prefix in action_interecation:
        all_features.append(
            category_popularity(df, product_properties, prefix=prefix))
        all_features.append(sku_popularity(df, prefix=prefix))

    all_features.append(sku_popularity(page_visit, sku='url', prefix='visit'))

    print('3')
    all_features.append(calculate_stability(page_visit))
    all_features.append(price_features(product_buy, product_properties))

    all_features.append(percent_product(product_buy, add_to_cart))
    all_features.append(
        percent_categories(product_buy, add_to_cart, product_properties))
    print('4')
    user_features = pd.DataFrame({'client_id': relevant_clients})
    user_features = merge(user_features, all_features)

    if not config.submit:
        mask = (user_features.drop(columns=['client_id']) != 0).any(axis=1)
        user_features = user_features[mask]

    client_ids = np.array(user_features['client_id'])
    embeddings = np.array(user_features.iloc[:, 1:])

    if not os.path.exists(
            f'{config.data_path}/embeddings_{SUFFIX}/stat_features/'):
        os.makedirs(f'{config.data_path}/embeddings_{SUFFIX}/stat_features/')

    np.save(
        os.path.join(f'{config.data_path}/embeddings_{SUFFIX}/stat_features/',
                     'embeddings.npy'), embeddings.astype('float16'))
    np.save(
        os.path.join(f'{config.data_path}/embeddings_{SUFFIX}/stat_features/',
                     'client_ids.npy'), client_ids)
    print(user_features.shape)
    print(user_features)


def load_data(data_path, submit):

    PREFIX = '' if submit else 'input'

    product_buy = pd.read_parquet(data_path + f'/raw/{PREFIX}/product_buy.parquet')
    add_to_cart = pd.read_parquet(data_path + f'/raw/{PREFIX}/add_to_cart.parquet')
    page_visit = pd.read_parquet(data_path + f'/raw/{PREFIX}/page_visit.parquet')
    remove_from_cart = pd.read_parquet(data_path + f'/raw/{PREFIX}/remove_from_cart.parquet')
    product_properties = pd.read_parquet(data_path + '/raw/product_properties.parquet')
    relevant_clients = np.load(data_path + '/raw/input/relevant_clients.npy')

    product_buy = product_buy[product_buy['client_id'].isin(relevant_clients)]
    add_to_cart = add_to_cart[add_to_cart['client_id'].isin(relevant_clients)]
    page_visit = page_visit[page_visit['client_id'].isin(relevant_clients)]
    remove_from_cart = remove_from_cart[remove_from_cart['client_id'].isin(relevant_clients)]
    return product_buy, add_to_cart, page_visit, remove_from_cart, product_properties, relevant_clients

def calculate_interaction(df, client_id='client_id', timestamp='timestamp', prefix='add'):

    df = df.copy()
    df[timestamp] = pd.to_datetime(df[timestamp])

    feature_data = (df.groupby(client_id).agg(total_interactions=(timestamp, 'size')).reset_index())

    if prefix:
        feature_data = feature_data.rename(columns={'total_interactions': f'{prefix}_total_interactions'})

    return feature_data


def calculate_stability(df, client_id='client_id', timestamp='timestamp'):

    df = df.copy()
    df[timestamp] = pd.to_datetime(df[timestamp])

    df['year_week'] = df[timestamp].dt.strftime('%Y-%U')
    df['year_month'] = df[timestamp].dt.strftime('%Y-%m')

    group = df.groupby(client_id)
    feature_data = group.agg(unique_weeks=('year_week', 'nunique'),
                             unique_months=('year_month', 'nunique')).reset_index()

    return feature_data


def calculate_days_to_start(df, client_id='client_id', timestamp='timestamp', prefix='add'):

    df = df.copy()
    df[timestamp] = pd.to_datetime(df[timestamp])

    start_date = df[timestamp].min()

    group = df.groupby(client_id)[timestamp].agg(['min', 'max']).reset_index()

    group['days_from_first_to_start'] = (group['min'] - start_date).dt.days
    group['days_from_last_to_start'] = (group['max'] - start_date).dt.days

    group = group.drop(['min', 'max'], axis=1)

    group = group.rename(columns={
        'days_from_first_to_start': f'{prefix}_days_from_first_to_start',
        'days_from_last_to_start': f'{prefix}_days_from_last_to_start'
    })

    return group


def sku_popularity(data, sku='sku', client_id='client_id', prefix='buy'):
    count = data[sku].value_counts().reset_index(drop=False).rename(columns={'count': f'average_pop_{prefix}'})

    data = data.merge(count, on=sku, how='left')

    count_pop = data.groupby(client_id)[f'average_pop_{prefix}'].agg('mean').reset_index()
    return count_pop


def category_popularity(data, product_properties, sku='sku', category="category", client_id='client_id', prefix='buy'):
    category_pop = (data.merge(
        product_properties[[sku, category]], on=sku,
        how='left')[category].value_counts().reset_index().rename(
            columns={'count': f'category_popularity_{prefix}'}))

    merged_data = (data.merge(product_properties[[sku, category]],
                              on=sku,
                              how='left').merge(category_pop,
                                                on=category,
                                                how='left'))

    avg_category_pop = merged_data.groupby(client_id)[f'category_popularity_{prefix}'].agg('mean').reset_index()
    return avg_category_pop


def percent_product(product_buy, add_to_cart, client_id='client_id', sku='sku'):
    added_items = add_to_cart[[client_id, sku]].drop_duplicates()

    bought_items = product_buy[[client_id, sku]].drop_duplicates()

    merged = added_items.merge(bought_items.assign(bought=1),
                               on=[client_id, sku],
                               how='left')
    merged['bought'] = merged['bought'].fillna(0)
    user_conversion = merged.groupby(client_id)['bought'].mean()
    return user_conversion


def percent_categories(product_buy, add_to_cart, product_properties, client_id='client_id', category='category', sku='sku'):

    product_buy = product_buy.merge(product_properties[[sku, category]],
                                    on=sku,
                                    how='left')
    add_to_cart = add_to_cart.merge(product_properties[[sku, category]],
                                    on=sku,
                                    how='left')

    added_items = add_to_cart[[client_id, category]].drop_duplicates()

    bought_items = product_buy[[client_id, category]].drop_duplicates()

    merged = added_items.merge(bought_items.assign(bought=1),
                               on=[client_id, category],
                               how='left')
    merged['bought_cat'] = merged['bought'].fillna(0)
    user_conversion = merged.groupby(client_id)['bought_cat'].mean()
    return user_conversion


def mean_interaction_interval(data, prefix='buy', timestamp='timestamp', client_id='client_id'):
    df = data.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp]):
        df[timestamp] = pd.to_datetime(df[timestamp])

    grouped = df.sort_values(timestamp).groupby(client_id)[timestamp]

    intervals = grouped.apply(lambda x: x.diff().dt.total_seconds().mean() /
                              (24 * 3600) if len(x) > 1 else 0)

    return intervals.rename(f'mean_{prefix}_interval_days')


def calculate_interaction_diff(events, prefix='buy', timestamp='timestamp', client_id='client_id'):
    df = events.copy()

    df[timestamp] = pd.to_datetime(df[timestamp])
    df['date'] = df[timestamp].dt.date

    anchor_date = df['date'].max()
    anchor = pd.Timestamp(anchor_date)

    last_week_start = anchor - pd.Timedelta(days=6)
    prev_week_start = last_week_start - pd.Timedelta(days=7)

    last_month_start = anchor - pd.offsets.MonthBegin(1)
    prev_month_start = last_month_start - pd.offsets.MonthBegin(1)
    last_month_end = last_month_start + pd.offsets.MonthEnd(0)
    prev_month_end = prev_month_start + pd.offsets.MonthEnd(0)

    last_week_data = df[df['date'].between(last_week_start.date(), anchor_date)]

    prev_week_data = df[df['date'].between(
        prev_week_start.date(),
        (last_week_start - pd.Timedelta(days=1)).date()
    )]

    last_month_data = df[df['date'].between(
        last_month_start.date(),
        last_month_end.date()
    )]

    prev_month_data = df[df['date'].between(
        prev_month_start.date(),
        prev_month_end.date()
    )]

    last_week_counts = last_week_data.groupby(client_id).size()
    prev_week_counts = prev_week_data.groupby(client_id).size()
    last_month_counts = last_month_data.groupby(client_id).size()
    prev_month_counts = prev_month_data.groupby(client_id).size()

    result = pd.concat([
        last_week_counts.rename('last_week'),
        prev_week_counts.rename('prev_week'),
        last_month_counts.rename('last_month'),
        prev_month_counts.rename('prev_month')
    ], axis=1).fillna(0)

    result[f'diff_{prefix}_week'] = result['last_week'] - result['prev_week']
    result[f'diff_{prefix}_month'] = result['last_month'] - result['prev_month']

    return result[[f'diff_{prefix}_week', f'diff_{prefix}_month']].reset_index()


def merge(user_features, frames, column='client_id'):
    for data in frames:
        user_features = user_features.merge(data, on=column, how='left').fillna(0)
    return user_features


def price_features(product_buy, product_properties, client_id='client_id', timestamp='timestamp', sku='sku', price='price'):
    purchases_with_prices = pd.merge(
    product_buy[[client_id, timestamp, sku]],
    product_properties[[sku, "price"]],
    on=sku,
    how='left')

    user_spending = purchases_with_prices.groupby(client_id).agg(
        total_spent=(price, 'sum'),
        average_spent=(price, 'mean'),
    ).reset_index()
    return user_spending

if __name__ == "__main__":

    main()
