import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UBCData:

    relevant_clients: pd.DataFrame
    product_properties: pd.DataFrame
    product_buy: pd.DataFrame
    add_to_cart: pd.DataFrame
    remove_from_cart: pd.DataFrame
    page_visit: pd.DataFrame
    search_query: pd.DataFrame

    @classmethod
    def from_disk(cls, data_path_raw, data_path):

        relevant_clients = np.load(os.path.join(data_path_raw, 'input/relevant_clients.npy'))
        product_properties = pd.read_parquet(os.path.join(data_path_raw, 'product_properties.parquet'))

        # Read and process each DataFrame
        product_buy = pd.read_parquet(os.path.join(data_path, 'product_buy.parquet'))
        product_buy = pd.merge(product_buy, product_properties[['sku', 'category', 'price']])

        add_to_cart = pd.read_parquet(os.path.join(data_path, 'add_to_cart.parquet'))
        add_to_cart = pd.merge(add_to_cart, product_properties[['sku', 'category', 'price']])

        remove_from_cart = pd.read_parquet(os.path.join(data_path, 'remove_from_cart.parquet'))
        remove_from_cart = pd.merge(remove_from_cart, product_properties[['sku', 'category', 'price']])

        page_visit = pd.read_parquet(os.path.join(data_path, 'page_visit.parquet'))
        search_query = pd.read_parquet(os.path.join(data_path, 'search_query.parquet'))

        return cls(
            relevant_clients=relevant_clients,
            product_properties=product_properties,
            product_buy=product_buy,
            add_to_cart=add_to_cart,
            remove_from_cart=remove_from_cart,
            page_visit=page_visit,
            search_query=search_query
        )
