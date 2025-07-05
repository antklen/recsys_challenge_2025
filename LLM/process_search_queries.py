import json
import os

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@hydra.main(version_base=None, config_path="config", config_name="process_search_queries")
def main(config):
    data_path = os.path.join(config.data_path, "raw")
    if not config.submit:
        data_path = os.path.join(data_path, "input")

    rel_clients = np.load(os.path.join(config.data_path, "raw", "input", "relevant_clients.npy"))

    print("Preeprocessing data")
    data = pd.read_parquet(os.path.join(data_path, "search_query.parquet"))
    data = data[data["client_id"].isin(rel_clients)]
    data["query"] = data["query"].apply(lambda x: x[1:-1].split())
    data["query"] = data["query"].apply(lambda x: np.array(list(map(int, x))))

    user_id = data["client_id"].values
    timestamp = data["timestamp"].values
    table = pd.DataFrame(np.array(data["query"].values.tolist()))
    table["timestamp"] = timestamp
    table["client_id"] = user_id

    print(f"Shape - {table.shape}")

    save_path = os.path.join(data_path, "text")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, config.save_file_name), "w") as file:
        for i in tqdm(table["client_id"].unique()):
            subset = table[table["client_id"] == i]
            clients, text = table_2_text(subset, config.max_length)
            entry = {"client_id": str(clients), "text": text}
            json.dump(entry, file, cls=NpEncoder)
            file.write("\n")


def table_2_text(trx, max_len=90):
    client_id = trx["client_id"].iloc[0]
    trx = trx.drop(columns="client_id")

    result = trx.tail(max_len)
    return client_id, result.to_string(index=False)


if __name__ == "__main__":
    main()
