import numpy as np


def get_relevant_clients(
    client_ids: np.ndarray,
    embeddings: np.ndarray,
    relevant_clients: np.ndarray,
    add_missing: bool = False,
    fill_value: float = 0.0,
):
    embeddings = embeddings[np.isin(client_ids, relevant_clients)]
    client_ids = client_ids[np.isin(client_ids, relevant_clients)]

    if add_missing:
        relevant_clients_remaining = relevant_clients[~np.isin(relevant_clients, client_ids)]
        client_ids = np.hstack([client_ids, relevant_clients_remaining])
        embeddings = np.vstack(
            [
                embeddings,
                np.full((len(relevant_clients_remaining), embeddings.shape[1]), fill_value),
            ]
        )

    return client_ids, embeddings
