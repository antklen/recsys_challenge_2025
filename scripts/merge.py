import logging
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.inference import get_relevant_clients


def sanitize(
    client_ids: np.ndarray, embeddings: np.ndarray, drop_zeros: bool
) -> tuple[np.ndarray, np.ndarray]:
    nonzero_mask = (embeddings != 0).any(axis=1)

    if nonzero_mask.sum() < embeddings.shape[0]:
        msg = "Detected zero embeddings. "

        if drop_zeros:
            msg += "They will be removed."
            logging.info(msg)
            return client_ids[nonzero_mask], embeddings[nonzero_mask]
        else:
            msg += "Since `drop_zeros=False`, they will be retained."
            logging.warning(msg)

    return client_ids, embeddings


def process_embeddings(embeddings: np.ndarray, processor_config: DictConfig) -> np.ndarray:
    processor = hydra.utils.instantiate(processor_config)
    logging.info(f"Processed with {processor.__class__.__name__}.")
    return processor.fit_transform(embeddings)


def scale_embeddings(embeddings: np.ndarray, scaler_config: DictConfig) -> np.ndarray:
    scaler_config = OmegaConf.to_container(scaler_config, resolve=True)
    apply_globally = scaler_config.pop("global", False)
    scaler = hydra.utils.instantiate(scaler_config)

    if apply_globally:
        logging.info(f"Applied global {scaler.__class__.__name__}.")
        return scaler.fit_transform(embeddings.reshape(-1, 1)).reshape(embeddings.shape)
    else:
        logging.info(f"Applied local {scaler.__class__.__name__}.")
        return scaler.fit_transform(embeddings)


def merge_embeddings(
    client_ids1: np.ndarray,
    embeddings1: np.ndarray,
    client_ids2: np.ndarray,
    embeddings2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    client_ids = np.union1d(client_ids1, client_ids2)

    aligned_embeddings1 = np.full((client_ids.size, embeddings1.shape[1]), np.nan)
    aligned_embeddings2 = np.full((client_ids.size, embeddings2.shape[1]), np.nan)

    aligned_embeddings1[np.searchsorted(client_ids, client_ids1)] = embeddings1
    aligned_embeddings2[np.searchsorted(client_ids, client_ids2)] = embeddings2

    return client_ids, np.hstack((aligned_embeddings1, aligned_embeddings2))


def impute_missing_values(embeddings: np.ndarray, imputer_config: DictConfig) -> np.ndarray:
    imputer = hydra.utils.instantiate(imputer_config)
    logging.info(f"Imputed missing data with {imputer.__class__.__name__}.")
    return imputer.fit_transform(embeddings)


@hydra.main(config_path="config", config_name="best_solution", version_base=None)
def main(config: DictConfig) -> None:
    if config.submit:
        embeddings_path = os.path.join(config.challenge_data_path, "embeddings_submit")
    else:
        embeddings_path = os.path.join(config.challenge_data_path, "embeddings_local")

    # Save config
    save_path = os.path.join(embeddings_path, config.save_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    OmegaConf.save(config, os.path.join(save_path, "config.yaml"))

    # Start pipeline
    merged_client_ids, merged_embeddings = None, None

    for embedding_config in config.embeddings_list:
        client_ids = np.load(os.path.join(embeddings_path, embedding_config.name, "client_ids.npy"))
        embeddings = np.load(os.path.join(embeddings_path, embedding_config.name, "embeddings.npy"))

        assert client_ids.shape[0] == embeddings.shape[0]
        logging.info(f"Loaded embeddings '{embedding_config.name}' of shape {embeddings.shape}.")

        # Check zero embeddings
        client_ids, embeddings = sanitize(client_ids, embeddings, config.drop_zeros)

        # Process embeddings if necessary
        processor_config = embedding_config.get("processor", None)

        if processor_config is not None:
            embeddings = process_embeddings(embeddings, processor_config)

        # Scale embeddings if necessary
        scaler_config = embedding_config.get("scaler", None)

        if scaler_config is not None:
            embeddings = scale_embeddings(embeddings, scaler_config)

        # Merge embeddings
        if merged_client_ids is None:
            merged_client_ids, merged_embeddings = client_ids, embeddings
        else:
            merged_client_ids, merged_embeddings = merge_embeddings(
                merged_client_ids, merged_embeddings, client_ids, embeddings
            )

        logging.info(f"Finished processing embeddings for '{embedding_config.name}'.\n")

    logging.info("Finished merging embeddings.")

    # Add missing clients
    if config.submit:
        relevant_clients = np.load(
            os.path.join(config.challenge_data_path, "raw", "input", "relevant_clients.npy")
        )
        merged_client_ids, merged_embeddings = get_relevant_clients(
            merged_client_ids,
            merged_embeddings,
            relevant_clients,
            add_missing=True,
            fill_value=np.nan,
        )
        logging.info("Added missing clients.")

    # Fill missing values
    merged_embeddings = impute_missing_values(merged_embeddings, config.imputer)

    # Save embeddings and client IDs
    np.save(os.path.join(save_path, "client_ids.npy"), merged_client_ids)
    np.save(os.path.join(save_path, "embeddings.npy"), merged_embeddings.astype("float16"))


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    logging.info(f"Finished job in {end - start:.3f} seconds.")
