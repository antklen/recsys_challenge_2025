import os

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig


def parse_codes(event_data: pl.DataFrame, column: str) -> pl.DataFrame:
    return event_data.with_columns(pl.col(column).str.extract_all(r"\d+").list.join(" "))


def convert_to_text(event_data: pl.DataFrame, event_type: str) -> pl.DataFrame:
    text = "event type: " + event_type.replace("_", " ") + "; "

    for column in event_data.columns:
        if column == "client_id":
            continue
        elif column == "timestamp":
            text += f"{column}: " + pl.col(column).dt.strftime("%Y-%m-%d %H:%M") + "; "
        else:
            text += f"{column}: " + pl.col(column).cast(pl.String) + "; "

    return event_data.with_columns(text.str.strip_chars().alias("text"))


@hydra.main(config_path="config", config_name="process_all_interactions", version_base=None)
def main(config: DictConfig) -> None:
    DATASET_PATH = os.path.join(config.challenge_data_path, "raw")
    SUFFIX = "" if config.submit else "input"

    # Load relevant clients
    relevant_clients = np.load(os.path.join(DATASET_PATH, "input", "relevant_clients.npy"))

    # Load & preprocess product properties
    product_properties = pl.read_parquet(os.path.join(DATASET_PATH, "product_properties.parquet"))
    product_properties = parse_codes(product_properties, "name")

    # Load & preprocess interaction data
    data = []

    for event_type in (
        "add_to_cart",
        "remove_from_cart",
        "product_buy",
        "page_visit",
        "search_query",
    ):
        event_data = (
            pl.read_parquet(os.path.join(DATASET_PATH, SUFFIX, f"{event_type}.parquet"))
            # Filter out non-relevant clients
            .filter(pl.col("client_id").is_in(relevant_clients))
            # Drop duplicates
            .sort("timestamp")
            .unique(pl.exclude("timestamp"), keep="last")
        )

        if event_type == "search_query":
            # Parse quantized search queries
            event_data = parse_codes(event_data, "query")
        elif event_type in {"add_to_cart", "remove_from_cart", "product_buy"}:
            # Add product metadata & rename columns
            event_data = event_data.join(product_properties, on="sku", how="left")
            event_data = event_data.rename({"sku": "item"})

        data.append(convert_to_text(event_data, event_type)["client_id", "timestamp", "text"])

    # Concatenate event data, truncate to `max_length` & aggregate texts
    data = (
        pl.concat(data)
        # Truncate to `max_length`
        .sort("timestamp", descending=True)
        .with_columns(pl.col("client_id").cum_count().over("client_id").alias("position"))
        .filter(pl.col("position") <= config.max_length)
        .drop("position")
        # Aggregate texts
        .group_by("client_id")
        .agg(pl.col("text"))
        .with_columns(pl.col("text").list.join("\n"))
    )

    # Save output
    SAVE_PATH = os.path.join(DATASET_PATH, SUFFIX, "text")

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    data.write_parquet(os.path.join(SAVE_PATH, config.save_filename))


if __name__ == "__main__":
    main()
