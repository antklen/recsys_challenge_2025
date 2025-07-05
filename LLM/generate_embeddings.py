import os

import hydra
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt


@hydra.main(version_base=None, config_path="config", config_name="generate_embeddings")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    if hasattr(config, "cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    data_path = os.path.join(config.data_path, "raw")

    if not config.submit:
        data_path = os.path.join(data_path, "input")

    data_path = os.path.join(data_path, "text")

    print("Loading data.")
    dataset = load_dataset(data_path, data_files=config.data_files, split="train")

    print("Loading model.")
    model = LLM(
        model=config.model_path,
        tokenizer=config.tokenizer_path,
        task="embed",
        trust_remote_code=True,
        **config.vllm_model_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    print("Starting inference.")
    client_ids, embeddings = inference_embeddings(
        dataset, model, tokenizer, batch_size=config.batch_size, max_length=config.max_length
    )

    if config.submit:
        save_path = os.path.join(config.data_path, "embeddings_submit")
    else:
        save_path = os.path.join(config.data_path, "embeddings_local")

    save_path = os.path.join(save_path, config.save_dir_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "embeddings.npy"), embeddings.astype("float16"))
    np.save(os.path.join(save_path, "client_ids.npy"), np.array(client_ids).astype("int"))

    print(f"Save embeddings on {save_path}")


def inference_embeddings(dataset, model, tokenizer, batch_size=16, max_length=8192):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    client_ids, embeddings = [], []

    for batch in tqdm(dataloader, total=len(dataloader)):
        client_ids.extend(batch["client_id"])
        emb = embed_texts_vllm_by_tokens(batch["text"], model, tokenizer, max_length)
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings)

    return client_ids, embeddings


def embed_texts_vllm_by_tokens(texts, llm, tokenizer, max_length=8192):
    inputs = tokenizer(
        texts, return_tensors=None, padding=False, truncation=True, max_length=max_length
    ).to("cuda")

    token_prompt = []
    for input_ids in inputs["input_ids"]:
        token_prompt.append(TokensPrompt(prompt_token_ids=input_ids))

    outputs = llm.embed(token_prompt, use_tqdm=False)
    embeddings = np.array([output.outputs.embedding for output in outputs])

    return embeddings


if __name__ == "__main__":
    main()
