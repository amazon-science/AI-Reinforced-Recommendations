import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def setup_model_and_tokenizer(model_name='sentence-transformers/all-mpnet-base-v2'):
    """Initialize the model and tokenizer based on specified model name."""
    print(f'Loading model {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for training.')
            model = torch.nn.DataParallel(model)
    return model, tokenizer, device


def encode_in_batches(corpus, model, tokenizer, device, batch_size=128):
    """Encode the corpus in batches and return embeddings."""

    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i + batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_embeddings = model.module(**encoded_input) if hasattr(model, 'module') else model(**encoded_input)
            batch_embeddings = _mean_pooling(batch_embeddings, encoded_input['attention_mask'])
            embeddings.extend(batch_embeddings.cpu().numpy() if tokenizer else batch_embeddings)

    return np.array(embeddings)


def save_embeddings(embeddings, product_ids, save_path):
    """Save embeddings and product IDs to files."""
    assert len(embeddings) == len(product_ids)
    np.save(save_path, embeddings)
    pd.Series(product_ids).to_csv(save_path.replace('.npy', '_ids.csv'), index=False, header=False)
    print(f"Embeddings and product IDs saved to {save_path} and {save_path.replace('.npy', '_ids.csv')}.")


def load_embeddings(save_path):
    """Load embeddings and product IDs from files."""
    id_path = save_path.replace('.npy', '_ids.csv')
    embeddings = np.load(save_path)
    product_ids = pd.read_csv(id_path, header=None).squeeze().tolist()
    return {pid: emb for pid, emb in zip(product_ids, embeddings)}


if __name__ == '__main__':
    model, tokenizer, device = setup_model_and_tokenizer()

    CATEGORY = 'All_Beauty'
    meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{CATEGORY}", split="full", trust_remote_code=True).to_pandas()
    item_id2name = dict(zip(meta.parent_asin, meta.title))

    corpus = list(item_id2name.values())
    embeddings = encode_in_batches(corpus, model, tokenizer, device)
    save_embeddings(embeddings, list(item_id2name.keys()), f'dataset/amazon-review-{CATEGORY}-cans10/embeddings.npy')
