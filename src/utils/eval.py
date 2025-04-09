import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def calculate_relevance(predicted_ids, ground_truth_label, k):

    # Restrict predictions to the top k
    predicted_ids = predicted_ids[:k]

    # Calculate Precision@k
    precision_at_k = 1.0 if ground_truth_label in predicted_ids else 0.0

    # Calculate NDCG@k
    if ground_truth_label in predicted_ids:
        rank = predicted_ids.index(ground_truth_label) + 1
        dcg = 1 / np.log2(rank + 1)  # Discounted Cumulative Gain
    else:
        dcg = 0.0

    # Ideal DCG (IDCG) is always 1 since there's only one relevant item
    idcg = 1.0

    ndcg_at_k = dcg / idcg  # Normalized Discounted Cumulative Gain

    return precision_at_k, ndcg_at_k


def calculate_diversity(product_ids, embeddings, k):
    """Compute diversity (ILD) for a list of candidates using precomputed embeddings."""
    candidate_embeddings = np.array([embeddings[i] for i in product_ids[:k]])
    pairwise_distances = cosine_distances(candidate_embeddings)
    diversity_score = np.sum(pairwise_distances) / (k * (k - 1))
    return diversity_score
