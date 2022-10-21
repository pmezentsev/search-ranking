import numpy as np


def dcg_k(ys_true: np.array, ys_pred: np.array, ndcg_top_k) -> float:
    indexes = np.argsort(-ys_pred)
    ys_true = ys_true[indexes][:ndcg_top_k]
    gain = 2 ** ys_true - 1
    discount = np.arange(0, ys_true.shape[0], dtype=np.float)
    discount = np.log2(discount + 2.)
    discounted_gain = gain / discount
    return discounted_gain.sum().item()


def ndcg_k(ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
    dcg_score = dcg_k(ys_true, ys_pred, ndcg_top_k)
    ideal_dcg_score = dcg_k(ys_true, ys_true, ndcg_top_k)
    if ideal_dcg_score == 0 or np.isnan(ideal_dcg_score) or np.isnan(dcg_score):
        return 0
    return dcg_score / ideal_dcg_score
