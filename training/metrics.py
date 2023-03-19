import numpy as np
import pandas as pd
import torch

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


def evaluate(model: torch.nn.Module = None, data: torch.utils.data.DataLoader = None) -> float:
    labels_and_groups = data.dataset.samples_list
    labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

    pred = [model.predict(batch).detach().numpy() for batch, _ in data]
    pred = np.concatenate(pred, axis=0)
    labels_and_groups['pred'] = pred

    ndcg_list = [ndcg_k(df.rel.values, df.pred.values) for _, df in labels_and_groups.groupby('left_id')]
    mean_ndcg = float(np.mean(ndcg_list))
    return mean_ndcg
