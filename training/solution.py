import time
import typing as t

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from knrm import KNRM
from dataset import TrainDatasetFactory, TestDatasetFactory
from metrics import ndcg_k


class Solution:
    def __init__(self,
                 model: KNRM,
                 train_factory: TrainDatasetFactory,
                 test_factory: TestDatasetFactory,
                 train_lr: float = 0.01,
                 change_train_loader_ep: int = 6,
                 early_stopping_epochs=10,
                 ):
        """
        :param glue_qqp_dir:  directory with glue quora questions pairs dataset
        :param glove_vectors_path:  path to gloves vectors (dim=50)
        :param min_token_occurancies: how much time word should appear in training to be filtered as input feature
        :param emb_rand_uni_bound:  half width of the interal for random uniform embeddings generation for unknown words
        :param freeze_knrm_embeddings: train embeddings during the model training
        :param knrm_kernel_num: number of kernels in knrm
        :param knrm_out_mlp: output MLP configuration
        :param dataloader_bs: batch size during the model training and evaluation
        :param train_lr: Train learning rate
        :param change_train_loader_ep: how often to change train dataset
        """
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep
        self.model = model
        self.early_stopping_epochs = early_stopping_epochs

        self.train_factory = train_factory
        self.test_factory = test_factory
        self.test_dataloader = test_factory.get_dataloader()

    def _evaluate(self, model: torch.nn.Module = None, data: torch.utils.data.DataLoader = None) -> float:
        labels_and_groups = data.dataset.samples_list
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        pred = [model.predict(batch).detach().numpy() for batch, _ in data]
        pred = np.concatenate(pred, axis=0)
        labels_and_groups['pred'] = pred

        ndcg_list = [ndcg_k(df.rel.values, df.pred.values) for _, df in labels_and_groups.groupby('left_id')]
        mean_ndcg = float(np.mean(ndcg_list))
        return mean_ndcg

    def evaluate(self):
        return self._evaluate(self.model, self.test_dataloader)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        train_dataloader = None

        for epoch in tqdm(range(n_epochs)):
            if epoch % self.change_train_loader_ep == 0:
                train_dataloader = self.train_factory.get_dataloader()
            self.model.train()
            for batch_idx, (left_batch, right_batch, y_true) in enumerate(train_dataloader):
                left_batch = {k: v for k, v in left_batch.items()}
                right_batch = {k: v for k, v in right_batch.items()}
                opt.zero_grad()
                y_pred = self.model.forward(left_batch, right_batch)
                query_loss = criterion(y_pred, y_true)
                query_loss.backward()
                opt.step()
