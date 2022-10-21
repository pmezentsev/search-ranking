import torch
import typing as t
import numpy as np
import torch.nn.functional as F

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        """
        GaussianKernel doesn't contain trainable params and is used as simple linear operator
        """
        result = torch.exp(-((x-self.mu)**2)/2/self.sigma**2)
        return result


class KNRM(torch.nn.Module):
    """See https://github.com/AdeDZY/K-NRM for the model description"""
    def __init__(self,
                 embedding_matrix: np.ndarray,
                 freeze_embeddings: bool,
                 kernel_num: int = 21,
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001,
                 out_layers: t.List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )
        print("KNRM embeddings is created", flush=True)

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()
        print("KNRM kernels is created", flush=True)
        self.mlp = self._get_mlp()
        print("KNRM mlp is created", flush=True)
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        """
        _get_kernels_layers — forms list of all kernels that are used in algorithm
        """
        kernels = torch.nn.ModuleList()

        step_size = 2/(self.kernel_num-1)
        min_value = step_size / 2 - 1

        for idx in range(self.kernel_num):
            mu = min(min_value + step_size * idx, 1.)
            sigma = self.sigma if mu < 1. else self.exact_sigma
            kernels.append(GaussianKernel(mu, sigma))

        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        """
        _get_mlp — creates output MLP layer on base of kernels. The exact structure depends on out_layers attribute.
        For example for  out_layers = [10, 5] than archetictue will looks like  K->ReLU->10->ReLU->5->ReLU->1
        """
        layers_list = []
        current_dimension = self.kernel_num
        for next_dim in self.out_layers:
            layers_list.append(torch.nn.Linear(current_dimension, next_dim))
            layers_list.append(torch.nn.ReLU())
            current_dimension = next_dim
        layers_list.append(torch.nn.Linear(current_dimension, 1))
        mlp = torch.nn.Sequential(*layers_list)
        return mlp

    def forward(self, input_1: t.Dict[str, torch.Tensor], input_2: t.Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)
        logits_diff = logits_1 - logits_2
        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        """
        _get_matching_matrix — creates interation matrix between words in document and in query.
        It uses cosine similarity as a similarity measurement between vectors
        """
        query = self.embeddings(query)
        doc = self.embeddings(doc)
        query = F.normalize(query.to(torch.float32), dim=2)
        doc = F.normalize(doc.to(torch.float32), dim=2)

        matching_matrix = torch.einsum("ijk,ilk->ijl", query, doc)
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: t.Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out
