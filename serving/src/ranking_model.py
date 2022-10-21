import typing as t
import torch


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        """
        GaussianKernel не содержит обучаемых параметров и служит простым нелинейным оператором —
        необходимо перевести формулу из лекции (или Википедии, если удобно) в метод forward класса.
        Параметр mu отвечает за "среднее" ядра, точку внимания, sigma — за ширину “бина” (см. лекцию).
        """
        result = torch.exp(-((x-self.mu)**2)/2/self.sigma**2)
        return result


class KNRM(torch.nn.Module):
    def __init__(self,
                 embedding_dict: t.Dict[str, torch.Tensor],
                 mlp_dict: t.Dict[str, torch.Tensor],
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001):
        super().__init__()
        self.sigma = sigma
        self.exact_sigma = exact_sigma

        self.embeddings = torch.nn.Embedding.from_pretrained(
            embedding_dict['weight'],
            freeze=True,
            padding_idx=0
        )

        mlp_dimentions = [mlp_dict[k].shape[1] for k in mlp_dict.keys() if 'weight' in k]
        self.kernel_num = mlp_dimentions[0]
        self.out_layers = mlp_dimentions[1:]
        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(mlp_dict)

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()

        step_size = 2/(self.kernel_num-1)
        min_value = step_size / 2 - 1

        for idx in range(self.kernel_num):
            mu = min(min_value + step_size * idx, 1.)
            sigma = self.sigma if mu < 1. else self.exact_sigma
            kernels.append(GaussianKernel(mu, sigma))

        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
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
        _get_matching_matrix — формирует матрицу взаимодействия “каждый-с-каждым” между словами одного и второго
        вопроса (запрос и документ). В качестве меры используется косинусная схожесть (cosine similarity) между
        эмбеддингами отдельных токенов.
        """
        query = self.embeddings(query)
        doc = self.embeddings(doc)
        query = torch.nn.functional.normalize(query.to(torch.float32), dim=2)
        doc = torch.nn.functional.normalize(doc.to(torch.float32), dim=2)

        matching_matrix = torch.einsum("ijk,ilk->ijl", query, doc)
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        """
        _apply_kernels — применяет ядра к matching_matrix согласно формуле и иллюстрации из презентации. Метод
        реализован, постарайтесь разобраться в том, что означают суммы в формуле и что получается на выходе.
        """
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: t.Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """
        Методы forward и predict уже реализованы. Обратите внимание на указанные размерности в методе predict.
        По этим методам ясно, что KNRM будет обучаться в PairWise-режиме. Для этого необходимо соответственно
        подготовить данные (см. следующий блок).

        """
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingModel:
    def __init__(self,
                 embeddings_knrm_path,
                 mlp_path: str):
        with open(embeddings_knrm_path, 'rb') as f:
            emb_dict = torch.load(f,  map_location=torch.device('cpu'))

        with open(mlp_path, 'rb') as f:
            mlp_dict = torch.load(f,  map_location=torch.device('cpu'))

        self.model = KNRM(emb_dict, mlp_dict)

    def predict(self, query_tokens, document_tokens):
        query_tokens = torch.tensor(query_tokens).reshape(1, -1)
        document_tokens = torch.tensor(document_tokens).reshape(1, -1)
        prediction = self.model.predict({'query': query_tokens, 'document': document_tokens})
        prediction = prediction.item()
        return prediction
