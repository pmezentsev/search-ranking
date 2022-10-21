import faiss
import numpy as np
import time
import typing as t

# OPQM_D,...,PQMx4fsr
# OPQ5,IVF100,PQ20
# IVF100,PQ10
# OPQ5,IVF800,PQ5
class CandidateModel:
    def __init__(self,
                 glove_embeddings_path,
                 index_config: str = "IVF800,PQ10",
                 top_n: int = 20,
                 n_probe: int = 10):
        self.index = None
        self.index_config = index_config
        self.vocab, self.embeddings = self.read_glove_embeddings(glove_embeddings_path)
        self.top_n = top_n
        self.n_probe = n_probe

    def is_initialized(self):
        return self.index is not None

    @staticmethod
    def read_glove_embeddings(file_path: str):
        with open(file_path, 'r') as f:
            glove_list = f.readlines()

        glove_list = map(lambda x: x.strip().split(" ", 1), glove_list)
        glove_list = map(lambda x: (x[0], x[1].split(" ")), glove_list)
        glove_list = map(lambda x: (x[0], np.array(list(map(float, x[1])), dtype=np.float16)), glove_list)
        word_list, emb_list = list(zip(*glove_list))
        vocab = {w: idx for idx, w in enumerate(word_list)}
        emb = np.vstack(emb_list)
        return vocab, emb

    def make_text_repr(self, tokens_list):
        token_repr = [self.embeddings[self.vocab[t], :] for t in tokens_list if t in self.vocab]
        if len(token_repr) == 0:
            return np.zeros(shape=self.embeddings.shape[1]).astype(np.float32)
        token_repr = np.vstack(token_repr)
        token_repr = np.mean(token_repr, axis=0).astype(np.float32)
        return token_repr

    def update_index(self, tokenized_documents_list: t.List[t.List[str]]):
        # start = time.time()
        documents_repr = np.vstack([self.make_text_repr(d) for d in tokenized_documents_list])
        index = faiss.index_factory(documents_repr.shape[1], self.index_config)
        index.train(documents_repr)
        index.add(documents_repr)
        index.nprobe = self.n_probe
        # print(f"Candidate model index updated in {time.time() -start:.2f}s")
        self.index = index
        return self.index.ntotal

    def search(self, tokenized_queries_list: t.List[t.List[str]]) -> np.ndarray:
        # start = time.time()
        assert self.index is not None, 'Candidate model index is not initialized'
        queries_repr = np.vstack([self.make_text_repr(d) for d in tokenized_queries_list])
        documents, indexes = self.index.search(queries_repr, self.top_n)
        # print(f"Candidate model search of {len(tokenized_queries_list)} "
        #             f"documents perfromed in {time.time() -start:.2f}s")
        return indexes
