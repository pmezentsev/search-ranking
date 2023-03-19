import typing as t
from langdetect import detect
import os
from ranking_model import RankingModel
from candidate_model import CandidateModel
from tokenizer import Tokenizer


class SearchEngine:
    def __init__(self,
                 emb_path_knrm: str = os.environ.get('EMB_PATH_KNRM'),
                 vocab_path: str = os.environ.get('VOCAB_PATH'),
                 mlp_path: str = os.environ.get('MLP_PATH'),
                 glove_embeddings_path: str = os.environ['EMB_PATH_GLOVE'],
                 top_n: int = 10
                 ):
        self.doc_ids = None
        self.doc_texts = None
        self.tokenizer = Tokenizer(vocab_path)
        self.ranking_model = RankingModel(emb_path_knrm, mlp_path)
        self.candidate_model = CandidateModel(glove_embeddings_path)
        self.top_n = top_n

    def update_index(self, documents: t.Dict[str, str]):
        # convert documents into tokens List[List[int]] using tokenizer
        # intiialize candidate model
        # store documents tokens
        self.doc_ids, self.doc_texts = list(zip(*documents.items()))
        doc_tokens = [self.tokenizer.tokenize(t) for t in self.doc_texts]
        clean_doc_tokens = [self.tokenizer.remove_stopwords(t) for t in doc_tokens]
        self.doc_token_ids = [self.tokenizer.token_to_ids(t) for t in doc_tokens]
        index_size = self.candidate_model.update_index(clean_doc_tokens)
        return index_size

    def _process_single_query(self, query):
        language = detect(query)
        if language == 'en':
            tokens = self.tokenizer.tokenize(query)
            clean_tokens = self.tokenizer.remove_stopwords(tokens)
            candidates = self.candidate_model.search([clean_tokens])
            candidates = candidates.reshape(-1).tolist()
            tokens_ids = self.tokenizer.token_to_ids(tokens)
            docs_with_scores = [{'query': query,
                                 'doc_id': self.doc_ids[candidate_idx],
                                 'document': self.doc_texts[candidate_idx],
                                 'score': self.ranking_model.predict(tokens_ids, self.doc_token_ids[candidate_idx])}
                                for candidate_idx in candidates]
            docs_with_scores = sorted(docs_with_scores, key=lambda x: -x['score'])[:self.top_n]
            result = [(doc['doc_id'], doc['document'], doc['score']) for doc in docs_with_scores]
            return True, result
        else:
            return False, None

    def query(self, queries: t.List[str]) -> t.Dict:
        if not self.candidate_model.is_initialized():
            return dict(status='FAISS is not initialized!')
        results = [self._process_single_query(query) for query in queries]
        lang_check, suggestions = list(zip(*results))
        return {
            'lang_check': lang_check,
            'suggestions': suggestions
        }

    def score(self, query: str, doc: str):
        query_tokens_ids = self.tokenizer.text_to_ids(query)
        doc_tokens_ids = self.tokenizer.text_to_ids(doc)
        score = self.ranking_model.predict(query_tokens_ids, doc_tokens_ids)
        return score

