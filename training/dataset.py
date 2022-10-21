import torch
import typing as t
import pandas as pd
import numpy as np
import random
import aux


class RankingDataset(torch.utils.data.Dataset):
    """
    Data wrappers to iterate over data.
    """

    def __init__(self,
                 samples_list: t.List[t.List[t.Union[str, float]]],
                 idx_to_text_mapping: t.Dict[str, str],
                 vocab: t.Dict[str, int],
                 oov_val: int,
                 tokenize_func: t.Callable,
                 out_list_len: int = 30):
        self.samples_list = samples_list
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.tokenize_func = tokenize_func
        self.out_list_len = out_list_len

    def __len__(self):
        return len(self.samples_list)

    def _pad_list(self, token_ids_list):
        token_ids_list = token_ids_list[:self.out_list_len]
        pad_len = self.out_list_len - len(token_ids_list)
        return token_ids_list + [0] * pad_len

    def _tokenized_text_to_index(self, tokenized_text: t.List[str]) -> t.List[int]:
        tokens_ids = [self.vocab.get(token, self.oov_val) for token in tokenized_text]
        tokens_ids = self._pad_list(tokens_ids)
        return tokens_ids

    def _convert_text_idx_to_token_idxs(self, idx: int) -> t.List[int]:
        text = self.idx_to_text_mapping[idx]
        tokenized_text = self.tokenize_func(text)
        tokens_ids = self._tokenized_text_to_index(tokenized_text)
        return tokens_ids

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    """
    Provides triplets for training.
    """

    def __getitem__(self, idx):
        triplet_index = self.samples_list[idx]
        query_idx, left_idx, right_idx, score = triplet_index
        query_tokens = self._convert_text_idx_to_token_idxs(query_idx)
        left_tokens = self._convert_text_idx_to_token_idxs(left_idx)
        right_tokens = self._convert_text_idx_to_token_idxs(right_idx)

        return (
            {'query': query_tokens, 'document': left_tokens},
            {'query': query_tokens, 'document': right_tokens},
            score
        )


class ValPairsDataset(RankingDataset):
    """
    Provides pairs for validation.
    """

    def __getitem__(self, idx):
        pair_index = self.samples_list[idx]
        query_idx, document_idx, score = pair_index
        sample = {
            'query': self._convert_text_idx_to_token_idxs(query_idx),
            'document': self._convert_text_idx_to_token_idxs(document_idx)
        }

        return sample, score


class BufferedRandomValuesGenerator:
    def __init__(self, buffer_size):
        self.offset = 0
        self.buffer_size = buffer_size
        self.random_values_buffer = np.random.uniform(0, 1, size=buffer_size)

    def uniform(self):
        value = self.random_values_buffer[self.offset]
        self.offset = (self.offset + 1) % self.buffer_size
        return value

    def choice(self, lst):
        uniform_0_1 = self.uniform()
        random_idx = int(uniform_0_1 * len(lst))
        return lst[random_idx]


def collate_pairs_fn(batch_objs: t.List[t.Union[t.Dict[str, torch.Tensor], torch.FloatTensor]]
                     ) -> t.Tuple[t.Dict[str, torch.Tensor], torch.FloatTensor]:
    """
    Collects batches from several training samples for KNRM. Get list of datasets outputs and packs them
    into one dictionary with tensors as values
    """
    queries = torch.LongTensor([elem[0]['query'] for elem in batch_objs])
    documents = torch.LongTensor([elem[0]['document'] for elem in batch_objs])
    labels = torch.FloatTensor([[elem[-1]] for elem in batch_objs])
    query_document = {'query': queries, 'document': documents}
    return query_document, labels


def collate_triplets_fn(batch_objs: t.List[t.Union[t.Dict[str, torch.Tensor], torch.FloatTensor]]):
    """
    Collects batches from several training samples for KNRM. Get list of datasets outputs and packs them
    into one dictionary with tensors as values
    """
    left_queries = torch.LongTensor([elem[0]['query'] for elem in batch_objs])
    left_documents = torch.LongTensor([elem[0]['document'] for elem in batch_objs])
    left_query_document = {'query': left_queries, 'document': left_documents}

    right_queries = torch.LongTensor([elem[1]['query'] for elem in batch_objs])
    right_documents = torch.LongTensor([elem[1]['document'] for elem in batch_objs])
    right_query_document = {'query': right_queries, 'document': right_documents}

    labels = torch.FloatTensor([[elem[-1]] for elem in batch_objs])
    return left_query_document, right_query_document, labels


def sample_data_for_train_iter(inp_df: pd.DataFrame,
                               sample_size: int = 10,
                               min_group_size: int = 2,
                               zeros_frac: float = 0.5
                               ) -> t.List[t.List[t.Union[str, float]]]:
    """
    The idea of given implementation of data sampling - is to generate uniformly distributed samples
    of triplets from dataset.

    :param sample_size:  Number of triplets to generate
    :param min_group_size:  Minimum group size of triples to use in generation
    :param zeros_frac: Fraction of pairs with zero relevance
    """
    # this part is copied from create_val_pairs and get groups of answers to request id_left
    # this is bigger than given threshold
    inp_df_select = inp_df[['id_left', 'id_right', 'label']]
    inf_df_group_sizes = inp_df_select.groupby('id_left').size()
    glue_dev_leftids_to_use = list(
        inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
    groups = inp_df_select[inp_df_select.id_left.isin(
        glue_dev_leftids_to_use)].groupby('id_left')
    groups = dict(list(groups))
    query_ids = list(groups.keys())

    all_ids = list(set(inp_df['id_left']).union(set(inp_df['id_right'])))
    if sample_size is None:
        sample_size = len(inp_df)

    out_pairs = []
    max_iterations = sample_size * 3
    # random numbers generation is the slowest operation in all logic. That's why
    # it was implemented buffered random numbers generator, that generates an array
    # with random numbers and returns them one by one during the sampling
    random_generator = BufferedRandomValuesGenerator(sample_size * 6)

    def generate_sample(query_df):
        if random_generator.uniform() > zeros_frac:
            _, row = random_generator.choice(list(query_df.iterrows()))
            return row.id_right, row.label + 1
        else:
            used_ids = query_df.id_right.tolist() + [query_df.id_left.iloc[0]]
            while True:
                doc_id = random_generator.choice(all_ids)
                if doc_id not in used_ids:
                    return doc_id, 0

    for _ in range(max_iterations):
        query_id = random.choice(query_ids)
        df = groups[query_id]
        first_idx, first_score = generate_sample(df)
        second_idx, second_score = generate_sample(df)
        if first_idx != second_idx and first_score != second_score:
            out_pairs.append([query_id, first_idx, second_idx, float(first_score > second_score)])
            if len(out_pairs) == sample_size:
                return out_pairs

    return out_pairs


def create_val_pairs(data_df: pd.DataFrame,
                     padding_size: int = 15,
                     min_group_size: int = 2) -> t.List[t.List[t.Union[str, float]]]:
    data_df = data_df[['id_left', 'id_right', 'label']]
    group_sizes = data_df.groupby('id_left').size()
    groups_to_use = group_sizes[group_sizes >= min_group_size].index.tolist()

    all_ids = set(data_df['id_left']).union(set(data_df['id_right']))
    out_pairs_df = data_df.copy()
    out_pairs_df = out_pairs_df[out_pairs_df.id_left.isin(groups_to_use)]
    out_pairs_df['label'] += 1

    out_pairs_list = out_pairs_df.values.tolist()
    random_generator = BufferedRandomValuesGenerator(len(data_df))

    for group_id in group_sizes[group_sizes < padding_size].index:
        df = data_df[data_df.id_left == group_id]
        group_ids = set(df.id_right.values).union({group_id})
        num_pad_items = padding_size - len(df)
        unrelevant_ids = list(all_ids - group_ids)
        # pad_sample = np.random.choice(, num_pad_items, replace=False).tolist()
        # out_pairs_df.append({
        #     'id_left': [group_id] * num_pad_items,
        #     'id_right': pad_sample,
        #     'label': [0] * num_pad_items
        # }, ignore_index=True)
        for i in range(num_pad_items):
            right_id = random_generator.choice(unrelevant_ids)
            out_pairs_list.append([group_id, right_id, 0])
    return out_pairs_list


def make_train_dataloader(train_df, vocabulary: t.List[str], train_sample_size=10_000, zeros_frac=0.5, batch_size=1024):
        idx_to_text_mapping = aux.get_idx_to_text_mapping(train_df)
        vocabulary_dict = {word: idx for idx, word in enumerate(vocabulary)}
        train_data = sample_data_for_train_iter(train_df,
                                                sample_size=train_sample_size,
                                                zeros_frac=zeros_frac)
        train_dataset = TrainTripletsDataset(train_data,
                                             idx_to_text_mapping,
                                             vocabulary_dict,
                                             oov_val=vocabulary_dict['OOV'],
                                             tokenize_func=aux.tokenize_text)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=0,
            collate_fn=collate_triplets_fn, shuffle=True)
        return train_dataloader


def make_test_dataloader(test_df, vocabulary: t.List[str], batch_size=1024):
        test_pairs = create_val_pairs(test_df)
        idx_to_text_mapping = aux.get_idx_to_text_mapping(test_df)
        vocabulary_dict = {word: idx for idx, word in enumerate(vocabulary)}

        val_dataset = ValPairsDataset(test_pairs,
                                           idx_to_text_mapping,
                                           vocab=vocabulary_dict,
                                           oov_val=vocabulary_dict['OOV'],
                                           tokenize_func=aux.tokenize_text)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=0,
            collate_fn=collate_pairs_fn, shuffle=False)
        return val_dataloader
