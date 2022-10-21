import pandas as pd
import typing as t
import numpy as np
import re


def tokenize_text(inp_str: str) -> t.List[str]:
    """
    полный препроцессинг строки. Должно включать в себя обработку пунктуации и приведение к нижнему регистру,
    а в качестве токенизации (разбиения предложения на слова или их усеченную версию — токены) необходимо
    использовать метод nltk.word_tokenize из библиотеки nltk. На выходе — лист со строками (токенами).
    """
    inp_str = re.sub('\\W+', ' ', inp_str)
    inp_str = inp_str.strip()
    inp_str = inp_str.lower()
    tokens = inp_str.split(" ")
    return tokens


def build_vocabulary(df: pd.DataFrame, min_occurancies: int = 1) -> t.Dict[str, int]:
    """
    params:
     list_of_df - dataframe which text to be preprocesse to get all tokens.
     min_occurancies - minimum number of occurencies for element to be added to the final list
    result - list of tokens, that will be used to make embeddings
    """

    all_texts = pd.concat([
        df[col]
        for col in df.columns
        if "text" in col
    ])

    all_texts = all_texts.drop_duplicates()
    all_texts = all_texts.str.lower()
    all_texts = all_texts.str.replace('\\W+', ' ', regex=True).str.strip().str.split(' ')
    all_texts = all_texts.explode().value_counts()
    vocabulary_list = all_texts[all_texts >= min_occurancies].index.tolist()
    vocabulary_list = ['PAD', 'OOV'] + vocabulary_list
    return vocabulary_list


def read_glove_embeddings(file_path: str) -> t.Dict[str, t.List[str]]:
    """
    считывание файла эмбеддингов в словарь, где ключ — это слово, а значение — это вектор эмбеддинга
    (можно не приводить к float-значениям).
    """
    with open(file_path, 'r') as f:
        glove_list = f.readlines()

    glove_list = map(lambda x: x.strip().split(" "), glove_list)
    glove_list = map(lambda x: (x[0], x[1:]), glove_list)
    return dict(glove_list)


# def create_glove_emb_from_file(file_path: str,
#                                inner_keys: t.List[str],
#                                random_seed: int,
#                                rand_uni_bound: float
#                                ) -> t.Tuple[np.ndarray, t.Dict[str, int]]:
#     """
#     Creates glove embeddings and vocabularies
#     """
#     print("\t\tcreate_glove_emb_from_file")
#     np.random.seed(random_seed)
#     glove_dict = read_glove_embeddings(file_path)
#     print("\t\tglove_dict is read from file")
#     embedding_size = len(next(iter(glove_dict.values())))
#     vocab = {token: idx for idx, token in (['PAD', 'OOV'] + inner_keys)}
#     make_random_vector = lambda: np.random.uniform(-rand_uni_bound, rand_uni_bound, embedding_size)
#
#     # add PAD and OOV value
#     embeddings_list = [np.zeros(embedding_size), make_random_vector()]
#     for key in inner_keys:
#         emb = np.array(glove_dict.get(key, make_random_vector()), dtype=np.float)
#         embeddings_list.append(emb)
#
#     embeddings = np.stack(embeddings_list)
#     print("\t\tglove_dict is embeddings matrix is created")
#     return embeddings, vocab


def get_idx_to_text_mapping(inp_df: pd.DataFrame) -> t.Dict[str, str]:
    left_dict = (
        inp_df
        [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
        ['text_left']
            .to_dict()
    )
    right_dict = (
        inp_df
        [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
        ['text_right']
            .to_dict()
    )
    left_dict.update(right_dict)
    return left_dict


def create_word_embeddings(
        glove_vector_path: str,
        vocabulary: t.List[str],
        random_init_max_value: float = 0.2):
    assert vocabulary[0] == 'PAD'
    assert vocabulary[1] == 'OOV'

    glove_dict = read_glove_embeddings(glove_vector_path)
    embedding_size = len(list(glove_dict.values())[0])
    make_random_vector = lambda: np.random.uniform(-random_init_max_value, random_init_max_value, embedding_size)

    # add PAD and OOV value
    embeddings_list = [np.zeros(embedding_size), make_random_vector()]
    for key in vocabulary:
        emb = np.array(glove_dict.get(key, make_random_vector()), dtype=np.float)
        embeddings_list.append(emb)

    embeddings = np.stack(embeddings_list)
    return embeddings

#
#
#
# class Embeddings:
#     def __init__(self,
#                  glove_vector_path: str,
#                  vocabulary_dict: t.Dict[str, int],
#                  emb_rand_uni_bound: float = 0.2,
#                  random_seed: int = 0
#                  ):
#         dataset_tokens = get_all_tokens(dataset, min_token_occurences)
#         emb_matrix, vocab, unk_words = create_glove_emb_from_file(
#             glove_vector_path, dataset_tokens, random_seed, emb_rand_uni_bound)
#         self.embedding_matrix = emb_matrix
#         self.vocabulary = vocab
#         self.unk_words = unk_words
#
#
#
#
