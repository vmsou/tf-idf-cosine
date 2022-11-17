from __future__ import annotations

import math
import numpy as np
from typing import Set, List, Generic, TypeVar, Iterable, Dict, Final

import pandas as pd
import spacy
from spacy.tokens.doc import Doc

from corpus import NLP

# Type Alias
_T = TypeVar("_T")


def word_tokenize(text: str) -> List[str]:
    """ Separates words from text. """
    doc: spacy.tokens.doc.Doc = NLP(text)
    return [token.text for token in doc if not token.is_punct]


def cosine_similarity(a: List[int], b: List[int]) -> float:
    value: float = 0
    try:
        value = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except (FloatingPointError, RuntimeWarning):
        pass
    return value if (not np.isnan(value)) else 0.0


class Vocabulary(Generic[_T], Iterable):
    def __init__(self):
        self.data: List[_T] = []
        self.unique: Set[_T] = set()
        self.position: Dict[_T, int] = dict()

    def __str__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def index(self, item: _T):
        return self.position[item]

    def add(self, __o: _T) -> None:
        """ Appends an unique element to List. """
        if __o in self.unique: return
        self.position[__o] = len(self.data)
        self.unique.add(__o)
        self.data.append(__o)

    def union(self, vec: Iterable[_T]) -> None:
        """ Adds all elements from vec to self. """
        for element in vec:
            self.add(element)

    def vectorize(self, text: str) -> List[int]:
        vector: List[int] = [0 for _ in range(len(self.data))]
        for word in word_tokenize(text):
            vector[self.index(word)] += 1
        return vector

    def to_matrix(self, sentences: List[str]) -> pd.DataFrame:
        """ Converts data to Document-term matrix. """
        matrix: pd.DataFrame = pd.DataFrame(columns=self.data)
        for i in range(len(sentences)):
            sentence: str = sentences[i]
            vector: List[int] = self.vectorize(sentence)
            matrix.loc[i + 1] = vector
        return matrix

    def to_tf(self, sentences: List[str]) -> pd.DataFrame:
        """ Converts data to Term Frequency matrix. """
        def tf(t: int, d: int) -> float:
            if d == 0: return 0
            return t / d

        matrix: pd.DataFrame = pd.DataFrame(columns=self.data)
        for i in range(len(sentences)):
            sentence: str = sentences[i]
            vector: List[int] = self.vectorize(sentence)
            total: int = sum(vector)
            matrix.loc[i + 1] = [tf(t, total) for t in vector]
        return matrix

    def to_idf(self, sentences: List[str]) -> pd.DataFrame:
        """ Converts data to Inverse Document Frequency matrix. """
        n_doc = len(sentences)

        def idf(t: int) -> float:
            if t == 0: return 0.0
            return math.log10(n_doc / t)

        matrix: pd.DataFrame = pd.DataFrame(columns=self.data)

        for i in range(len(sentences)):
            sentence: str = sentences[i]
            vector: List[int] = self.vectorize(sentence)
            matrix.loc[i + 1] = [idf(t) for t in vector]
        return matrix

    def to_tf_idf(self, sentences: List[str]) -> pd.DataFrame:
        """ Converts data to Term Frequency-Inverse Document Frequency matrix. """
        N: Final[int] = len(sentences)

        def tf(t: int, d: int) -> float:
            if d == 0: return 0.0
            return t / d

        def idf(t: int) -> float:
            if t == 0: return 0.0
            return math.log10(N / t)

        def tf_idf(t: int, d: int) -> float: return tf(t, d) * idf(t)

        # tf_matrix: pd.DataFrame = self.to_tf(sentences)
        # idf_matrix: pd.DataFrame = self.to_idf(sentences)
        matrix: pd.DataFrame = pd.DataFrame(columns=self.data)
        for i in range(len(sentences)):
            sentence: str = sentences[i]
            vector: List[int] = self.vectorize(sentence)
            total: int = sum(vector)
            matrix.loc[i + 1] = [tf_idf(t, total) for t in vector]

        # return pd.DataFrame(tf_matrix.values * idf_matrix.values, columns=tf_matrix.columns, index=tf_matrix.index)
        return matrix

    def to_similarity(self, sentences: List[str]) -> pd.DataFrame:
        """ Converts sentences to distance matrix utilizing cosine similarity as distance. """
        tf_idf_matrix: pd.DataFrame = self.to_tf_idf(sentences)
        vectors: List[List[int]] = tf_idf_matrix.loc[:, :].values.tolist()
        distance_matrix: pd.DataFrame = pd.DataFrame(columns=range(1, len(vectors)+1), index=range(1, len(vectors)+1))
        for i in distance_matrix.index:
            for j in distance_matrix.columns:
                distance_matrix.loc[i, j] = cosine_similarity(vectors[i-1], vectors[j-1])
        return distance_matrix

    @staticmethod
    def text_to_vocabulary(text: str) -> 'Vocabulary'[_T]:
        """ Converts text to words. """
        words: Vocabulary[str] = Vocabulary()
        for word in word_tokenize(text):
            words.add(word)
        return words

    @staticmethod
    def texts_to_vocabulary(sentences: List[str]) -> Vocabulary[_T]:
        words: Vocabulary[str] = Vocabulary()
        for sentence in sentences:
            words.union(Vocabulary.text_to_vocabulary(sentence))
        return words
