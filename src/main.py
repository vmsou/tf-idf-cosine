"""
Aluno: Vinícius Marques da Silva de Oliveira

1. Sua tarefa será gerar a matriz termo-documento usando TF-IDF por meio da aplicação das
fórmulas  TF-IDF  na  matriz  termo-documento  criada  com  a  utilização  do  algoritmo  Bag of
Words. Sobre o Corpus que recuperamos anteriormente. O entregável desta tarefa é uma
matriz termo-documento onde a primeira linha são os termos e as linhas subsequentes são
os vetores calculados com o TF-IDF.

2. Sua tarefa será gerar uma matriz de distância, computando o cosseno do ângulo entre todos
os vetores que encontramos usando o tf-idf. Para isso use a seguinte fórmula para o cálculo
do  cosseno  use  a  fórmula  apresentada  em  Word2Vector  (frankalcantara.com)
(https://frankalcantara.com/Aulas/Nlp/out/Aula4.html#/0/4/2)  e  apresentada  na  figura  a
seguir:

O resultado deste trabalho será uma matriz que relaciona cada um dos vetores já calculados
com todos os outros vetores disponíveis na matriz termo-documento mostrando a distância
entre cada um destes vetores.
"""

from typing import List

import pandas as pd

from vocabulary import Vocabulary
from corpus.scraper import Article, sentences_from_articles

DEFAULT_ARTICLES: List[Article] = [
    Article(
        title="Your Guide to Natural Language Processing (NLP)",
        link="https://www.datasciencecentral.com/your-guide-to-natural-language-processing-nlp/"
    ),
    Article(
        title="Part 1: Step by Step Guide to Master NLP – Introduction",
        link="https://www.analyticsvidhya.com/blog/2021/06/part-1-step-by-step-guide-to-master-natural-language-processing-nlp-in-python/"
    ),
    Article(
        title="What is NLP? Natural language processing explained",
        link="https://www.cio.com/article/228501/natural-language-processing-nlp-explained.html"
    ),
    Article(
        title="Overview of Artificial Intelligence and Role of Natural Language Processing in Big Data",
        link="https://www.datasciencecentral.com/overview-of-artificial-intelligence-and-role-of-natural-language"
    ),
    Article(
        title="Natural Language Processing (NLP)",
        link="https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP"
    )
]


def main() -> None:
    sentences: List[str] = [
        "A carteira colocou a carteira na carteira.",
        "O carteiro não tem carteira.",
        "O carteiro comprou uma carteira nova."
    ]

    articles: list[Article] = DEFAULT_ARTICLES.copy()

    print("[Articles]".center(80, '-'))
    for article in articles:
        print(f"{article.title}: {article.link}")
    print("".center(80, '-'))

    documents_sentences: List[List[str]] = sentences_from_articles(articles)
    if documents_sentences:
        # Flatten sentences
        sentences = [sentence for sentences in documents_sentences for sentence in sentences]

    # Generate vocabulary from sentences
    print("Generating vocabulary...", end=' ')
    vocabulary: Vocabulary[str] = Vocabulary.texts_to_vocabulary(sentences)
    print('Done.')

    # Generate Document-term matrix
    print("Generating Document-term matrix...", end=' ')
    matrix: pd.DataFrame = vocabulary.to_matrix(sentences)
    print("Done.")
    print(matrix.head())
    print()

    # Generate TF matrix
    print("Generating Term Frequency matrix...", end=' ')
    tf_matrix: pd.DataFrame = vocabulary.to_tf(sentences)
    print("Done.")
    print(tf_matrix.head())
    print()

    # Generate IDF matrix
    print("Generating Inverse Document Frequency matrix...", end=' ')
    idf_matrix: pd.DataFrame = vocabulary.to_idf(sentences)
    print("Done.")
    print(idf_matrix.head())
    print()

    # Generate TF-IDF matrix
    print("Generating Term Frequency-Inverse Document Frequency matrix...", end=' ')
    tf_idf_matrix: pd.DataFrame = vocabulary.to_tf_idf(sentences)
    print("Done.")
    print(tf_idf_matrix.head())
    print()

    # Cosine Similarity
    print("Comparing Cosine Similarity")
    distance_matrix: pd.DataFrame = vocabulary.to_similarity(sentences)
    print(distance_matrix.head())


if __name__ == "__main__":
    main()
