from typing import List, Dict

import bs4 as bs4
import requests
import logging

import spacy.tokens

from corpus import NLP

logger: logging.Logger = logging.getLogger("scraper")
logger.setLevel(logging.INFO)

HEADER: Dict[str, str] = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


class Article:
    def __init__(self, title: str, link: str):
        self.title: str = title
        self.link: str = link


def sentences_from_text(text: str) -> List[str]:
    """ Extracts sentences from text string. """
    document: spacy.tokens.Doc = NLP(text)
    reports: List[str] = []
    for sent in document.sents:
        text: str = sent.text.replace("\n", ' ').replace("\t", ' ')
        if text: reports.append(text)
    return reports


def sentences_from_site(url: str) -> List[str]:
    """ Extracts sentences from website (PDF or HTML).  """
    print(f"Generating reports from site: {url}...", end=' ')
    response: requests.Response = requests.get(url, headers=HEADER)
    if response.status_code != 200:
        logger.warning(f"Couldn't reach {url} for scraping. Status Code: {response.status_code}")
        print("Failed.")
        return []

    text: str = ""
    soup: bs4.BeautifulSoup = bs4.BeautifulSoup(response.text, "html.parser")
    for p in soup.select("p"):
        text += p.text
    reports: List[str] = sentences_from_text(text)
    print("Done.")
    return reports
