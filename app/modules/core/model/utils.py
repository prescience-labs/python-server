import collections
import re

import nltk
from nltk.tokenize import sent_tokenize

import spacy


def doc2dict(doc):
    """
    """
    info = []
    for token in doc:
        info.append({'token': token.text,
                     'lemma': token.lemma_,
                     'pos': token.pos_,
                     'dep': token.dep_,
                     'stop': token.is_stop})

    return info


def preprocess(input: str) -> list:
    """
    """
    nlp = spacy.load("en_core_web_sm")
    sents = sent_tokenize(input)
    sentences = {}
    for i, sent in enumerate(sents):
        doc = nlp(sent)
        info = doc2dict(doc)
        sentences[f'sentence_{i}'] = (sent, info)

    return sentences


def load_opinion() -> set:
    """
    """
    pass
