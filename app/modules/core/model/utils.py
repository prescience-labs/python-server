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
        sentences[f'sentence_{i}'] = (s, info)

    return sentences


def check_if_aspect(token, lower_candidates):
    """
    """
    if token.text.lower() in lower_candidates:
        return True
    else:
        return False


def extract_candidates(doc, stops) -> set:
    """
    """
    candidates = []
    for token in doc:
        if token.dep_ in ['nsubj', 'dobj', 'pobj', 'conj', 'compound']:
            if token.text.lower() not in stops:
                candidates.append(token.text)

    return set(candidates)


def build_dict_from_doc(doc, lower_candidates) -> dict:
    """
    """
    # intialize data structure
    doc_dict = collections.OrderedDict()
    for i, sent in enumerate(doc.sents):
        for token in sent:
            doc_dict[f'sent_{i}'] = sent

    # fill data structure by reassignment with actual values
    for sent_idx, sent in doc_dict.items():
        proposed_dict = collections.OrderedDict()
        for i, token in enumerate(sent):
            proposed_dict[token] = {'idx': i, 'pos': token.pos_, 'dep': token.dep_, 'is_aspect': check_if_aspect(token, lower_candidates), 'children': [child for child in token.children]}
        doc_dict[sent_idx] = proposed_dict

    return doc_dict


def auto_aspect(in_review: str) -> dict:
    """
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(in_review)

    stops = ['i', 'we', 'were', 'was', 'is', 'had', "'s"]

    candidates = extract_candidates(doc, stops)  # is it necessary to preserve case-sensitive version?
    lower_candidates = [c.lower() for c in candidates]

    # better TODO comment
    doc_dict = build_dict_from_doc(doc, lower_candidates)

    # distill candidates into clean aspect lexicon
    aspects = set()
    # the singles
    for candidate in lower_candidates:
        aspects.add(candidate)
    # extract multi-word entities
    for sent_idx, dict_of_token_dicts in doc_dict.items():
        for i,j in zip(list(dict_of_token_dicts.items()), list(dict_of_token_dicts.items())[1:]):
            if i[1]['is_aspect']==j[1]['is_aspect'] and i[1]['is_aspect']:
                aspects.add(f'{i[0]} {j[0]}'.lower())

    return list(aspects)
