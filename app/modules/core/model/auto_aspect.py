import collections

import spacy


def auto_aspect(in_review: str) -> dict:
    """Given a raw string of review text, return a dictionary of aspect candidates
    as keys with all of their inflected forms as values.
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(in_review)

    # this is a place for easy-updates to the model;
    # if the module is considering uninformative function words like those below
    # to be aspects, add them to this list!
    stops = ['i', 'we', 'were', 'was', 'is', 'had', "'s"]

    # use linguistically-inspired heuristics to get all tokens which may be whole or parts of aspects
    candidates = extract_candidates(doc, stops)
    # is it necessary to preserve a case-sensitive version distinct from lower_candidates?
    lower_candidates = [c.lower() for c in candidates]

    # flesh out a better format of the data, combining spacy token attributes with the novel aspect-detection results
    doc_dict = build_dict_from_doc(doc, lower_candidates)

    # distill candidates into clean aspect lexicon
    aspects = set()
    for candidate in lower_candidates:
        aspects.add(candidate)

    # extract multi-word entities by grouping aspect terms that are neighbors
    # WARNING: there may be cases where this heuristic fails and mistakenly creates aspect compounds that should be kepy separate
    for sent_idx, dict_of_token_dicts in doc_dict.items():
        for i,j in zip(list(dict_of_token_dicts.items()), list(dict_of_token_dicts.items())[1:]):
            if i[1]['is_aspect']==j[1]['is_aspect'] and i[1]['is_aspect']:
                aspects.add(f'{i[0]} {j[0]}'.lower())

    # exclude aspects that were contextually POS-tagged as Verbs of any kind or adjectives
    # play around with these triggers with an eye on recall
    for sent_idx, dict_of_token_dicts in doc_dict.items():
        for token, info in dict_of_token_dicts.items():
            low_token = token.text.lower()
            if low_token in aspects:
                if info['pos'].startswith('V') or info['pos'].startswith('AD'):  # TODO: fix for "happy they"
                    aspects.remove(low_token)

    # spin up synonyms for each aspect into dict format
    aspect_dict = {}
    for asp in aspects:
        # could also do wordnet synonyms or words with embeddings above some high threshold of similarity (and pipe them into some overall aspect lexicon)
        aspect_dict[asp] = [asp, asp.upper(), asp.lower(), asp + 's']

    return aspect_dict


def build_dict_from_doc(doc, lower_candidates) -> dict:
    """Given the spacy doc of the input text and the lowerized version of the
    aspect candidates, return a document-level dictionary containing token-level
    dictionaries. The token-level dictionaries contain reformatted or exactly
    copied over attributes from spacy, along with importantly whether or not it
    is an aspect or not.
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


def check_if_aspect(token, lower_candidates):
    """Helper function for build_dict_from_doc() which checks if the current
    token in the spacy doc is one of our candidates for aspects
    """
    if token.text.lower() in lower_candidates:
        return True
    else:
        return False


def extract_candidates(doc, stops) -> set:
    """Based on linguistic theory and manual analyses, use certain depedency
    labels as criteria for aspect-ness. Also exclude stop words here.

    Given a spacy doc object and a list of stop words, return a clean set() of
    aspect candidates.
    """
    candidates = []
    for token in doc:
        if token.dep_ in ['nsubj', 'dobj', 'pobj', 'conj', 'compound']:
            if token.text.lower() not in stops:
                candidates.append(token.text)

    return set(candidates)
