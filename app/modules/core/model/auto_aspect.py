"""
TODO:
- multi-method aspect extraction including
-    NER
-    POS pattern matching extrapolated from baseline aspect lexicon
-    vector similarity against a baseline aspect lexicon
-    V1 of this on amazon dataset into baseline to bootstrap into V2 (> POS patterns, embeddings for similarity)
"""
import collections

import spacy


def auto_aspect(in_review: str, testing:bool = False) -> dict:
    """
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(in_review)

    stops = ['i', 'we', 'were', 'was', 'is', 'had', "'s"]

    candidates = extract_candidates(doc, stops)  # is it necessary to preserve case-sensitive version?
    lower_candidates = [c.lower() for c in candidates]

    # better TODO comment
    doc_dict = build_dict_from_doc(doc, lower_candidates)

    # distill aspects into clean lexicon
    aspects = set()
    for sent_idx, dict_of_token_dicts in doc_dict.items():
        for token, info in dict_of_token_dicts.items():
            if info['is_aspect']:
                aspects.add(token.text.lower())

    # extract multi-word entities by grouping aspect terms that are neighbors (heuristic!)
    for sent_idx, dict_of_token_dicts in doc_dict.items():
        for i,j in zip(list(dict_of_token_dicts.items()), list(dict_of_token_dicts.items())[1:]):
            if i[1]['is_aspect']==j[1]['is_aspect'] and i[1]['is_aspect']:
                aspects.add(f'{i[0]} {j[0]}'.lower())

    # spin up synonyms for each aspect into dict format
    aspect_dict = {}
    for asp in aspects:
        if testing:
            import nltk
            # wordnet syns, embeddings above sim threshold
            # syns = wn.synsets(asp, pos=wn.NOUN)
            text = nltk.Text(word.lower() for word in in_review)
            syns = text.similar(asp)  # do auto-aspect on all of amazon dataset, then do this and emb similarity at inference against that baseline using some kind of search algorithms?
            print(asp, syns, sep='\t')
        aspect_dict[asp] = [asp, asp.upper(), asp.lower(), asp + 's']

    if testing:
        print([(str(x), x.label_) for x in doc.ents], flush=True)

    return aspect_dict


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


def check_if_aspect(token, lower_candidates):
    """
    """
    if token.text.lower() in lower_candidates:
        if token.pos_.startswith('V'):
            return False
        elif token.pos_ == 'ADJ':
            return False
        else:
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


if __name__ == '__main__':
    """Unit testing and experimentation"""
    # test_string = "Fantastic experience! If I was able to put 10 stars I would. I loved my experience it was really well done. The dogs were so cute and happy. I thought the staff were very knowledgeable they certainly know there stuff and the dogs really well. I loved how happy they were sitting on my lap. ( the dogs not the staff )    The tea and cake were delicious and made my experience perfect. What a great atmosphere and experience to have."
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testing", action="store_true", help="testing and experimentation")
    parser.add_argument("-b", "--batch", action="store_true", help="batch mode")
    args = parser.parse_args()
    if args.testing:
        result = auto_aspect(test_string, testing)
        from pprint import pprint
        pprint(result)
