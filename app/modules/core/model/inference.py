"""A module for running leveraging a pretrained model to analyze a single
review.
"""
import csv

import pandas as pd


def check_polarity(list_of_tokens, opinion_lex):
    """
    """
    polarity_hits = {}
    for token in list_of_tokens:
        if token.lower() in opinion_lex:
            polarity_hits[token] = opinion_lex[token.lower()]

    return polarity_hits


def inference(preprocessed) -> dict:
    """
    """
    # aspect_lex = pd.read_csv('aspects.csv')
    # opinion_lex = pd.read_csv('opinions.csv')
    opinion_lex = {'bad': 'NEG',
                   'awful': 'NEG',
                   'different': 'NEU',
                   'love': 'POS',
                   'excellent': 'POS',
                   'comfortable': 'POS'}

    result = {}
    for sentence, info in preprocessed.items():
        result[sentence] = (info[0], check_polarity(info[1], opinion_lex))

    return result
