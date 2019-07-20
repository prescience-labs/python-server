import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def preprocess(input: str) -> list:
    """
    """
    return word_tokenize(input)
