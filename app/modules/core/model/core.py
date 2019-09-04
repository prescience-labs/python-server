# from .inference import inference
from .train import train
from .update import update

from .auto_aspect import auto_aspect
from .architect import inference
from .utils import load_opinion
from .utils import preprocess


def process_single(input: str):
    """
    """
    # derive custom aspects programmatically
    custom_aspects = auto_aspect(input)

    # load a pre-trained opinion lexicon
    # opinion_lex = load_opinion()
    opinion_lex = 'opinion_lex.csv'

    # preprocessing
    # preprocessed = preprocess(input)
    preprocessed = input

    # perform inference with the model
    result = inference(preprocessed, custom_aspects, opinion_lex)

    return result
    # return custom_aspects


def train_model():
    """
    """
    pass


def update_model():
    """
    """
    pass
