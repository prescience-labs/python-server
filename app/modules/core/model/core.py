from .train import train  # TODO
from .update import update  # TODO

from .auto_aspect import auto_aspect
from .inference import inference


def process_single(input: str):
    """
    """
    # derive custom aspects programmatically
    custom_aspects = auto_aspect(input)

    # get rid of this part?
    opinion_lex = 'opinion_lex.csv'

    # preprocessing // and this part
    preprocessed = input

    # perform inference with the model
    result = inference(preprocessed, custom_aspects, opinion_lex)

    return result


def train_model():
    """
    """
    pass


def update_model():
    """
    """
    pass
