from .train import train  # TODO
from .update import update  # TODO

from .auto_aspect import auto_aspect
from .inference import inference


def process_single(input: str):
    """
    """
    # derive custom aspects programmatically
    custom_aspects = auto_aspect(input)

    # get rid of this part in favor of hard-coding this lexicon at a lower level at least during inference?
    opinion_lex = 'opinion_lex.csv'

    # perform inference with the model
    result = inference(input, custom_aspects, opinion_lex)

    return result


def train_model():
    """
    """
    pass


def update_model():
    """
    """
    pass
