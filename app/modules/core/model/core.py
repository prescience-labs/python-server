from .auto_aspect import auto_aspect
from .inference import inference
# from .train import train
# from .update import update


def process_single(input: str):
    """Given a raw string, automatically extract aspects of interest, then pass
    those along with the filename of the opinion lexicon to the inference engine
    to retrieve and return the sentiment-analyzed document.
    """
    # derive custom aspects programmatically
    custom_aspects = auto_aspect(input)

    # get rid of this part in favor of hard-coding this lexicon at a lower level at least during inference?
    # instantiate the name of the opinion lexicon
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
