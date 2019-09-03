from .inference import inference
from .train import train
from .update import update
from .utils import auto_aspect


def process_single(input: str):
    """
    """
    # auto-aspect
    custom_aspects = auto_aspect(input)

    # perform inference with the model
    # result = inference(preprocessed)

    # return result
    return custom_aspects


def train_model():
    """
    """
    pass


def update_model():
    """
    """
    pass
