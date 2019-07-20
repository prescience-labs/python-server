from .inference import inference
from .train import train
from .update import update
from .utils import preprocess


def process_single(input: str):
    """
    """
    # preprocessing
    preprocessed = preprocess(input)

    # perform inference with the model
    # result = inference(preprocessed)

    # result = {}
    # for token in preprocessed:
    #     result[token] = 'POLARITY'

    # return result
    return preprocessed


def train_model():
    """
    """
    pass


def update_model():
    """
    """
    pass
