# import .inference
# import .train
# import .update
from .utils import preprocess


def process_single(input: str):
    """
    """
    # preprocessing
    preprocessed = preprocess(input)

    # perform inference with the model
    # result = inference.run(preprocessed)

    result = {}
    for token in preprocessed:
        result[token] = 'POLARITY'

    return result


def train_model():
    """
    """
    pass


def update_model():
    """
    """
    pass
