# import .inference
# import .train
# import .update


def process_single(input: str):
    """
    """
    # preprocessing
    tokenized = input.split()

    # perform inference with the model
    # result = inference.run(preprocessed)

    result = {}
    for token in tokenized:
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
