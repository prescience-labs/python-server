from flask import Blueprint

from .model.core import process_single

mod_core = Blueprint('core', __name__, url_prefix='/')


@mod_core.route('/infer')
def core():
    """HTTP infer endpoint
    """
    # change this so it takes an input string from the API request
    # validate encoding, formatting?
    input = "Fantastic experience! If I was able to put 10 stars I would. I loved my experience it was really well done. The dogs were so cute and happy. I thought the staff were very knowledgeable they certainly know there stuff and the dogs really well. I loved how happy they were sitting on my lap. ( the dogs not the staff )    The tea and cake were delicious and made my experience perfect. What a great atmosphere and experience to have. "
    result = process_single(input)
    return {
        'core': result,
    }


@mod_core.route('/train')
def train():
    """HTTP train model endpoint
    """
    pass
