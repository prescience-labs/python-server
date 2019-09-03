from flask import Blueprint

from .model.core import process_single

mod_core = Blueprint('core', __name__, url_prefix='/')


@mod_core.route('/infer')
def core():
    """HTTP infer endpoint
    """
    input = "I really love the fit of this blouse, but the color is different than it seems online. The texture is excellent and it's comfortable to wear."
    result = process_single(input)
    return {
        'core': result,
    }


@mod_core.route('/train')
def train():
    """HTTP train model endpoint
    """
    result = train_model()
    return result
