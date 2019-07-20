from flask import Blueprint

from .model.core import process_single

mod_core = Blueprint('core', __name__, url_prefix='/')


@mod_core.route('/')
def core():
    """HTTP root
    """
    input = "I really love the fit of this blouse, but the color is different than it seems online. The texture is excellent and it's comfortable to wear."
    result = process_single(input)
    return {
        'core': result,
    }
