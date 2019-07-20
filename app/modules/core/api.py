from flask import Blueprint

from .model.core import process_single

mod_core = Blueprint('core', __name__, url_prefix='/')


@mod_core.route('/')
def core():
    """HTTP root
    """
    input = "Doctor Jenkins was a complete idiot but I love his staff..."
    result = process_single(input)
    return {
        'core': result,
    }
