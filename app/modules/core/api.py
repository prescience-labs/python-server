from flask import Blueprint

mod_core = Blueprint('core', __name__, url_prefix='/')

@mod_core.route('/')
def core():
    """HTTP root
    """
    return {
        'core': 'here',
    }
