from flask import Blueprint

from ...settings import Settings

mod_status = Blueprint('status', __name__, url_prefix='/status')

@mod_status.route('/')
def status():
    return {
        'status': 'ok',
        'environment': Settings.env,
    }
