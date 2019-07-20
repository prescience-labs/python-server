import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class Settings():
    env = os.getenv('ENVIRONMENT', 'development')
    debug = True if env == 'development' else False
    secret_key = os.getenv('SECRET_KEY')
