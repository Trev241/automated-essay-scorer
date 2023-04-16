import os

from src.autograder import Autograder
from flask import Flask

autograder = Autograder()
app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


from src import routes