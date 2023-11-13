from flask import Flask
import os
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'submissions')
RES_FOLDER = os.path.join(os.getcwd(), 'actual_results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RES_FOLDER'] = RES_FOLDER
app.config['Q_KEYS'] = dict()
app.config['Q_THRES'] = {1 : 0.94, 2 : 0.86, 3 : 0.80}
for k in range(1, 4):
    app.config['Q_KEYS'][k] = os.path.join(app.config['RES_FOLDER'], f'Question{k}_test_labels.csv')
app.config['MAX_CONTENT_LENGTH'] =  3 * 1000 * 1000 # max 2mb file
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.app_context().push()
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

if(not os.path.exists(app.config['RES_FOLDER'])):
    os.makedirs(app.config['RES_FOLDER'])

if(not os.path.exists(app.config['UPLOAD_FOLDER'])):
    os.makedirs(app.config['UPLOAD_FOLDER'])

from flask_leaderboard import routes