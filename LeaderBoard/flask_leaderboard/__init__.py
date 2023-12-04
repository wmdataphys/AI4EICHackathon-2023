from flask import Flask
import os
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_leaderboard.OpenAIAg.OpenAIChat import OpenAIChat
from flask_leaderboard.utils import OPENAI_Utils, DB_Utils
from flask_session import Session
from datetime import timedelta
from uuid import uuid5, NAMESPACE_OID
from flask_wtf.csrf import CSRFProtect

#from flask_bootstrap import Bootstrap5

app = Flask(__name__)

app.secret_key = "5791628bb0b13ce0c676dfde280ba245"
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SESSION_PERMANENT"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=60)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
#'mysql://ai4eichackathon:Hack_2023@ai4eichackathon.mysql.pythonanywhere-services.com/ai4eichackathon$beta_users' #'sqlite:///users.db' # this should change to MySQL now.
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://ai4eichackathon:Hack_2023@ai4eichackathon.mysql.pythonanywhere-services.com/ai4eichackathon$users'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@/hackathon?unix_socket=/cloudsql/eighth-gamma-406401:us-central1:hackathon'
#app.config['SQLALCHEMY_DATABASE_URI'] ='mysql+mysqlconnector://root:@34.122.152.125/hackathon'
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = { 'pool_recycle': 200, 'pool_pre_ping': True}
app.app_context().push()
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


csrf = CSRFProtect(app)

config = {
    "DEBUG": False,          # some Flask specific configs
}
app.config.from_mapping(config)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
login_manager.refresh_view = "accounts.reauthenticate"
login_manager.needs_refresh_message = (
    u"To protect your account, please reauthenticate to access this page."
)
login_manager.needs_refresh_message_category = "info"
sess = Session(app)



UPLOAD_FOLDER = os.path.join(os.getcwd(), 'submissions')
RES_FOLDER = os.path.join(os.getcwd(), 'actual_results')
constants = OPENAI_Utils()
app.config["OPENAI_params"] = constants

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MDEDITOR_FILE_UPLOADER'] = UPLOAD_FOLDER
app.config['RES_FOLDER'] = RES_FOLDER
app.config['Q_KEYS'] = dict()
app.config["OPENAI_USERS"] = dict()
app.config['Q_THRES'] = {1 : 0.94, 2 : 0.86, 3 : 0.80}
for k in range(1, 3):
    app.config['Q_KEYS'][k] = os.path.join(app.config['RES_FOLDER'], f'AnswerKey_Part{k}.csv')
app.config['MAX_CONTENT_LENGTH'] =  3 * 1000 * 1000 # max 2mb file
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_TOKENS"] = constants.MAX_TOKENS
app.config["GPT_MODEL"] = constants.GPT_MODEL
app.config["TEMPERATURE"] = constants.TEMPERATURE

session_team_user = dict()

if(not os.path.exists(app.config['RES_FOLDER'])):
    os.makedirs(app.config['RES_FOLDER'])

if(not os.path.exists(app.config['UPLOAD_FOLDER'])):
    os.makedirs(app.config['UPLOAD_FOLDER'])

from flask_leaderboard import routes

#app.run(host = "127.0.0.1", port = "5500", debug=True)
