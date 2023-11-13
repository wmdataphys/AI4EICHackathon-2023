from datetime import datetime
from flask_leaderboard import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Team(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    # declaring name only 30 characters max.
    # Caps and small no special characters
    name = db.Column(db.String(30), nullable = False, unique = True)
    users = db.relationship('User', backref = 'team_name', lazy = True)
    questions = db.relationship('Question', backref = 'qteam_name', lazy = True)
    password = db.Column(db.String(60), unique = False, nullable = False)
    q1_bestscore = db.Column(db.Float, nullable = True, default = 0.0)
    q2_bestscore = db.Column(db.Float, nullable = True, default = 0.0)
    q3_bestscore = db.Column(db.Float, nullable = True, default = 0.0)
    overallscore = db.Column(db.Float, nullable = True, default = 0.0)
    def __repr__(self):
        return f"Team('{self.name}')"

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=False, nullable=False)
    teamname = db.Column(db.String(30), db.ForeignKey('team.name'), nullable = False)
    password = db.Column(db.String(60), unique = False, nullable = False)
    def __repr__(self):
        return f"User('{self.username}')"

class Question(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    teamname = db.Column(db.String(30), db.ForeignKey('team.name'), nullable = False)
    username = db.Column(db.String(60), nullable = False)
    qnumber = db.Column(db.Integer, nullable = False)
    qscore = db.Column(db.Float, unique = False, nullable = False)
    filename = db.Column(db.String(200), nullable = False)
    submit_time = db.Column(db.DateTime, nullable = False, default = datetime.utcnow)
    remarks = db.Column(db.String(300), nullable = False, default = "")

    def __repr__(self):
        return f"Question('{self.teamname}', '{self.username}', {self.qnumber}, {self.qscore}, {self.submit_time})"
