from datetime import datetime
from flask_leaderboard import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Team(db.Model, UserMixin):
    __tablename__ = 'team'
    id = db.Column(db.Integer, primary_key = True)
    # declaring name only 30 characters max.
    # Caps and small no special characters
    name = db.Column(db.String(30), 
                     nullable = False, 
                     unique = True
                     )
    users = db.relationship('User', 
                            backref = 'TEAM_NAME', 
                            lazy = True
                            )
    questions = db.relationship('Question', 
                                backref = 'Q_TEAM_NAME', 
                                lazy = True
                                )
    password = db.Column(db.String(60), 
                         unique = False, 
                         nullable = False
                         )
    q1_bestscore = db.Column(db.Float, nullable = True, default = 0.0)
    q2_bestscore = db.Column(db.Float, nullable = True, default = 0.0)
    q3_bestscore = db.Column(db.Float, nullable = True, default = 0.0)
    overallscore = db.Column(db.Float, nullable = True, default = 0.0)
    def __repr__(self):
        return f"Team('{self.name}')"

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), 
                         unique=False, 
                         nullable=False
                         )
    teamname = db.Column(db.String(30), 
                         db.ForeignKey('team.name'), 
                         unique = False, 
                         nullable = False
                         )
    password = db.Column(db.String(60), 
                         unique = False, 
                         nullable = False
                         )
    OPENAI_API_KEY = db.Column(db.String(60), 
                               unique = False, 
                               nullable = False
                               )
    TotalSessions = db.Column(db.Integer, 
                              nullable = True, 
                              default = 0
                              )
    questions = db.relationship('Question',
                                backref = 'Q_USER_NAME',
                                lazy = True
                                )
    
    AllChatSessions = db.relationship('ChatSessions', 
                                  backref = 'USER_NAME', 
                                  lazy = True
                                  )
    def __repr__(self):
        return f"User('{self.username}')"

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, primary_key = True)
    teamname = db.Column(db.String(30), 
                         db.ForeignKey('team.name'), 
                         nullable = False
                         )
    username = db.Column(db.String(60), 
                         db.ForeignKey('user.username'),
                         nullable = False
                         )
    qnumber = db.Column(db.Integer, nullable = False)
    qscore = db.Column(db.Float, unique = False, nullable = False)
    filename = db.Column(db.String(200), nullable = False)
    submit_time = db.Column(db.DateTime, 
                            nullable = False, 
                            default = datetime.utcnow
                            )
    remarks = db.Column(db.String(300), nullable = False, default = "")

    def __repr__(self):
        return f"Question('{self.teamname}', '{self.username}', {self.qnumber}, {self.qscore}, {self.submit_time})"

class ChatSessions(db.Model, UserMixin):
    __tablename__ = 'chatsessions'
    id = db.Column(db.Integer, primary_key = True)
    sessioname = db.Column(db.String(60), nullable = False)
    username = db.Column(db.String(60), db.ForeignKey('user.username'), nullable = False)
    index = db.Column(db.Integer, nullable = False)
    uuid = db.Column(db.String(60), unique = True, nullable = False)
    start_time = db.Column(db.DateTime, 
                           nullable = False, 
                           default = datetime.utcnow
                           )
    end_time = db.Column(db.DateTime,
                         nullable = False,
                         default = datetime.utcnow
                        )
    const_sys_context = db.Column(db.Text(), nullable = False, default = "")
    user_sys_context = db.Column(db.Text(), nullable = False, default = "")
    num_chats = db.Column(db.Integer, nullable = False, default = 0)
    ChatHistory = db.relationship('ChatInfo', backref = "session_index", lazy = True)
    
    def __repr__(self):
        return f"ChatSessions('{self.username}', '{self.session_id}')"

class ChatInfo(db.Model, UserMixin):
    __tablename__ = 'chatinfo'
    id = db.Column(db.Integer, primary_key = True)
    index = db.Column(db.Integer, nullable = False)
    chat_id = db.Column(db.Text(), nullable = False)
    username = db.Column(db.String(60), nullable = False)
    session_uuid = db.Column(db.Integer, db.ForeignKey('chatsessions.index'), nullable = False)
    time_of_chat = db.Column(db.DateTime, 
                             nullable = False, 
                             default = datetime.utcnow
                             )
    user_prompt = db.Column(db.Text(), nullable = False)
    ai_response = db.Column(db.Text(), nullable = False)
    system_response = db.Column(db.Text(), nullable = False)
    feedback = db.Column(db.Boolean, nullable = False)
    prompt_tokens = db.Column(db.Integer, nullable = False)
    completion_tokens = db.Column(db.Integer, nullable = False)
    is_reponse_downloaded = db.Column(db.Boolean, nullable = False, default = False)
    def __repr__(self):
        return f"ChatInfo('{self.username}')"
