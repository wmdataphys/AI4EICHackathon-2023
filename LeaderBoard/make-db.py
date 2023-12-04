from flask_leaderboard import db, bcrypt

import string
import secrets
import random

def make_password(password):
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    return hashed_password

db.drop_all()

db.create_all()

from flask_leaderboard.models import Team, User

# creating teams - leave commented until morning of 10/14
team_names = []
for t in range(1, 11):
    team_names.append(f"Team {t}")

# creating 4 passwords

passwords = ["AI4EIC"]*len(team_names)
# creating users,
# Note that there could be people with same name in different teams
users = dict()
for player in range(0, 41, 4):
    users[f"Team {player//4 + 1}"] = [f"Player_{player + 1}", f"Player_{player + 2}", f"Player_{player + 3}", f"Player_{player + 4}"]

db.create_all()
for team, pword in zip(team_names, passwords):
    pword = make_password(pword)
    print (team)
    tem = Team(name = team, password = pword)
    db.session.add(tem)
    db.session.commit()
    for user in users[team]:
        print (tem.id)
        db.session.add(User(username = user,
                            teamname = tem.name,
                            password = pword,
                            OPENAI_API_KEY = "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"
                            )
                       )
        db.session.commit()

