from flask_leaderboard import db, bcrypt

import string

def make_password(password):
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    return hashed_password

from flask_leaderboard.models import Team, User, Question

db.session.commit()


# creating teams - leave commented until morning of 10/14
#team_names = ["Team Manitoba","Team AM","Jets","In Principle","PLYD","Team CUK","JINR","Team Regina","Team WM and DK","JB and EC"]
team_names = ["Team Regina"]

# creating 4 passwords

pword = "yj5SGm2p"

# creating users,
# Note that there could be people with same name in different teams
users = dict()
team = "Team Regina"
users[team] = ["Azizah_M", "Gabriel_C"]


for user in users["Team Regina"]:
    db.session.add(User(username = user, teamname = team, password = pword))
    db.session.commit()

# turn this section ON for generating random scores for questions
"""
for submissions in range(20):
    # Logic would be run the evaluator in this core.
    db.session.add(Question(teamname = secrets.choice(team_names),
                            username = secrets.choice(user_names),
                            qnumber = secrets.choice([1, 2, 3]),
                            qscore = 100*random.uniform(0.5, 1)
                            )
                    )
    db.session.commit()
"""
