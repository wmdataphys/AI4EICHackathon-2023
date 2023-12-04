import os
from flask import render_template, url_for, flash, redirect, abort, request, jsonify, session
from flask_leaderboard import app, db, bcrypt, session_team_user
from flask_leaderboard.forms import LoginForm, SubmitForm, OpenAISessionForm, ForceLogoutForm
from flask_leaderboard.models import Team, User, Question, ChatSessions, ChatInfo
from flask_login import login_user, current_user, logout_user, login_required, user_needs_refresh
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_leaderboard.evaluator import Evaluate, evaluate
import flask_leaderboard.aiapi
import flask_leaderboard.config
from flask_leaderboard.OpenAIAg.OpenAIChat import OpenAIChat
import logging
from markupsafe import Markup
# Some default settings
from flask_leaderboard.utils import OPENAI_Utils
from uuid import NAMESPACE_OID, uuid1, uuid5
from functools import wraps

utility = OPENAI_Utils()

"""
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if(current_user.is_authenticated):
            tmp = str(uuid5(NAMESPACE_OID, f"{current_user.teamname}_{current_user.username}"))
            print (tmp, session.get("user_uuid"))
            if(not session.get("user_uuid")):
                return redirect(url_for('login', next_page = 'chat'))
            elif (session.get("user_uuid") != tmp):
                return abort(403)
            else:
                return f(*args, **kwargs)
        else:
            return redirect(url_for('login', next_page = 'chat'))
    return wrap
"""

@app.before_request
def make_session_permanent():
    session.permanent = False

"""
    @app.errorhandler(Exception)
def handle_exception(e):
    return render_template("error_500.html", e=e), 500
"""
@app.errorhandler(404)
def page_not_found(e):
    return render_template('page_not_found.html'), 404

@app.errorhandler(403)
def access_forbidden(e):
    return render_template('acess_denied.html'), 403


@app.route("/")
@app.route("/leaderboard")
def leaderboard():
    TeamInfo = Team.query.order_by(Team.overallscore.desc()).all()
    #fancy way
    #ques_info = {T.name : set([s.qnumber for s in T.questions]) for T in TeamInfo}
    ques_info = dict()
    for T in TeamInfo:
        unique_ques = set([f"Q {q.qnumber}" for q in T.questions])
        if len(unique_ques) == 0:
            ques_info[T.name] = "Yet to submit"
        else:
            ques_info[T.name] = ", ".join(unique_ques)
    # Here I assume 3 Questions
    ques_dict = dict()
    for qnum in range(1, 4):
        ques_dict[qnum] = Question.query.filter(Question.qnumber == qnum).order_by(Question.qscore.desc()).all()[:3]
    return render_template('leaderboard.html', teaminfo = TeamInfo, ques_info = ques_info, ques_dict = ques_dict, sess_id = session.get("uuid"))


@app.route("/allteams")
def allteams():
    teams = Team.query.all()
    return render_template('allteams.html', title='All Teams Info', teams = teams)

@app.route("/current_teamstat")
@login_required
def current_teamstat():
    tname = current_user.teamname
    all_questions = Question.query.filter_by(teamname = tname)
    ques_dict = dict()
    for i in range(1, 4):
        ques_dict[i] = all_questions.filter(Question.qnumber == i).order_by(Question.submit_time.desc()).all()

    return render_template('current_teamstat.html', title=f'{tname} Team Stats', tname = tname, ques_dict = ques_dict)

@app.route("/force_logout", methods=['GET', 'POST'])
def force_logout(**kwargs):
    form = ForceLogoutForm()
    if form.validate_on_submit():
        teamname = form.teamname.data
        team = Team.query.filter_by(name=teamname).first()
        user = User.query.filter_by(username=form.username.data, teamname = form.teamname.data).first()
        if user and team and bcrypt.check_password_hash(team.password, form.password.data):
            # if authenticated, Check for the last session
            if(session.get("user")):
                usr = session.get("user")
                usr.isLoggedIn = False
                db.session.commit()
                logout_user()
                session.pop("openai_session", None)
                session.pop("openai_user", None)
                session.pop("user_uuid", None)
                session.pop("uuid", None)
                session.pop("user", None)
            session.clear()
            login_user(user, remember=form.remember.data, fresh=True)
            user.isLoggedIn = True
            db.session.commit()
            print (f"Current user name : {current_user.username}, Team name : {current_user.teamname}")
            session["user_uuid"] = str(uuid5(NAMESPACE_OID, f"{current_user.teamname}_{current_user.username}"))
            session["openAI_active"] = False
            session["user"] = user
            default_redirect = url_for('user_profile', teamname = teamname, username = user.username)
            return redirect(request.args.get('next', default_redirect), **kwargs)
        else:
            flash('Login Unsuccessful. Please check team, username and password', 'danger')
    return render_template('force_logout.html', title='Login', form=form, sess_id = session.get("uuid"))

@app.route("/logout")
def logout():
    try:
        tname = current_user.teamname
        uname = current_user.username
        session_team_user.pop(f'{tname}_{uname}', None)
    except AttributeError as e:
        print ("Already logged out")
    user = session.get("user")
    if(user):
        print ("THis is user.")
        user = User.query.filter_by(username = user.username, teamname = user.teamname).first()
        user.isLoggedIn = False
        db.session.commit()
    logout_user()
    session.pop("openai_session", None)
    session.pop("openai_user", None)
    session.pop("user_uuid", None)
    session.pop("uuid", None)
    session.pop("user", None)
    session.clear()

    print ("State is: ", current_user.is_authenticated)

    return redirect(url_for('leaderboard'))

@app.route("/login", methods=['GET', 'POST'])
def login(**kwargs):
    if current_user.is_authenticated:
        default_redirect = url_for('user_profile', teamname = current_user.teamname, username=current_user.username)
        return redirect(request.args.get('next', default_redirect), **kwargs)
    form = LoginForm()
    if form.validate_on_submit():
        teamname = form.teamname.data
        team = Team.query.filter_by(name=teamname).first()
        user = User.query.filter_by(username=form.username.data, teamname = form.teamname.data).first()
        if user and team and bcrypt.check_password_hash(team.password, form.password.data):
            # if authenticated, Check for the last session
            if(user.isLoggedIn):
                flash(Markup(r"You are already logged in elsewhere, please logout.."), 'danger')
            else:
                login_user(user, remember=form.remember.data, fresh=True)
                user.isLoggedIn = True
                db.session.commit()
                print (f"Current user name : {current_user.username}, Team name : {current_user.teamname}")
                session["user_uuid"] = str(uuid5(NAMESPACE_OID, f"{current_user.teamname}_{current_user.username}"))
                session["openAI_active"] = False
                session["user"] = user
                default_redirect = url_for('user_profile', teamname = teamname, username = user.username)
                return redirect(request.args.get('next', default_redirect), **kwargs)
        else:
            flash('Login Unsuccessful. Please check team, username and password', 'danger'), 400
    return render_template('login.html', title='Login', form=form, sess_id = session.get("uuid")), 200


@app.route("/profile/<teamname>/<username>", methods=['GET', 'POST'])
@login_required
def user_profile(teamname = None, username = None):
    if(current_user.username!= username or current_user.teamname!= teamname):
        return abort(403)
    return render_template("user_profile.html", title=f'{teamname} {username} Profile', session_id = session.get("uuid")), 200

def EvaluateQScore(team):
    qscores = dict()
    for question in team.questions:
        print (question.qnumber)

@app.route("/submit", methods=['GET', 'POST'])
def submit():
    uname = ""
    tname = ""
    if current_user.is_authenticated:
        uname = current_user.username
        tname = current_user.teamname
    form = SubmitForm()
    if form.validate_on_submit():
        team = Team.query.filter_by(name=form.teamname.data).first()
        user = User.query.filter_by(username=form.username.data, teamname = form.teamname.data).first()
        if user and team and bcrypt.check_password_hash(team.password, form.password.data):
            f = form.result_file.data
            qnumber = int(form.qnumber.data)
            team_folder = os.path.join(app.config['UPLOAD_FOLDER'], team.name)
            if(not os.path.exists(team_folder)):
                os.makedirs(team_folder)
            user_folder = os.path.join(team_folder, user.username)
            if(not os.path.exists(user_folder)):
                os.makedirs(user_folder)
            now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            #filename = f"Question{qnumber}_" + now + "_" + secure_filename(f.filename)
            file_path = str(f) # Remote file
            filename = file_path.split('/')[-1]
            #filepath = os.path.join(user_folder, filename)
            push = False # Means pull
            utility.scp_file(file_path,user_folder ,push)
            eval_path = os.path.join(user_folder,filename)
            status, accuracy_score = evaluate(eval_path, qnumber)
            question = Question(teamname = team.name,
                                username = user.username,
                                qnumber = qnumber,
                                qscore = accuracy_score,
                                filename = filename,
                                remarks = status
                                )
            db.session.add(question)
            db.session.commit()
            if (qnumber == 1 and team.q1_bestscore < accuracy_score):
                team.q1_bestscore = accuracy_score
                team.overallscore = team.q1_bestscore + team.q2_bestscore + team.q3_bestscore

            if (qnumber == 2 and team.q2_bestscore < accuracy_score):
                team.q2_bestscore = accuracy_score
                team.overallscore = team.q1_bestscore + team.q2_bestscore + team.q3_bestscore

            if (qnumber == 3 and team.q3_bestscore < accuracy_score):
                team.q3_bestscore = accuracy_score
                team.overallscore = team.q1_bestscore + team.q2_bestscore + team.q3_bestscore

            db.session.commit()
            if(status == 'OK'):
                flash(f"Score for the submission is {accuracy_score:.4f} for Question : {qnumber}", "info")
            else:
                flash(status, "danger")
            #return redirect(url_for('submit'))

        else:
            flash("Invalid Credentials, Check team, username and password is correct", 'danger')
            #return redirect(url_for('leaderboard'))
    return render_template('submit.html', title='Submit for Evaluations', form=form,
                            tname = tname, uname = uname), 200

"""
def submit():
    return render_template("will_open.html")
"""
@app.route("/start_session/<teamname>/<username>", methods = ['GET', 'POST'])
@login_required
def start_session(teamname = None, username = None):
    if (current_user.username!= username or current_user.teamname!= teamname):
        return abort(403), 403
    session_index = User.query.get(current_user.id).TotalSessions + 1
    form = OpenAISessionForm()
    if form.validate_on_submit():
        username = current_user.username
        teamname = current_user.teamname
        session_name = form.name.data
        if (session_name == ""):
            session_name = f"Chat Session {session_index}"
        context = form.context.data
        temp_str = f"{teamname}_{username}_{session_index}" + str(uuid1())
        session_uuid = str(uuid5(NAMESPACE_OID, temp_str))
        session["uuid"] = session_uuid
        print ("|------ Session Information --------|")
        print (f"Session Name : {session_name}")
        print (f"Session UUID : {session_uuid}")
        print (f"Session Context : {context}")
        print (f"session from flask")
        for k, v in session.items():
            print (f"{k} : {v}")
        print ("|-------------------------------------|")
        Agent = OpenAIChat(username, flask_leaderboard.config.DevelopmentConfig.OPENAI_KEY,
                           session_name = session_name,
                           session_id=session_index,
                           session_uuid=session_uuid,
                           user_context=[context]
                           )
        session["OpenAI_messages"] = Agent.msgs

        chatsession = ChatSessions(sessioname = session_name,
                               username = username,
                               index = session_index,
                               uuid = session_uuid,
                               start_time = datetime.now(),
                               end_time = datetime.now(),
                               const_sys_context = "",
                               user_sys_context = context)
        user = User.query.get(current_user.id)
        print ("user name is ", user.username)
        user.TotalSessions+=1
        db.session.add(chatsession)
        db.session.commit()
        session["openAI_active"] = True
        print ("DOne with session here")
        return redirect(url_for('chat', teamname=teamname, username = username, session_id = session_uuid))
    return render_template('start_session.html', title = 'Start Session', form = form, session_name = f"Chat Session {session_index}", session_id = session.get("uuid"))


@app.route("/chatGPT/<teamname>/<username>/<openai_session_id>", methods = ['GET', 'POST'])
@login_required
def chatGPT(teamname = None, username = None, openai_session_id = None):
    print ("|------ Chat Information --------|")
    print ("In here at chatGPT", current_user.username, username)
    print ("Session ID : ", openai_session_id, session.get("uuid"))
    print (session.get("openai_messages"), session.get("user").OPENAI_API_KEY)
    """
    agent = app.config["OPENAI_USERS"][username]
    if not agent:
        agent = OpenAIChat(username,
                           flask_leaderboard.config.DevelopmentConfig.OPENAI_KEY,
                           session_name = lastsession.name,
                           session_id = lastsession.index,
                           session_uuid = session.get("uuid")
                           )
        agent.msgs = session.get("msgs")
    agent.user_input = request.form['prompt']
    chat_id = lastsession.num_chats + 1
    return_reason = agent.Chat()
    if (return_reason != "stop"):
        print ("something is wrong with ChatGPT check it ", return_reason)
    chatinfo = ChatInfo(index = chat_id,
                        chat_id = agent.msg_id,
                        username = username,
                        session_uuid = session.get("uuid"),
                        user_prompt = agent.user_input,
                        ai_response = agent.output,
                        system_response = "", # feedback reponse
                        feedback = True,
                        prompt_tokens = agent.prompt_tokens,
                        completion_tokens = agent.output_tokens
                        )


    lastsession.end_time = datetime.now()
    agent.chat_count+=1
    lastsession.num_chats = agent.chat_count
    db.session.add(chatinfo)
    db.session.commit()
    if (agent.total_tokens > app.config["OPENAI_params"].MAX_TOKENS):
        pass # NEED TO CHANGE THIS

    #answer = flask_leaderboard.aiapi.generateChatResponse(prompt)
    """

    res = {}
    res['prompt'] = request.form['prompt']
    res['session_id'] = 0
    code,text,download = "0", "Go to Hell", "0"
    res['code'] =  code
    res['text'] = text
    res['n_code'] = len(code)
    res['n_text'] = len(text)
    res['is_downloadable'] = download
    return jsonify(res), 200

@app.route('/chat/<teamname>/<username>/<session_id>', methods = ['GET', 'POST'])
@login_required
def chat(teamname = None, username = None, session_id = None):
    print ("In here at chat", current_user.username, username)
    print ("Session ID : ", session_id, session.get("uuid"))
    if(current_user.teamname!= teamname or current_user.username!= username):
        return abort(403)
    elif (not session.get("openAI_active") or not session.get("uuid")):
        return redirect(url_for('start_session', teamname = teamname, username = username, session_id = session.get("uuid")))
    elif(session.get("uuid")!= session_id):
        flash("Your session has becom invalid. Please login in again to continue. ", "danger")
        return redirect(url_for('logout', next_page = 'login'))
    elif(session.get("uuid")== session_id):
        user = User.query.filter_by(username = username, teamname = teamname).first()
        last_session = user.AllChatSessions[-1]
        session_names = [s.sessioname for s in user.AllChatSessions]
        return render_template('chat.html', title='Chat Bot',
                               teamname = teamname, username = username,
                               openai_session_id = session_id,
                               last_session = last_session,
                               session_names = session_names,
                               session_id = session.get("uuid"))
    else:
        return abort(500)

@app.route('/process_text', methods=['POST'])
@login_required
def process_text():
    # Get JSON data from the request
    data = request.get_json()

    # Extract filename and code from the JSON data
    filename = data.get('filename')
    code = data.get('code')
    file_path = utility.write_file(filename,code,current_user.username)
    push = True
    utility.scp_file(file_path, r"/home/user/workspace",push)
    # Log messages
    app.logger.info('Received filename: %s', filename)
    app.logger.info('Received code: %s', code)

    # Return a response (you can customize this based on your needs)
    return jsonify({'status': 'success'})


@app.route('/process_feedback', methods=['POST'])
@login_required
def process_feedback():
    # Get JSON data from the request
    data = request.get_json()
    full_interaction = data.get('containerId')
    feedback = data.get('feedback')
    feedback = ['Negative','Positive','Neutral'][feedback]
    app.logger.info('Received feedback: %s', feedback)

    # Return a response (you can customize this based on your needs)
    return jsonify({'status': 'success'})
