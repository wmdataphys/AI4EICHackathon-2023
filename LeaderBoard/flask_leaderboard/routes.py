import os,sys
import secrets
import json
from flask import render_template, url_for, flash, redirect, request, abort, jsonify, session
from flask_leaderboard import app, db, bcrypt
from flask_leaderboard.forms import LoginForm, SubmitForm, OpenAISessionForm
from flask_leaderboard.models import Team, User, Question, ChatSessions, ChatInfo
from flask_login import login_user, current_user, logout_user, login_required, user_needs_refresh
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_leaderboard.evaluator import Evaluate, evaluate
import flask_leaderboard.aiapi
import flask_leaderboard.config
from flask_leaderboard.OpenAIAg.OpenAIChat import OpenAIChat
<<<<<<< HEAD
import logging
# Some default settings
from flask_leaderboard.utils import OPENAI_Utils
=======
from uuid import uuid1, uuid5
>>>>>>> 6e6556e (my changes akrthik)

utility = OPENAI_Utils()

@app.before_request
def make_session_permanent():
    session.permanent = False

@app.errorhandler(Exception)
def handle_exception(e):
    return render_template("error_500.html", e=e), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('page_not_found.html'), 404

@app.errorhandler(403)
def access_forbidden(e):
    return render_template('access_denied.html'), 403


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

@app.route("/logout")
def logout():
    logout_user()
    session.pop("openai_session", None)
    session.pop("openai_user", None)
    session.pop("uuid", None)
    session.pop("user", None)
    return redirect(url_for('leaderboard'))

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if (request.args.get("next")):
            return redirect(request.args["next"])
        else:
            return redirect(url_for('leaderboard'))
    form = LoginForm()
    if form.validate_on_submit():
        teamname = form.teamname.data
        team = Team.query.filter_by(name=teamname).first()
        user = User.query.filter_by(username=form.username.data, teamname = form.teamname.data).first()
        if user and team and bcrypt.check_password_hash(team.password, form.password.data):
            # if authenticated, Check for the last session
            session["openAI_active"] = False
            session["user"] = user.username
            login_user(user, remember=form.remember.data)
            return redirect(url_for(request.args.get("next", 'leaderboard')))
        else:
            flash('Login Unsuccessful. Please check team, username and password', 'danger')
    return render_template('login.html', title='Login', form=form, sess_id = session.get("uuid"))

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
            filename = f"Question{qnumber}_" + now + "_" + secure_filename(f.filename)
            filepath = os.path.join(user_folder, filename)
            f.save(filepath)
            status, accuracy_score = evaluate(filepath, qnumber)
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
                            tname = tname, uname = uname)

"""
def submit():
    return render_template("will_open.html")
"""
@app.route("/start_session", methods = ['GET', 'POST'])
@login_required
def start_session():
    if not current_user.is_authenticated:
        return redirect(url_for('login', next_page = 'chat'))
    form = OpenAISessionForm()
    if form.validate_on_submit():
        user_name = current_user.username
        team_name = current_user.teamname
        session_name = form.name.data
        print (f"session name is {session_name}")
        context = form.context.data
<<<<<<< HEAD
        app.config["OPENAI_USERS"][user_name].resetAndStartSession(session_name = session_name, user_context = [context])
        session_id = app.config["OPENAI_USERS"][user_name].session_id
        chatsession = ChatSessions(sessioname = session_name,
                               username = user_name,
                               session_id = session_id,
                               const_sys_context = "",
=======
        session["openAIChatAgent"] = OpenAIChat(user_name, 
                                                flask_leaderboard.config.DevelopmentConfig.OPENAI_KEY
                                                )
        
        session_index = session["user"].TotalSessions
        
        temp_str = f"{team_name}_{user_name}_{session_index}" + str(uuid1())
        session_uuid = uuid5(uuid.NAMESPACE_OID, temp_str)
        
        chatsession = ChatSessions(sessioname = session_name, 
                               username = user_name, 
                               index = session_index, 
                               uuid = session_uuid,
                               start_time = datetime.now(),
                               end_time = datetime.now(),
                               const_sys_context = "", 
>>>>>>> 6e6556e (my changes akrthik)
                               user_sys_context = context)
        session["user"].TotalSessions+=1
        db.session.add(chatsession)
        db.session.commit()
        session["openAI_active"] = True
<<<<<<< HEAD
        return redirect(url_for('chat', session_id = session_id))
    session_html = render_template('start_session.html', title = 'Start Session', form = form)
    with open("test.html", "w") as _f:
        _f.write(session_html)

    return render_template('start_session.html', title = 'Start Session', form = form)
=======
        return redirect(url_for('chat', session_id = session_uuid))
    return render_template('start_session.html', title = 'Start Session', form = form, index=session_index)

>>>>>>> 6e6556e (my changes akrthik)
@app.route("/chat_test/<session_id>", methods = ['GET', 'POST'])
def chat_test(session_id):
    sessionList = [{"name" : "Chat Session 1", "id" : 1}, {"name" : "Chat Session 2", "id" : 2}]
    chats = {"chats" : [], "tokensused" : 300, "name" : f"Chat Session {session_id}", "id" : 1}
    return render_template('chats.html', sessionsList = sessionList, chats = chats)

@app.route("/chatGPT", methods = ['POST','GET'])
@login_required
def chatGPT():
    # get the OPENAIAgent object
    user_name = current_user.username
    if(not session.get("openAI_active")):
        user_needs_refresh()
    agent = session["openAIChatAgent"]
    agent.user_input = request.form['prompt']
    chat_id = request.form['chat_id']
    return_reason = agent.Chat()
    if (return_reason != "stop"):
        print ("something is wrong with ChatGPT check it ", return_reason)
<<<<<<< HEAD
    answer = agent.output
    chatinfo = ChatInfo(username = user_name,
                        session_id = agent.session_id,
                        user_prompt = agent.user_input,
                        ai_response = agent.output,
                        system_response = "",
                        feedback = True,
=======
    
    chatinfo = ChatInfo(index = chat_id,
                        chat_id = agent.msg_id,
                        username = user_name, 
                        session_uuid = session.get("uuid"), 
                        user_prompt = agent.user_input, 
                        ai_response = agent.output, 
                        system_response = "", # feedback reponse
                        feedback = True, 
>>>>>>> 6e6556e (my changes akrthik)
                        prompt_tokens = agent.prompt_tokens,
                        completion_tokens = agent.output_tokens
                        )
    session["user"].AllChatsSessions[-1].end_time = datetime.now()
    agent.chat_count+=1
    session["user"].AllChatsSessions[-1].num_chats = agent.chat_count
    db.session.add(chatinfo)
    db.session.commit()
    if (agent.total_tokens > app.config["OPENAI_params"].MAX_TOKENS):
        pass # NEED TO CHANGE THIS

    #answer = flask_leaderboard.aiapi.generateChatResponse(prompt)
    res = {}
    res['prompt'] = agent.user_input
    res['session_id'] = agent.session_id
    code,text,download = agent.OPENAI_params.split(agent.output)
    res['code'] =  code
    res['text'] = text
    res['n_code'] = len(code)
    res['n_text'] = len(text)
    res['is_downloadable'] = download
    agent.write_file('your_code.py')
    return jsonify(res), 200

@app.route('/chat/', methods = ['GET', 'POST'])
@login_required
def chat_dummy():
    if not current_user.is_authenticated:
        return redirect(url_for('login', nex_page = 'start_session'))
    if (not session["openAI_active"] or not session.get("uuid")):
        return redirect(url_for('start_session'))
    else:
        return redirect(url_for('error_500.html', sess_id = session["uuid"]))
@app.route('/chat/<session_id>', methods = ['GET', 'POST'])
@login_required # need to be logged in to chat
def chat(session_id):
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    if (not session_id):
        return redirect(url_for('start_session', next_page = 'chat'))
    print (session_id)
<<<<<<< HEAD
    return render_template('chat.html', title='Chat Bot', **locals())
=======
    chat_id = session["user"].AllChatSessions[-1].num_chats
    return render_template('chat.html', title='Chat Bot', chat_id = chat_id)
    
>>>>>>> 6e6556e (my changes akrthik)

@app.route('/process_text', methods=['POST'])
@login_required
def process_text():
    # Get JSON data from the request
    data = request.get_json()

    # Extract filename and code from the JSON data
    filename = data.get('filename')
    code = data.get('code')
    utility.write_file(filename,code)
    utility.scp_file(filename)
    # Log messages
    app.logger.info('Received filename: %s', filename)
    app.logger.info('Received code: %s', code)

    # Return a response (you can customize this based on your needs)
    return jsonify({'status': 'success'})
