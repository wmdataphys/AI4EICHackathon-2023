from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_leaderboard.models import User, Team

"""
class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')
"""

class SubmitForm(FlaskForm):
    tname = ""
    uname = ""
    pword = ""
    if(current_user):
        tname = current_user.teamname
        uname = current_user.username
        pword = current_user.password
        #print (f"in SUbmit form it is {uname}")
        #pword = current_user.password
    username = StringField('User name', validators=[DataRequired()])
    teamname = StringField('Team name', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    qnumber = SelectField('Submitting Solution for Question',
                        choices=[(-1, "Select the Question"), (1, 'Question 1'),
                        (2, 'Question 2'), (3, 'Question 3')],
                        validators=[DataRequired()],
                        default = -1
                        )
    result_file = FileField(r"Upload Results file ('.csv')",
                        validators=[FileRequired(), FileAllowed(['csv'], "Only csv files please")]
                        )

    submit = SubmitField('Evaluate Results', render_kw={"onclick": "loading();"})

    def validate_qnumber(self, qnumber):
        print (qnumber.data)
        if int(qnumber.data) < 0:
            raise ValidationError("Select the Question Number corresponding to the uploaded solution")
"""
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')
"""


class LoginForm(FlaskForm):
    teamname = StringField('Team Name',
                        validators = [DataRequired()])
    username = StringField('User name',
                        validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class OpenAISessionForm(FlaskForm):
    name = StringField('Session Name', validators=[DataRequired()], default="Chat Session")
    context = StringField('Set your Context', default="")
    submit = SubmitField('Open Session')
    
    