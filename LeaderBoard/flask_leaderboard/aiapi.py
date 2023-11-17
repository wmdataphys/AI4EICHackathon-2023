from openai import OpenAI
import flask_leaderboard.config
import sys

api_key = flask_leaderboard.config.DevelopmentConfig.OPENAI_KEY
client = OpenAI(api_key=api_key)


def generateChatResponse(prompt):
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})

    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)

    response = client.chat.completions.create(model="gpt-3.5-turbo",messages=messages)

    try:
        answer = response.choices[0].message.content.replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI!'

    return answer
