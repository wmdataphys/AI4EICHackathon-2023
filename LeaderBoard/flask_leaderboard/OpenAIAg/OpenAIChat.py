from openai import OpenAI
from typing import Dict, List, Any, Text
from flask_leaderboard.utils import OPENAI_Utils
#from MessageDB import CustomMessageConverter
#from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

class OpenAIChat:
    def __init__(self, user_name: Text, api_key: Text, session_id: int = 1):
        self.username = user_name
        self.OpenAIClient = OpenAI(api_key=api_key)
        self.session_name = None
        self.session_id = 0 # each time the buffer is clear near session comes in
        self.memory = None
        self.msgs = []
        self.msg_id = None
        self.user_input = None
        self.output = None
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.OPENAI_params = OPENAI_Utils()
        self.python_file = None

        # private variables
        self._setDefaultSystemContext()

    def setUserInput(self, user_input: str):
        self.user_input = user_input

    def setUserDefaultContext(self, user_context: list = []):
        for context in user_context:
            self.msgs.append({"role" : "user", "content" : context})

    def getMessages(self):
        return self.msgs

    def resetAndStartSession(self, session_name: str, user_context: list = []):
        self.session_name = session_name
        self.msgs = []
        self._setDefaultSystemContext()
        if(len(user_context) > 0):
            self.setUserDefaultContext(user_context)
        self.used_tokens = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.session_id += 1
        self.memory = None

    def write_file(self,filename='your_code.py'):
        if self.python_file is not None:
            with open(filename, 'w') as file:
                file.write(self.python_file)

    def Chat(self):
        self.msgs.append({"role" : "user", "content" : self.user_input})
        return_string = ""
        try:

            output = self.OpenAIClient.chat.completions.create(model=self.OPENAI_params.GPT_MODEL,
                                                    messages=self.msgs,
                                                    temperature=self.OPENAI_params.TEMPERATURE,
                                                    max_tokens=self.OPENAI_params.MAX_TOKENS,
                                                    )
            self.msg_id = output.id
            self.output = output.choices[0].message.content
            self.output_tokens = output.usage.completion_tokens
            self.prompt_tokens = output.usage.prompt_tokens
            self.total_tokens = output.usage.total_tokens
            self.msgs.append({"role" : "assistant", "content" : output.choices[0].message.content})
            return_string = output.choices[0].finish_reason
        except Exception as e:
            return_string = e
        return return_string

    def _setDefaultSystemContext(self, context_msgs= None):
        if (not context_msgs):
            for msg in self.OPENAI_params.getDefaultContexts():
                self.msgs.append({"role" : "system", "content" : msg})

        else:
            for msg in context_msgs:
                self.msgs.append({"role" : "system", "content" : msg})
