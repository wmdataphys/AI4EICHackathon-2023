from openai import OpenAI
#from MessageDB import CustomMessageConverter
#from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

class OpenAIChat:
    def __init__(self, user_name: str, api_key: str, db_connection: str):
        self.username = user_name
        self.OpenAIClient = OpenAI(api_key=api_key)
        self.session_id = 1 # each time the buffer is clear near session comes in
        self.memory = None
        self.msgs = []
        self.prompt = None
        self._system_context = self._setSystemContext()
        self._db_connection = db_connection
        #self._dbManager = SQLChatMessageHistory(self.username, 
        #                                        self._db_connection, 
        #                                        CustomMessageConverter(self.username, self.session_id)
        #                                        )
    
    def Chat(self, ques = None):
        if (ques == None):
            return "No input provided"
        self.msgs.append({"role" : "user", "content" : ques})
        output = self.OpenAIClient.chat.completions.create(model="gpt-3.5-turbo-1106", 
                                                messages=self.msgs,
                                                )
        self.msgs.append({"role" : "assistant", "content" : output.choices[0].message.content})
        #feedback = input(">> Is the response correct? (y/n): ")
        #print (">> AI: Thanks for the feedback. Duely noted the reponse")
        """
        if(feedback == "y"):
            self.msgs.append({"role" : "system", "content" : "Thanks for the feedback."})
        else:
            self.msgs.append({"role" : "system", "content" : "Sorry for the inconvenience. We will try to improve."})
        """
        
        return output.choices[0].message.content
        # store the messages in the database
        #self._dbManager.add_user_message(HumanMessage(content = ques))
        #self._dbManager.add_ai_message(AIMessage(content = output.choices[0].message.content))
        #self._dbManager.add_system_message(SystemMessage(content = self.msgs[-1].content))
            
    
    def _setSystemContext(self, context_msgs = None) -> list:
        if (not context_msgs):
            context_1 = """You are an expert python programmer, very proficient in the following python packages. 
            1. numpy 
            2. pandas
            3. pytorch especially using cuda for GPU acceleration
            4. hdf5 
            5. tensorflow
            """
            self.msgs.append({"role" : "system", "content" : context_1})
            context_2 = """You are very critical in writing code with no Run Time errors. You can write code snippets in python."""
            #self._dbManager.add_system_message(SystemMessage(content = context_1))
            self.msgs.append({"role" : "system", "content" : context_2})
            #self._dbManager.add_system_message(SystemMessage(content = context_2))
        else:
            for msg in context_msgs:
                self.msgs.append({"role" : "system", "content" : msg})
                #self._dbManager.add_system_message(SystemMessage(content = msg))
        return self.msgs
    