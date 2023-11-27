import os

class DB_Utils:
    def __init__(self):
        self.HOST = "ai4eichackathon.mysql.pythonanywhere-services.com"
        self.USERNAME = "ai4eichackathon"
        self.PASSWORD = "Hack_2023"
        self.DB_NAME = "ai4eichackathon$users"
    def getDB_URL(self):
        return self.DB_URL
    def getDB_NAME(self):
        return self.DB_NAME
        
class OPENAI_Utils:
    def __init__(self):
        self.MAX_TOKENS = 4096
        self.GPT_MODEL = "gpt-3.5-turbo-1106"
        self.TEMPERATURE = 1.0

    def getMaxTokens(self):
        return self.MAX_TOKENS

    def getGPTModel(self):
        return self.GPT_MODEL

    def getTemperature(self):
        return self.TEMPERATURE

    def split(self,input):
        delimiter = "```"
        splits = input.split(delimiter)
        codes = splits[1::2]
        text = splits[::2]

        code = []
        name = []
        print(codes)
        for i in range(len(codes)):
            n,c = codes[i].split('\n',maxsplit=1)
            if n in ['python','bash']:# Lets think about this.
                 n = True
            else:
                n = False
            code.append(c)
            name.append(n)

        return code,text,name


    def write_file(self,filename,code,username):
        filename = str(filename)
        username = str(username)
        try:
            if not os.path.exists(os.path.join(r"../../",username)):
                os.makedirs(os.path.join(r"../../",username))
            file_path = os.path.join(r"../../",username,filename)
            with open(file_path, 'w') as file:
                file.write(code)
            print('Wrote file.')
            return file_path
        except:
            print('Error writing your code. Likely an issue with the file name or your code cell is blank.')

    def scp_file(self,filename, destpath,push):
        if push: # Push to AWS instance
            host = r"@18.234.234.7"
            user = 'admin'
            password = r"15Ad456Hck"
            remote_path = destpath
            local_path = filename
            #print(filename)
            os.system(f"sshpass -p \"{password}\" scp {local_path} {user}{host}:{remote_path}")
        else: # Pull from their aws instance given a file_path
            host = r"@18.234.234.7"
            user = 'admin'
            password = r"15Ad456Hck"
            local_path = destpath
            remote_path = filename
            #print(filename)
            #print(local_path)
            os.system(f"sshpass -p \"{password}\" scp {user}{host}:{remote_path} "+ local_path.replace(" ","\ "))

        """
        ssh = paramiko.SSHClient()
        private_key = paramiko.RSAKey(filename=private_key_path)

        ssh.connect(host, username=user, pkey=private_key)

        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path)

        ssh.close()
        """


    def getDefaultContexts(self):
        """_summary_ : This function returns the default context for the chatbot"""
        # To do This is to be reviewed by @james
        sys_context_1 = """You are an expert python programmer, very proficient in the following python packages.
        1. numpy
        2. pandas
        3. pytorch especially using cuda for GPU acceleration
        4. hdf5
        5. tensorflow
        """
        sys_context_2 = """You are very critical in writing code with no Run Time errors. You can write code snippets in python."""
        sys_context_3 = """You will strictly not answer questions that are not related to programming, computer science and Hardonic physics.
        Politely decline answering any conversation that is not related to the topic."""

        return [sys_context_1, sys_context_2, sys_context_3]
