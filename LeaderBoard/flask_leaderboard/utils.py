import paramiko
from scp import SCPClient
import os

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
        for i in range(len(codes)):
            n,c = codes[i].split('\n',maxsplit=1)
            if n in ['python','bash']:# Lets think about this.
                 n = True
            else:
                n = False
            code.append(c)
            name.append(n)

        return code,text,name


    def write_file(self,filename,code):
        try:
            with open(str(filename), 'w') as file:
                file.write(code)
        except:
            print('Error writing your code. Likely an issue with the file name or your code cell is blank.')

    def scp_file(self,filename, destpath):
        host = r"@bora.sciclone.wm.edu"
        user = 'ksuresh'
        password = r"Pavi\$4696" 
        remote_path = destpath
        local_path = str(filename)
        os.system(f"sshpass -p \"{password}\" scp {local_path} {user}{host}:{remote_path}")
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
        sys_context_3 = """You will strictly not answer questions that are not related to programming, computer science and Hadronic physics.
        Politely decline answering any conversation that is not related to the topic."""

        return [sys_context_1, sys_context_2, sys_context_3]
