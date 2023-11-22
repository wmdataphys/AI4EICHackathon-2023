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
    
        