# Base class for language models

class Model:
    def __init__(self, name):
        """
        Args:
            name: name of the model
        """
        self.name = name
    
    def generate_response(self, prompt):
        """
        Generates a response to a prompt.
        Args:
            prompt: prompt to generate a response to
        Returns:
            response: response to the prompt
        """
        raise NotImplementedError
