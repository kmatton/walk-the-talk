# Class for Open AI Chat GPT model
import json
import os

import openai

from language_models.model import Model
from language_models.utils import add_retries, limiter


# set open ai config
apikey = None
OPENAI_CONFIG_PATH = "[INSERT YOUR PATH HERE]"
if os.path.exists(OPENAI_CONFIG_PATH):
    with open(OPENAI_CONFIG_PATH, 'r') as f:
        open_ai_config = json.load(f)
    apikey = open_ai_config['OPENAI_API_KEY']
else:
    print(f"WARNING: Did not find open ai config at {OPENAI_CONFIG_PATH}")
    print("Won't be able to use open ai models.")
openai.api_key = apikey


class ChatGPT(Model):
    def __init__(self, name, temperature=0.7):
        """
        Args:
            name: name of the model
            temperature: temperature parameter of model
        """
        super().__init__(name)
        self.temperature = temperature
    
    @add_retries
    @limiter.ratelimit('identity', delay=True)
    def generate_response(self, prompt, n_completions=1):
        """
        Generates a response to a prompt.
        Args:
            prompt: prompt to generate a response to
            n_completions: number of completions to generate
        Returns:
            response: response to the prompt
        """
        choices = openai.ChatCompletion.create(
            model=self.name, temperature=self.temperature, messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ], n=n_completions)["choices"]
        return [choice["message"]["content"] for choice in choices]
