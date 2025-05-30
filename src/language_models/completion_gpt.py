# Class for Open AI Completion GPT model
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


class CompletionGPT(Model):
    def __init__(self, name, max_tokens=256, temperature=.7, logprobs=None):
        super().__init__(name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logprobs = logprobs

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
        # NOTE: depending on model used, need to adapt max tokens to be less than an upper bound on max tokens
        # for text-davinci-003, this is 4097
        choices = openai.Completion.create(
            model=self.name, prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens,
            n=n_completions, logprobs=self.logprobs)["choices"]
        return [choice["text"] for choice in choices]
    