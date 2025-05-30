# Class for Anthropic Claude model
import json
import os

import anthropic

from language_models.model import Model
from language_models.utils import add_retries, limiter

# set anthropic config
apikey = None
ANTRHOPIC_CONFIG_PATH = "[INSERT YOUR PATH HERE]"
if os.path.exists(ANTRHOPIC_CONFIG_PATH):
    with open(ANTRHOPIC_CONFIG_PATH, 'r') as f:
        anthropic_config = json.load(f)
    apikey = anthropic_config["ANTHROPIC_API_KEY"]
    # set as enviornment variable
    os.environ["ANTHROPIC_API_KEY"] = apikey
else:
    print(f"WARNING: Did not find anthropic config at {ANTRHOPIC_CONFIG_PATH}")
    print("Won't be able to use open anthropic models.")


class Claude(Model):
    def __init__(self, name, max_tokens=256, temperature=0.7):
        """
        Args:
            name: name of the model
            temperature: temperature parameter of model
        """
        super().__init__(name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Client()
    
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
        completions = []
        for _ in range(n_completions):
            response = self.client.messages.create(
                model=self.name, 
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
            {"role": "user", "content": prompt}
            ])
            completions.append(response.content[0].text)
        return completions