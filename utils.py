import openai
import os


def list_models():
    return openai.Model.list()


class Client:
    def __init__(self, key="openai_api_key.txt", engine="text-davinci-002"):
        self.key = key
        self.engine = engine

        # Load the API key from file
        with open(self.key, 'r') as f:
            api_key = f.read().strip()

        # Set up the OpenAI API client
        openai.api_key = api_key

    def ask_question(self, prompt):
        # Generate text
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=60,
            n=1,
            stop=None,
            temperature=0.5,
        )

        return response

