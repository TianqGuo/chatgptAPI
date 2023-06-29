from dotenv import load_dotenv
import os
import openai


class Client:
    def __init__(self, engine="text-davinci-002"):
        load_dotenv()
        self.key = os.getenv("apikey")
        self.engine = engine
        #
        # # Load the API key from file
        # with open(self.key, 'r') as f:
        #     api_key = f.read().strip()

        # Set up the OpenAI API client
        # print(self.key)
        openai.api_key = self.key

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
