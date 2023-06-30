from utils.utils import Client


class GptService:
    def __init__(self):
        self.client = Client()

    def get_gpt_response(self, **kwargs):
        return self.client.ask_question(**kwargs)

    def set_gpt_temperature(self, temp):
        self.client.temperature = temp

