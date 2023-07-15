import os

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
import openai

from utils.utils import set_openai_key


def set_langchain_key():
    set_openai_key()
    os.environ["OPENAI_API_KEY"] = openai.api_key


class LangChainService:
    def __init__(self):
        set_langchain_key()
        self.template_string = ""
        self.style = ""
        self.template_chat = ChatOpenAI(temperature=0.0)
        self.chat = OpenAI(temperature=0.0)
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)
        self.current_question = None

    def set_langchain_template_question(self, text):
        self.current_question = self.prompt_template.format_messages(
            style=self.style,
            text=text)

    def get_langchain_response(self, message=None):
        if self.current_question and message is None:
            message = self.current_question
            return self.template_chat(message)

        return self.chat(message)

    def reset_template_string(self):
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)


if __name__ == "__main__":
    langchain_service = LangChainService()
    langchain_service.template_string = """Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"""
    langchain_service.style = """American English in a calm and respectful tone"""
    langchain_service.reset_template_string()
    customer_email = """Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, \
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """
    langchain_service.set_langchain_template_question(customer_email)
    print(langchain_service.current_question)
    print(langchain_service.get_langchain_response("How to learn ai tech quickly?"))
    print(langchain_service.get_langchain_response())

