import os

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
import openai
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

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

    def set_langchain_template_question(self, text, format_instructions=None):
        self.current_question = self.prompt_template.format_messages(
            style=self.style,
            text=text, format_instructions=format_instructions)

    def get_langchain_response(self, message=None):
        if self.current_question and message is None:
            message = self.current_question
            return self.template_chat(message)

        return self.chat(message)

    def reset_template_string(self):
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)


def sanity_test_template():
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

def sanity_test_parser():
    langchain_service = LangChainService()
    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """

    langchain_service.template_string = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """

    langchain_service.reset_template_string()
    langchain_service.set_langchain_template_question(text=customer_review)
    response = langchain_service.get_langchain_response()
    print(response.content)

    gift_schema = ResponseSchema(name="gift",
                                 description="Was the item purchased\
                                 as a gift for someone else? \
                                 Answer True if yes,\
                                 False if not or unknown.")
    delivery_days_schema = ResponseSchema(name="delivery_days",
                                          description="How many days\
                                          did it take for the product\
                                          to arrive? If this \
                                          information is not found,\
                                          output -1.")
    price_value_schema = ResponseSchema(name="price_value",
                                        description="Extract any\
                                        sentences about the value or \
                                        price, and output them as a \
                                        comma separated Python list.")

    response_schemas = [gift_schema,
                        delivery_days_schema,
                        price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    langchain_service.template_string = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product\
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    text: {text}

    {format_instructions}
    """
    langchain_service.reset_template_string()
    langchain_service.set_langchain_template_question(text=customer_review, format_instructions=format_instructions)
    response = langchain_service.get_langchain_response()
    print(response.content)
    output_dict = output_parser.parse(response.content)
    print(output_dict)




if __name__ == "__main__":
    sanity_test_template()
    sanity_test_parser()
