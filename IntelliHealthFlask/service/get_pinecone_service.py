import os

import openai
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.vectorstores import Pinecone
import pinecone

from service.get_langchain_service import set_langchain_key
from utils.utils import set_openai_key


class PineconeService:
    def __init__(self, index_name="vectordatabase", text_field="text"):
        set_langchain_key()
        self.pinecone_key = os.getenv("PINECONE_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRON")
        pinecone.init(api_key=self.pinecone_key, environment=self.pinecone_env)
        self.index = pinecone.Index(index_name)

    def query_match(self, input_text):
        input_text = 'user: ' + input_text
        embedding = openai.Embedding.create(
            input=input_text,
            model="text-embedding-ada-002"
        )
        # print the embedding (length = 1536)
        vector = embedding["data"][0]["embedding"]

        search_response = self.index.query(top_k=5, vector=vector, include_metadata=True)

        context, url = self.get_highest_score(search_response['matches'])

        print(context, url)

        prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {input_text}\nAnswer:"

        return prompt

    def get_highest_score(self, items):
        highest_score_item = max(items, key=lambda item: item["score"])

        if highest_score_item["score"] > 0.8:
            return highest_score_item["metadata"]['text'], highest_score_item["metadata"]['url']
        else:
            return ""

    def list_indexes(self):
        print(pinecone.list_indexes())

    def create_index(self, index_name, dimension):
        pinecone.create_index(name=index_name, dimension=dimension)

if __name__ == "__main__":
    pinecone_service = PineconeService()
    pinecone_service.list_indexes()
    pinecone_service.query_match("what is diabetes?")



