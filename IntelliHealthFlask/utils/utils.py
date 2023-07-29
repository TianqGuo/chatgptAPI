from dotenv import load_dotenv
import os
import openai
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pinecone
from openai.embeddings_utils import get_embedding
from sklearn.preprocessing import LabelEncoder


def set_openai_key():
    """Sets OpenAI key."""
    load_dotenv()
    openai.api_key = os.getenv("apikey")


class Client:
    def __init__(self, engine="text-davinci-002"):
        self.engine = engine
        set_openai_key()

    def ask_question(self, prompt, max_tokens=180, n=1, stop=None, temperature=0.5):
        # Generate text
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )

        return response


class ModelPrediction:
    def __init__(self, model_path="model_path"):
        load_dotenv()
        # print(os.getenv("apikey"))
        # print(os.getenv("model_path"))
        # self.model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), os.getenv("model_path")))
        self.model = joblib.load(os.path.join(os.path.dirname(__file__), os.getenv("model_path")))

    def predict(self, x_input):
        test_output = self.model.predict(x_input)
        return test_output

    def get_config(self):
        return self.config

    def get_model(self):
        return self.model

    def sanity_test(self):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), os.getenv("preprocessed_data_path")))
        # y = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        #      'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        #      'troglitazone', 'tolazamide',
        #      'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        #      'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        encoder = LabelEncoder()
        df['gender'] = encoder.fit_transform(df['gender'])
        df['smoking_history'] = encoder.fit_transform(df['smoking_history'])
        y = ["diabetes"]
        x = df.drop(y, axis=1)
        test_input = np.array(x.iloc[0])
        test_input = test_input[None, :]  # shape 1x4
        print(test_input)
        print("Test input shape is: ", test_input.shape)
        test_output = self.model.predict(test_input)
        print(test_output)
        print("Test output shape is: ", test_output.shape)
        print("Sanity test passed!")


class VectorDB:
    def __init__(self, index_name="vectordatabase"):
        set_openai_key()
        self.pinecone_key = os.getenv("PINECONE_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRON")
        pinecone.init(api_key=self.pinecone_key, environment=self.pinecone_env)
        self.index = pinecone.Index(index_name)

    def get_embeddings(self, input_text, is_upsert=False):
        input_text = 'user: ' + input_text
        # model = "text-embedding"
        # if not is_upsert:
        model = "text-embedding-ada-002"
        # print("current embedding model is: ", model)
        embedding = openai.Embedding.create(
            input=input_text,
            model=model
        )
        # print the embedding (length = 1536)
        vector = embedding["data"][0]["embedding"]
        # print(vector)
        # print(len(vector))
        return vector

    def query_match(self, input_text):
        vector = self.get_embeddings(input_text)

        search_response = self.index.query(top_k=5, vector=vector, include_metadata=True)

        # print(search_response['matches'])

        # context, url = self.get_highest_score(search_response['matches'])
        context = self.get_highest_score(search_response['matches'])

        # print(context)

        prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {input_text}\nAnswer:"

        return prompt

    def get_highest_score(self, items):
        highest_score_item = max(items, key=lambda item: item["score"])

        print(highest_score_item)

        if highest_score_item["score"] > 0.8:
            # return highest_score_item["metadata"]['text'], highest_score_item["metadata"]['url']
            return highest_score_item
        else:
            return ""

    def list_indexes(self):
        print(pinecone.list_indexes())

    def create_index(self, index_name, dimension):
        pinecone.create_index(name=index_name, dimension=dimension)

    def insert_data(self, index_name, data):
        self.index.upsert(items=data)

    def sanity_test(self):
        # test 1
        self.index.upsert([
            ("A", [i for i in range(1536)]),
            ("B", [i for i in range(1536, 0, -1)]),
            ("C", (np.ones(1536) * 3).tolist()),
            ("D", [i for i in range(0, 3072, 2)]),
            ("E", [i for i in range(3072, 0, -2)]),
        ])
        response1 = self.index.query(
            vector=(np.ones(1536) * 0.3).tolist(),
            top_k=3,
            include_values=False
        )
        print(response1)

        # test 2

        self.index.upsert([("sanity test2", self.get_embeddings("What is diabetes?"))])

        self.index.upsert([("sanity test3", self.get_embeddings("How can I help you?"))])

        self.index.upsert([("sanity test4", self.get_embeddings("What is wrong with you?"))])

        response2 = self.index.query(
            vector=self.get_embeddings("What is diabetes?"),
            top_k=3,
            include_values=False
        )

        print(response2)

        response3 = self.index.query(
            vector=self.get_embeddings("is diabetes what?"),
            top_k=3,
            include_values=False
        )

        print(response3)

        print("Sanity test passed!")


if __name__ == "__main__":
    # cur_predict = ModelPrediction()
    # cur_predict.sanity_test()
    # client = Client()
    # print(os.environ['HOME'])
    # client.ask_question("What is the meaning of life?")
    # print(np.random.rand(1, 74))
    cur_vector_db = VectorDB()
    cur_vector_db.sanity_test()
    print(cur_vector_db.index.describe_index_stats())
    cur_vector_db.query_match("what is diabetes?")
