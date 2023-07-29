from dotenv import load_dotenv
import os
import openai
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pinecone
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
    def __init__(self, vector_path="vector_path"):
        load_dotenv()
        pinecone.init(api_key=os.getenv("PINECONE_KEY"), environment=os.getenv("PINECONE_ENVIRON"))
        self.cur_index = pinecone.Index('vector-database')

    def sanity_test(self):
        index = pinecone.Index('vector-database')
        index.upsert([
            ("A", [i for i in range(1024)]),
            ("B", [i for i in range(1024, 0, -1)]),
            ("C", (np.ones(1024) * 3).tolist()),
            ("D", [i for i in range(0, 2048, 2)]),
            ("E", [i for i in range(2048, 0, -2)]),
        ])
        response = index.query(
            vector=(np.ones(1024) * 0.3).tolist(),
            top_k=3,
            include_values=False
        )
        print(response)
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
    print(cur_vector_db.cur_index.describe_index_stats())
