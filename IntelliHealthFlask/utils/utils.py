from dotenv import load_dotenv
import os
import openai
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder


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


if __name__ == "__main__":
    cur_predict = ModelPrediction()
    cur_predict.sanity_test()
    # print(np.random.rand(1, 74))
