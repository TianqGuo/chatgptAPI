import numpy as np
from flask import Flask, request, json

from service.get_gpt_service import GptService
from service.get_model_service import ModelService

app = Flask(__name__)


@app.route('/api/v1/health')
def health():  # put application's code here
    return 'Status: Healthy, Version: 1.0'


@app.route('/api/v1/chat/<chat_content>')
def chat(chat_content):
    cur_service = GptService()
    response_content = cur_service.get_gpt_response(chat_content)
    return response_content


@app.route('/api/v1/model/prediction', methods=['POST'])
def predict():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        json_input = request.json
        print(json_input)
        cur_model = ModelService()
        input_data = np.array(json_input["data"])
        print(input_data)
        print(input_data.shape)
        prediction = cur_model.predict(input_data)
    else:
        return 'Content-Type not supported!', 400

    response = json.dumps({"prediction": str(prediction)})
    return response


if __name__ == '__main__':
    app.run()
