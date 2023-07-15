import numpy as np
from flask import Flask, request, json
from flask_cors import CORS, cross_origin

from service.get_gpt_service import GptService
from service.get_model_service import ModelService

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/v1/health')
@cross_origin()
def health():  # put application's code here
    return 'Status: Healthy, Version: 1.0'


@app.route('/api/v1/chat/<chat_content>')
@cross_origin()
def chat(chat_content):
    cur_service = GptService()
    response_content = cur_service.get_gpt_response(chat_content)
    # print(type(response_content))
    # print(response_content["choices"][0])
    return response_content["choices"][0]


@app.route('/api/v1/model/prediction', methods=['POST'])
@cross_origin()
def predict():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        json_input = request.json
        print("model request inputs: ", json_input)
        cur_model = ModelService()
        print(type(json_input))
        body = json.loads(json_input['body'])
        print(type(body))
        matrix = body["data"]
        print(matrix)
        input_data = np.array(matrix)
        # print(input_data)
        # print(input_data.shape)
        prediction = cur_model.predict(input_data)
        print("prediction: ", prediction)
    else:
        return 'Content-Type not supported!', 400

    response = json.dumps({"prediction": str(prediction)})
    return response


if __name__ == '__main__':
    app.run(host='127.0.0.1')
