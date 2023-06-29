from flask import Flask

app = Flask(__name__)


@app.route('/api/v1/health')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/api/v1/chat/<chat_content>')
def chat(chat_content):  # put application's code here
    return chat_content


if __name__ == '__main__':
    app.run()
