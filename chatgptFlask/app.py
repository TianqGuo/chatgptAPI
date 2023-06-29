from flask import Flask
from markupsafe import escape

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/test/<name>')
def test(name):  # put application's code here
    return f"Hello, {escape(name)}!"

if __name__ == '__main__':
    app.run()
