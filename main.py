from flask import Flask, redirect, render_template, request
from gevent.wsgi import WSGIServer
import logging
import sys

app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/')
def hello():
    return "Hello World"

@app.route('/classify/ticker', methods=['POST'])
def classify():
    return redirect('/')
    
if __name__ == '__main__':
    app.debug = True
    
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()