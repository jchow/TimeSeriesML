from flask import Flask, redirect, render_template, request
from gevent.wsgi import WSGIServer
import logging
import sys
from predict import ExampleClassifier
from model import ModelWorker
from fundamental.fundamentalmodeldatapreparer import FundamentalModelDataPreparer
from fundamental.fundamentalworker import FundamentalWorker

app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def hello():
    return "Hello World"

@app.route('/classify/example')
def classify():
    model = ExampleClassifier('models')
    return model.predict()
    
@app.route('/evaluate/model')
def evaluate():
    worker = ModelWorker('models')
    results = worker.evaluate_model('iris')
    return "Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)
    
@app.route('/build/model')
def build():
    worker = ModelWorker('models')
    worker.build_model("iris")
    return "Done building and compiling model"


@app.route('/build/fundamental')
def buildFundamentals():
    preparer = FundamentalModelDataPreparer()
    worker = FundamentalWorker()
    tickers = ['MSFT', 'AAPL']
    dataset, labels = preparer.get_dataset_for_RNN(tickers)
    model, test_set = worker.build_save_model_LSTM(dataset, labels)
    result = worker.predict(test_set, )
    return "Fitting result: " + result

    
if __name__ == '__main__':
    app.debug = True
    
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()