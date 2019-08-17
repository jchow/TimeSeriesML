from flask import Flask, redirect, render_template, request
from gevent.wsgi import WSGIServer
import logging
import sys
from exampleclassifier import ExampleClassifier
from model import ModelWorker
from fundamental.fundamentalmodeldatapreparer import FundamentalModelDataPreparer
from fundamental.worker import Worker

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

'''
Two approaches were attempted:

1. Not using ticket as one of the feature. Assume these tech stocks are from the same 
industry and so should have similar behaviours.
Results are not good as this assumption is quite not correct given the behaviour between
aapl and amzn are quite different already.

2. Use ticker as a feature and so data from each company are not the same. For prediction
it only makes sense if the to be predicted input is from one of these companies.

'''
@app.route('/build/fundamental/notickerfeature')
def build_fundamentals():
    tickers = ['MSFT', 'AAPL']

    preparer = FundamentalModelDataPreparer(file='/tmp/test_preparer_notickerfeature.log', loglevel=logging.DEBUG)
    data_array, labels = preparer.get_dataset_for_RNN(tickers)

    worker = Worker(file='/tmp/test_worker_notickerfeature.log', loglevel=logging.DEBUG)

    save_weights_at, test_set = worker.build_save_model_LSTM(data_array, labels, 'intrinio')
    y_predicted_df, rmse = worker.predict(save_weights_at, test_set)
    return "Fitting result:{} \n rmse:{}".format(y_predicted_df, rmse)


@app.route('/build/fundamental/tickerfeature')
def build_fundamentals():

    worker = Worker(file='/tmp/test_worker_tickerfeature.log', loglevel=logging.DEBUG)
    model, _ = worker.select_model()

    preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
    data_array, labels = preparer.get_data_from_intrinio_file()
    rmse, y_predicted_df = worker.get_test_prediction(data_array, labels, model)
    return "Fitting result:{} \n rmse:{}".format(y_predicted_df, rmse)

    
if __name__ == '__main__':
    app.debug = True
    
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()