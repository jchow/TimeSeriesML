## Price performance prediction of single stocks using fundamental data ##

Fundamental data includes values financial ratios like PE, revenues, market capitals etc.
Time series regression is performed using various machine learning methods on a set of technology stocks.

The RMSE (root mean square error) using the pervious data as prediction is used to compare with that of Random Forest, Light GBM, RNN/LSTM.

### Data ###
Data was taken from [Intrinio](https://intrinio.com/) which provides free data for limited stocks and periods. 

There are two approaches to treat the time series fundamental data coming from different stocks:

1. Treating the data as if they are all from one stock. 
   This assumes a single model can predict for these different stocks. Advantage is we have more data to train the model.
2. Stocks are differnt even they are from the technology/software industry.
   The stock symbol will be a feature that distinguish the data.
   This helps to include the relationships and dependences among different stocks. However the amount of data is small.
   There are only about 4 sets of data per year per stock. Companies with longer history were selected to provide more data.
   
### Performance measurement ###
Cross validation and RMSE (root mean square error) is used to compare performance between different models.
As a baseline performance, rmse was obtained for a prediction using the data from last quarter as the predicted values.

A forward-chaining cross validation is performed to produce rmse for each model(except for LSTM).

Other cross validation [methods](https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9) should be considered for better performance comparison 

### High level steps ###



### What is this repository for? ###

Deep learning for various basic examples and ultimately for fundamental analysis in stock market.

### How do I get set up? ###

Reference 
https://github.com/bzamecnik/deep-instrument-heroku
https://realpython.com/blog/python/flask-by-example-part-1-project-setup/

Add git remote:

e.g. heroku git:remote -a deepfundamental-stage

Fix gunicorn not found:
heroku run pip install gunicorn

Deploy to Heroku(with piplines dev/staging):
1. heroku create deepfundamental-staging --remote dev
2. heroku fork -a deepfundamental-staging deepfundamental
3. git remote add staging https://git.heroku.com/deepfundamental-staging.git
4. git push staging master
5. (create deepfundamental similarly)
6. heroku pipelines:create -a deepfundamental
7. heroku pipelines:add -a deepfundamental-staging deepfundamental
8. heroku pipelines:add -a deepfundamental deepfundamental
9. heroku pipelines:promote -r staging

### Contribution guidelines ###


