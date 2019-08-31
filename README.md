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

* Data retrival and pre-processing

   Due to different company profiles, some of the fundamental data does not exist for a number of companies.
   This creates some NaN values for different columns. Columns with a certain percentage od NaN values are dropped to retain more rows of data.

   Standarization of values were also applied (market cap and price are on very different scale).

```python

      data_df = Retriever(file='/tmp/retrieve_data.log', loglevel=logging.DEBUG).getData()
      steps = [('clean_data', Cleaner(file='/tmp/clean_data.log', loglevel=logging.DEBUG)),
            ('process_data', Processor(file='/tmp/process_data.log', loglevel=logging.DEBUG))]
   
      p = Pipeline(steps)
      data_array, labels = p.fit_transform(data_df)
           
```

* Regression and score calculation

Price performance from next quarter is used to create the labels for the time series regression. 
More could be done for hyperparameters tunning (e.g. GridSearch) and features engineering (e.g. PCA ) for each model.

``` python

      # random forest
      model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
      predictor = Predictor(model, 5, file='/tmp/predictor.log', loglevel=logging.DEBUG)
   
      score = predictor.score(data_array, labels)
```

* LSTM takes different steps especially with the extra step to save the model for reusing and of cause the layers setup. 

Pipeline is not used here obviously because of the set up of layers and diff

``` python

      # Split into train and test sets 7-3
      X_train, y_train, X_test, y_test = self.split_tran_test_data(data_value_arrays, labels)
        
      model = Sequential()
      model.add(LSTM(50, input_shape=(X_train.shape[0], X_train.shape[1])))
      model.add(Dense(1))
      model.compile(loss='mae', optimizer='adam')
      
      save_weights_at = os.path.join('keras_models', custom_name+'_Stock_Fundamental_RNN_weights.hdf5')
      save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                                    save_best_only=True, save_weights_only=False, mode='min',
                                    period=1)
      history = model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test), verbose=2,
                            shuffle=False, callbacks=[save_best])
```

### Results ###

RMSE for different models with different thresholds of removing columns with NaN:

|Model |0.95 threshold | 0.99 threshold |
|:---|:---|:---|
|Base line|0.157894 | 0.155359|
|Random Forest|0.123671 |0.132346 |
|Light GBM|0.128006 |0.139742 |

Error is generally larger with a more restrictive cleaning policy. This is probably because the amount of data for training the model decreases with more columns being deleted.
In addition light GBM seems to perform better than other options.

### Retrospective ###

1. The difference in performance of the models in scope is not marginal. This is probably due to the small amount
   of data for training. Question - what regression time seies ML is best for limited amount of data?
2. Deep learning is not necessarily better than traditional machine learning methodology.
3. Evaluation of the performance of time series regression can be quite tricky. More tunning needs to be done.
4. It is important to have a baseline prediction (the simplest model ever for prediction) for comparison.



### Reference to setting example Flask service in Heroku ###

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


