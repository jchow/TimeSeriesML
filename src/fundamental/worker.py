import logging
import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from loggermixin import LoggerMixin
from predictor import Predictor
from cleaner import Cleaner
from dataprocessor import Processor
from dataretriever import Retriever


class Worker(LoggerMixin, object):
    def __init__(self, file='temp.log', loglevel=logging.INFO):
        super().__init__(file, loglevel)
        self.logger.debug('Created an instance of %s', self.__class__.__name__)

    @staticmethod
    def get_test_prediction(data_value_arrays, labels, model):
        p = Predictor(model, file='/tmp/predict_gbm.log', loglevel=logging.DEBUG)
        y_predicted_df = p.predict(data_value_arrays, labels)
        rmse = p.score(data_value_arrays, labels)
        return rmse, y_predicted_df

    def select_model(self):
        model_rmse = {}

        data_df = Retriever(file='/tmp/retrieve_data.log', loglevel=logging.DEBUG).getData()
        steps = [('clean_data', Cleaner(file='/tmp/clean_data.log', loglevel=logging.DEBUG)),
                 ('process_data', Processor(file='/tmp/process_data.log', loglevel=logging.DEBUG))]

        p = Pipeline(steps)
        data_array, labels = p.fit_transform(data_df)

        # random forest
        model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
        predictor = Predictor(model, 5, file='/tmp/predictor.log', loglevel=logging.DEBUG)

        score = predictor.score(data_array, labels)
        model_rmse[model] = score

        self.logger.info('=== Random Forest rmse : {}'.format(score))


        # LGB
        model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
        predictor = Predictor(model, 5, file='/tmp/predictor.log', loglevel=logging.DEBUG)

        score = predictor.score(data_array, labels)
        model_rmse[model] = score

        self.logger.info('=== LGB rmse : {}'.format(score))

        # Baseline
        _, rmse = self.predict_baseline(data_array, labels)
        model_rmse[None] = rmse

        self.logger.info('=== baseline rmse : {}'.format(rmse))

        # LSTM
        save_weights_at, test_set = self.build_save_model_LSTM(data_array, labels, 'intrinio')
        model, y, rmse = self.predict(save_weights_at, test_set)
        model_rmse[model] = rmse
        self.logger.info('=== baseline rmse : {}'.format(rmse))

        # Select the model with min rmse
        m = min(model_rmse, key=model_rmse.get)
        return m, model_rmse[m]


    '''
    For the simplest guess for future performance, use the performance of T-1 to predict T
    '''
    def predict_baseline(self, data_array, labels):
        # Split into train and test sets 7-3
        X_train, y_train, X_test, y_test = self.split_tran_test_data(data_array, labels)

        y_predicted = y_test[:-1]
        y_test_shift = y_test[1:]
        rmse = Predictor.reg_error(y_predicted, y_test_shift)

        return y_predicted, rmse

    def predict_light_gbm(self, data_value_arrays, labels):
        model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
        rmse, y_predicted_df = self.get_test_prediction(data_value_arrays, labels, model)
        return y_predicted_df, rmse

    def predict_random_forest(self, data_value_arrays, labels):
        model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
        rmse, y_predicted_df = self.get_test_prediction(data_value_arrays, labels, model)
        return y_predicted_df, rmse

    def build_save_model_LSTM(self, data_value_arrays, labels, custom_name=''):

        # Split into train and test sets 7-3
        X_train, y_train, X_test, y_test = self.split_tran_test_data(data_value_arrays, labels)

        self.logger.debug('------ X_test shape = %s', X_test.shape)
        self.logger.debug('------ y_test shape = %s', X_test.shape)
        self.logger.debug('------ X_train shape = %s', X_train.shape)
        self.logger.debug('------ y_train shape = %s', X_train.shape)

        print('X_train shape:', X_train.shape)
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[0], X_train.shape[1])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        self.logger.info('------- Model Summary -------')
        self.logger.info(model.summary())

        # fit the network
        save_weights_at = os.path.join('keras_models',
                                       custom_name+'_Stock_Fundamental_RNN_weights.hdf5')
        save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                                    save_best_only=True, save_weights_only=False, mode='min',
                                    period=1)
        history = model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test), verbose=2,
                            shuffle=False, callbacks=[save_best])

        self.logger.debug("History: %s", history)

        return save_weights_at, (X_test, y_test)

    def split_tran_test_data(self, data_value_arrays, y):
        # why not use sklearn train test split?
        # Version of sklearn below 0.21 does not have shuffle as parameter
        # X_train, X_test, y_train, y_test = train_test_split(data_value_arrays, y, test_size=0.2, shuffle=False)

        # For time series this can be done easily

        train_number = int(0.8 * data_value_arrays.shape[0])
        X_train, y_train = data_value_arrays[:train_number, :], y[:train_number]
        X_test, y_test = data_value_arrays[train_number:, :], y[train_number:]

        self.logger.debug("-- Split data -- ")
        return X_train, y_train, X_test, y_test

    def predict(self, save_weights_at, test_set):
        model = load_model(save_weights_at)

        test_X = test_set[0]
        test_y = test_set[1]

        predicted_y = model.predict(test_X)

        self.logger.debug('------ test y  -----')
        self.logger.debug(test_y)

        self.logger.debug('------ predicted y -----')
        self.logger.debug(predicted_y)

        flatten_predicted_y = predicted_y.flatten().tolist()

        # Found the predicted error
        rmse = self.reg_error(flatten_predicted_y, test_y)

        return model, flatten_predicted_y, rmse

