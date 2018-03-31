from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model 
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import os


class FundamentalWorker(object):
    def __init__(self):
        pass

    def build_save_model_MLP(self, data_value_arrays, labels, plot=True):
        pass

    def build_save_model_LSTM(self, data_value_arrays, labels, plot=True):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_labels = scaler.fit_transform(labels)

        # Split into train and test sets 7-3
        test_X, test_Y, train_X, train_Y = self.split_tran_test_data(data_value_arrays, scaled_labels)

        print('------ test_Y after split-------')
        print(test_Y)
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        # print(keras.__version__)
        # fit the network
        save_weights_at = os.path.join('keras_models',
                                       'Stock_Fundamental_RNN_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
        save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                                    save_best_only=True, save_weights_only=False, mode='min',
                                    period=1)
        history = model.fit(train_X, train_Y, epochs=30, batch_size=10, validation_data=(test_X, test_Y), verbose=2,
                            shuffle=False, callbacks=[save_best])

        # plot history
        if plot:
            self.plot_result(history)

        return save_weights_at, (test_X, test_Y), scaler

    def plot_result(self, history):
        pass
        # pyplot.plot(history.history['loss'], label='train')
        # pyplot.plot(history.history['val_loss'], label='test')
        # pyplot.legend()
        # pyplot.show()

    def split_tran_test_data(self, data_value_arrays, labels):
        train_number = int(0.7 * data_value_arrays.shape[0])
        train_X, train_y = data_value_arrays[:train_number, :], labels[:train_number]
        test_X, test_y = data_value_arrays[train_number:, :], labels[train_number:]
        print('---- split ----')
        print(train_number)
        print(labels)
        return test_X, test_y, train_X, train_y

    def predict(self, save_weights_at, test_set, scaler):
        model = load_model(save_weights_at)
        test_X = test_set[0]
        test_y = test_set[1]

        predicted_y = model.predict(test_X)
        # No need to invert transform scaling? - basically the labels are not part of the input data and have different
        #  scaling. We might need to separately scale the label.
        inv_predicted_y = scaler.inverse_transform(predicted_y)
        inv_test_y = scaler.inverse_transform(test_y)

        # predicted_y is in the shape [[1],[2],[0]], test_y is [1,2,0], flatten it to shape of test_y
        print("---- Before reshape ----")
        print(inv_predicted_y)
        inv_predicted_y_to_test_shape = inv_predicted_y.flatten().tolist()
        rmse = sqrt(mean_squared_error(inv_predicted_y_to_test_shape, inv_test_y))

        return (inv_predicted_y_to_test_shape,inv_test_y), rmse

        # Scale it back to real values
        # For the predicted y
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # predicted_y_with_test = concatenate((predicted_y, test_X[:, 1:]), axis=1)  # why?
        # inv_predicted_y = scaler.inverse_transform(predicted_y_with_test)

        # For the actual y
