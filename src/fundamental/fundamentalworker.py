from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


class FundamentalWorker(object):
    def __init__(self):
        pass

    def build_model(self, data_value_arrays, labels, plot=True):
        # Split into train and test sets 7-3
        test_X, test_Y, train_X, train_Y = self.split_tran_test_data(data_value_arrays, labels)

        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        # print(keras.__version__)
        # fit the network
        history = model.fit(train_X, train_Y, epochs=30, batch_size=10, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

        # plot history
        if plot:
            self.plot_result(history)

        return model, (test_X, test_Y)

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

        return test_X, test_y, train_X, train_y

    def predict(self, model, test_set):
        test_X = test_set[0]
        test_y = test_set[1]

        predicted_y = model.predict(test_X)
        # test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])

        return predicted_y

        # Scale it back to real values
        # For the predicted y
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # predicted_y_with_test = concatenate((predicted_y, test_X[:, 1:]), axis=1)  # why?
        # inv_predicted_y = scaler.inverse_transform(predicted_y_with_test)

        # For the actual y
