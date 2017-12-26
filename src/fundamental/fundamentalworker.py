

class FundamentalWorker(object):
    def __init__(self):
        pass

    def build_model(self, data_value_arrays):
        # Split into train and test sets 7-3
        train_number = int(0.7*len(data_value_arrays))

        train = data_value_arrays[:train_number, :]
        test = data_value_arrays[train_number:, :]

        # Split into input and labels
        train_X, train_Y = train[:, :-1], train[:, -1]
        test_X, test_Y = test[:, :-1], test[:, -1]

        # reshape input to be 3D [samples, timesteps, features]

        pass
