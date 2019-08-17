from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import numpy
import pandas

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

def compileModel():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class ModelWorker(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
         
    def save_model(self, model, model_name):
        # serialize model to JSON
        model_json = model.to_json()
        filename = self.model_dir + "/" + model_name
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(filename + ".h5")
        print("Saved model to disk")
        
    def load_model(self, model_name):
        filename = self.model_dir + '/' + model_name
        with open(filename + ".json", "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(filename + ".h5")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Loaded model from disk")
        return model
        
    def load_dataset(self, data_name):
        # load dataset
        dataframe = pandas.read_csv("data/"+ data_name + ".data", header=None)
        dataset = dataframe.values
        X = dataset[:,0:4].astype(float)
        Y = dataset[:,4]
        return X,Y
        
    def inverse_encode(self, predictions, labels):
        encoder = LabelEncoder()
        encoder.fit(labels)
        return encoder.inverse_transform(predictions)
        
    def transform_one_hot(self, labels):
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoded_Y = encoder.transform(labels)
        # convert integers to dummy variables (i.e. one hot encoded)
        return np_utils.to_categorical(encoded_Y)
        
    def evaluate_model(self, model_name):
        X_train, X_test, Y_train, Y_test = self.getDataset(model_name)
        estimator = KerasClassifier(build_fn=compileModel, nb_epoch=200, batch_size=5, verbose=0)
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        
        # should have validation set instead
        results = cross_val_score(estimator, X_test, Y_test, cv=kfold)
        return results
        
    def build_model(self, model_name):
        X_train, X_test, Y_train, Y_test = self.getDataset(model_name)
        model = compileModel()
        model.fit(X_train, Y_train, nb_epoch=200, batch_size=5, verbose=0)
        
        # Save model
        self.save_model(model, model_name)
        
    def getDataset(self, model_name):
        X, labels = self.load_dataset(model_name)
        one_hot_labels = self.transform_one_hot(labels)       
        X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_labels, test_size=0.33, random_state=seed)
        return X_train, X_test, Y_train, Y_test

