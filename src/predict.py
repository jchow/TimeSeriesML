from model import ModelWorker

class ExampleClassifier(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
    def predict(self):
        X_test = [[4.7,3.1,1.8,0.5], [6.3,2.3,4.4,1.3]] # to be taken from user input
        
        worker = ModelWorker('models')
        model = worker.load_model('iris_model')
        predictions = model.predict(X_test)
        
        result_predictions = str(predictions)
        # result_labels = str(self.inverse_encode(predictions, labels))
        
        return '\n\n' + result_predictions    
        
if __name__ == '__main__':
        
    model = ExampleClassifier()
    print(model.predict())

