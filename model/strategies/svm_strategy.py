from sklearn.svm import SVC
from model.cnn_strategy import CNNStrategy
import pandas as pd

class SVMStrategy(CNNStrategy):
    def __init__(self, model_name, kernel="linear", C=1.0):
        self.model_name = model_name
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, data):
        print(f"Training {self.model_name}...")
        self.model.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        print(f"Predicting with {self.model_name}...")
        return self.model.predict(X_test)

    def evaluate(self, data):
        print(f"Evaluating {self.model_name}...")
        accuracy = self.model.score(data.X_test, data.y_test)
        print(f"Test Accuracy: {accuracy}")
