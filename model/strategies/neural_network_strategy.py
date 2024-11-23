from model.cnn_strategy import CNNStrategy
from sklearn.neural_network import MLPClassifier
import pandas as pd

"""
Strategy class for Neural Network
"""
class NeuralNetworkStrategy(CNNStrategy):
    def __init__(self, model_name: str = "NeuralNetwork", hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)
        self.model_name = model_name

    def train(self, data):
        self.model.fit(data.X_train, data.y_train)

    def predict(self, data: pd.Series):
        return self.model.predict(data)

    def evaluate(self, data):
        print("evaluate neural network model")