from sklearn.neural_network import MLPClassifier
from model.base import BaseModel


class NeuralNetworkModel(BaseModel):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam'):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)