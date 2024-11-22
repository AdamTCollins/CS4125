# Neural Network Model

# Imports
from sklearn.neural_network import MLPClassifier
from model.base import BaseModel


class NeuralNetworkModel(BaseModel):
    def __init__(self, model_name: str = "NeuralNetwork", hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)
        self.model_name = model_name

    def train(self, data):
        self.model.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def data_transform(self) -> None:
        ...
