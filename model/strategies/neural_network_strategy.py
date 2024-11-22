from model.cnn_strategy import CNNStrategy

"""
Strategy class for Neural Network
"""
class NeuralNetworkStrategy(CNNStrategy):
    def train(self, data):
        print("train neural network model")

    def evaluate(self, data):
        print("evaluate neural network model")