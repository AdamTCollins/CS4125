from model.cnn_strategy import CNNStrategy
"""
Strategy class for Random Forest
"""
class RandomForestStrategy(CNNStrategy):
    def train(self, data):
        print("train random forest model")

    def evaluate(self, data):
        print("evaluate random forest model")