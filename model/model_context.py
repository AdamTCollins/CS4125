from model.cnn_strategy import CNNStrategy
import pandas as pd
"""
Concrete model class for the model for strategy pattern
"""

class ModelContext:
    def __init__(self, strategy: CNNStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: CNNStrategy):
        self._strategy = strategy

    def train(self, data):
        self._strategy.train(data)

    def predict(self, data: pd.Series):
        return self._strategy.predict(data)

    def evaluate(self, data):
        self._strategy.evaluate(data)