from abc import ABC, abstractmethod

"""
Abstract CNN class for Strategy pattern
"""


class CNNStrategy(ABC):
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass
