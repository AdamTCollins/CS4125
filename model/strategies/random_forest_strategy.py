import modelling.modelling
from model.cnn_strategy import CNNStrategy
import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
from model.randomforest import RandomForestModel, seed

"""
Strategy class for Random Forest
"""
class RandomForestStrategy(CNNStrategy):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray,
                 n_estimators: int = 1000) -> None:
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None

    def train(self, data):
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, data):
        predictions = self.mdl.predict(data)
        self.predictions = predictions
        return pd.Series(predictions)

    def evaluate(self, data):
        modelling.modelling.evaluate_model(self.predictions, data.y_test)