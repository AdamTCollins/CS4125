import modelling.modelling
from model.cnn_strategy import CNNStrategy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random

num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
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