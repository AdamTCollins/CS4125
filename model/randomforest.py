# Random Forest Model

# Imports
import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random


num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)


class RandomForestModel(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray,
                 n_estimators: int = 1000) -> None:
        super(RandomForestModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl.fit(data.X_train, data.y_train)


    # Generating predictions for the test data.
    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions
        return predictions  # return predictions

    # Printing results.
    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))


    def data_transform(self) -> None:
        ...

