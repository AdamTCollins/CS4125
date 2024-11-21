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

# This file already contain the code for implementing randomforest model
# Carefully observe the methods below and try calling them in modelling.py

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
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        """
        Generates predictions for the test data

        Args:
            X_test pandas series: The test feature matrix.

        Returns:
            np.ndarray: the predictions for the test set.
        """

        predictions = self.mdl.predict(X_test)
        self.predictions = predictions
        return predictions  # return predictions

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))


    def data_transform(self) -> None:
        ...

