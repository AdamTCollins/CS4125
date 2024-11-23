# SVM Model
import numpy as np
# Imports
from sklearn.svm import SVC
from model.base import BaseModel


class SVMModel(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, kernel='linear', C=1.0):
        super().__init__(SVMModel).__init__()
        self.model = SVC(kernel=kernel, C=C)
        self.model_name = model_name


    def train(self, data):
        self.model.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def data_transform(self) -> None:
        ...