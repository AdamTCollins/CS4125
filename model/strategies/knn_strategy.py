from sklearn.neighbors import KNeighborsClassifier
from model.cnn_strategy import CNNStrategy
import pandas as pd

class KNNStrategy(CNNStrategy):
    def __init__(self, model_name="KNN", n_neighbors=5, weights="uniform"):
        """
        Initialize the KNN strategy.

        Args:
            model_name (str): Name of the model.
            n_neighbors (int): Number of neighbors to use.
            weights (str): Weight function used in prediction (uniform or distance)
        """
        self.model_name = model_name
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def train(self, data):
        """
        Train the KNN model.

        Args:
            data: The data object containing X_train and y_train.
        """
        print(f"Training {self.model_name}...")
        self.model.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        """
        Predict using the trained KNN model.

        Args:
            X_test: Test features.

        Returns:
            Predictions for the test data.
        """
        print(f"Predicting with {self.model_name}...")
        return self.model.predict(X_test)

    def evaluate(self, data):
        """
        Evaluate the KNN model.

        Args:
            data: The data object containing X_test and y_test.
        """
        print(f"Evaluating {self.model_name}...")
        accuracy = self.model.score(data.X_test, data.y_test)
        print(f"Test Accuracy: {accuracy}")
