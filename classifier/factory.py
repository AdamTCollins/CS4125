# The purpose of our factory pattern is to dynamically create instances of classifiers.

from model.randomforest import RandomForestModel
from model.svm import SVMModel
from model.neural_network import NeuralNetworkModel


class ClassifierFactory:
    """Factory class to create classifier instances based on type."""

    @staticmethod
    def get_classifier(data, df, classifier_type, **kwargs):
        """
        Dynamically create a classifier instance.

        Args:
            classifier_type (str): Type of classifier (e.g., 'random_forest', 'svm', 'neural_network').
            kwargs: Additional parameters for the classifier.

        Returns:
            An instance of the specified classifier.

        Raises:
            ValueError: If the classifier_type is invalid.
        """
        if classifier_type == "random_forest":
            model_name = kwargs.get("model_name", "RandomForest")
            embeddings = kwargs["embeddings"]
            y = kwargs["y"]
            n_estimators = kwargs.get("n_estimators", 1000)
            return RandomForestModel(model_name, embeddings, y, n_estimators=n_estimators)
        elif classifier_type == "svm":
            return SVMModel(**kwargs)
        elif classifier_type == "neural_network":
            return NeuralNetworkModel(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
