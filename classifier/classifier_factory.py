# Factory Pattern
# Used to dynamically create instances of classifiers.

# Imports
from model.randomforest import RandomForestModel
from model.svm import SVMModel
from model.neural_network import NeuralNetworkModel


# Factory Class that creates a classifier instance based on the type.
class ClassifierFactory:

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
            model_name = kwargs.get("model_name", "SVM")
            kernel = kwargs.get("kernel", "linear")
            C = kwargs.get("C", 1.0)
            return SVMModel(model_name, kernel=kernel, C=C)
        elif classifier_type == "neural_network":
            model_name = kwargs.get("model_name", "NeuralNetwork")
            hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100,))
            activation = kwargs.get("activation", "relu")
            solver = kwargs.get("solver", "adam")
            return NeuralNetworkModel(
                model_name=model_name,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
