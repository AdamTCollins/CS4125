# Imports
from model.strategies.neural_network_strategy import NeuralNetworkStrategy
from model.strategies.random_forest_strategy import RandomForestStrategy
from model.strategies.svm_strategy import SVMStrategy
from model.strategies.knn_strategy import KNNStrategy
from model.strategies.xgboost_strategy import XGBoostStrategy

class ClassifierFactory:
    @staticmethod
    def get_classifier(data, df, classifier_type, **kwargs):
        """
        Dynamically create a strategy instance.

        Args:
            data: The processed Data object.
            df: DataFrame containing training/testing data.
            classifier_type (str): Type of classifier (e.g., 'random_forest', 'svm', 'neural_network').
            kwargs: Additional parameters for the strategy.

        Returns:
            An instance of the specified strategy.

        Raises:
            ValueError: If the classifier_type is invalid.
        """
        if classifier_type == "random_forest":
            model_name = kwargs.get("model_name", "RandomForest")
            n_estimators = kwargs.get("n_estimators", 1000)
            return RandomForestStrategy(
                model_name=model_name,
                embeddings=kwargs["embeddings"],
                y = kwargs["y"],
                n_estimators=n_estimators
            )
        elif classifier_type == "svm":
            model_name = kwargs.get("model_name", "SVM")
            kernel = kwargs.get("kernel", "linear")
            C = kwargs.get("C", 1.0)
            return SVMStrategy(
                model_name=model_name,
                kernel=kernel,
                C=C
            )
        elif classifier_type == "neural_network":
            model_name = kwargs.get("model_name", "NeuralNetwork")
            hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100,))
            activation = kwargs.get("activation", "relu")
            solver = kwargs.get("solver", "adam")
            return NeuralNetworkStrategy(
                model_name=model_name,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver
            )
        elif classifier_type == "knn":
            model_name = kwargs.get("model_name", "KNN")
            n_neighbors = kwargs.get("n_neighbors", 5)
            weights = kwargs.get("weights", "uniform")
            return KNNStrategy(model_name=model_name, n_neighbors=n_neighbors, weights=weights)

        elif classifier_type == "xgboost":
            model_name = kwargs.get("model_name", "XGBoost")
            max_depth = kwargs.get("max_depth", 6)
            learning_rate = kwargs.get("learning_rate", 0.1)
            n_estimators = kwargs.get("n_estimators", 100)
            return XGBoostStrategy(model_name, max_depth=max_depth, learning_rate=learning_rate,
                                   n_estimators=n_estimators)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
