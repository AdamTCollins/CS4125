import unittest
from classifier.classifier_factory import ClassifierFactory
from classifier.classifier_facade import ClassifierFacade
from main import generate_export_path
from observers.observer import Publisher
from observers.subscriber import ConsoleLogger
from model.model_context import ModelContext
from model.strategies.random_forest_strategy import RandomForestStrategy
from model.strategies.svm_strategy import SVMStrategy
from model.strategies.neural_network_strategy import NeuralNetworkStrategy
from utils.translator import Translator
from modelling.data_model import Data
import numpy as np
import pandas as pd

# Mock data generator
def generate_mock_data():
    num_samples = 100
    num_features = 20
    X = np.random.rand(num_samples, num_features)
    y = np.random.choice([0, 1, 2], size=num_samples)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(num_features)])
    df["y2"] = y
    return Data(X, df)

# Factory Pattern Test
class TestFactoryPattern(unittest.TestCase):
    def setUp(self):
        self.data = generate_mock_data()
        self.kwargs = {"embeddings": self.data.X_train, "y": self.data.y_train}

    def test_random_forest_strategy(self):
        rf_model = ClassifierFactory.get_classifier(self.data, df=None, classifier_type="random_forest", **self.kwargs)
        self.assertEqual(rf_model.model_name, "RandomForest")

    def test_svm_strategy(self):
        svm_model = ClassifierFactory.get_classifier(self.data, df=None, classifier_type="svm", kernel="linear", C=1.0)
        self.assertEqual(svm_model.model_name, "SVM")

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            ClassifierFactory.get_classifier(self.data, df=None, classifier_type="invalid_type")

# Facade Pattern Test
class TestFacadePattern(unittest.TestCase):
    def setUp(self):
        self.facade = ClassifierFacade()
        self.file_path = "datasets/mock_data.csv"  # Replace with actual test file

    def test_load_data(self):
        df = self.facade.load_data(self.file_path)
        self.assertIsNotNone(df)

    def test_preprocess_data(self):
        df = self.facade.load_data(self.file_path)
        processed_df = self.facade.preprocess_data(df)
        self.assertIsNotNone(processed_df)

    def test_model_training(self):
        export_path = generate_export_path(model_name="svm", export_format="csv")
        df = self.facade.load_data(self.file_path)
        processed_df = self.facade.preprocess_data(df)
        data_object = self.facade.get_data_object(self.facade.get_embeddings(processed_df)[0], processed_df)
        self.facade.train_and_evaluate(data=data_object, df=processed_df, strategy_name="svm", export_format="csv",export_path = generate_export_path(model_name="svm", export_format="csv"))

# Observer Pattern Test
class TestObserverPattern(unittest.TestCase):
    def test_observer_pattern(self):
        publisher = Publisher(events=["data_loaded", "training_started"])
        logger = ConsoleLogger(name="TestLogger")
        publisher.register(event="data_loaded", who=logger)
        publisher.dispatch(event="data_loaded", message="Dataset loaded")
        publisher.unregister(event="data_loaded", who=logger)

# Strategy Pattern Test
class TestStrategyPattern(unittest.TestCase):
    def setUp(self):
        self.mock_data = generate_mock_data()

    def test_svm_strategy(self):
        svm_strategy = SVMStrategy(model_name="SVM")
        context = ModelContext(strategy=svm_strategy)
        context.train(self.mock_data)
        predictions = context.predict(self.mock_data.X_test)
        self.assertIsNotNone(predictions)

    def test_random_forest_strategy(self):
        rf_strategy = RandomForestStrategy(model_name="RandomForest", embeddings=self.mock_data.X_train, y=self.mock_data.y_train)
        context = ModelContext(strategy=rf_strategy)
        context.train(self.mock_data)
        predictions = context.predict(self.mock_data.X_test)
        self.assertIsNotNone(predictions)

# Singleton Pattern Test
class TestSingletonPattern(unittest.TestCase):
    def test_singleton_pattern(self):
        translator1 = Translator()
        translator2 = Translator()
        self.assertIs(translator1, translator2)
        result = translator1.translate("Hola")
        self.assertEqual(result, "Hello")  # Assuming translation works as expected

# Run all tests
if __name__ == "__main__":
    unittest.main()
