import random
import numpy as np

from model.strategies.neural_network_strategy import NeuralNetworkStrategy
from model.strategies.random_forest_strategy import RandomForestStrategy
from model.strategies.svm_strategy import SVMStrategy
from model.model_context import ModelContext
from utils.preprocess import get_input_data, de_duplication, noise_remover, translate_to_en, preprocess_data
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling import evaluate_model
from Config import Config
from classifier.classifier_factory import ClassifierFactory
from export.export_factory import ExportFactory



class ClassifierFacade:
    def __init__(self, seed=0):
        self.random = random
        self.np = np
        self.get_input_data = get_input_data
        self.preprocess_data = preprocess_data
        self.de_duplication = de_duplication
        self.noise_remover = noise_remover
        self.translate_to_en = translate_to_en
        self.get_tfidf_embd = get_tfidf_embd
        self.Data = Data
        self.evaluate_model = evaluate_model
        self.Config = Config

        self.random.seed(seed)
        self.np.random.seed(seed)

    def load_data(self, file_path):
        """
        Load the data from a filepath

        :param file_path:
        :return pandas Dataframe:
        """
        df = self.get_input_data(file_path)
        return df

    def preprocess_data(self, df):
        """
        Preprocess the dataframe using the preprocess_data function.

        :param df: input dataframe to preprocess.
        :return: processed dataframe.
        """
        return self.preprocess_data(df)

    def choose_strategy(self, strategy_name):
        """
        Choose the strategy to be used for modelling
        """
        if strategy_name == "RandomForest":
            return RandomForestStrategy()
        elif strategy_name == "SVM":
            return SVMStrategy()
        elif strategy_name == "NeuralNetwork":
            return NeuralNetworkStrategy()
        else:
            raise ValueError(f"Strategy not found: {strategy_name}")

    def get_embeddings(self, df):
        X = self.get_tfidf_embd(df)
        return X, df

    def get_data_object(self, X, df):
        return self.Data(X, df)

    def train_and_evaluate(self, data, strategy_name):
        """
        Applies strategy and trains/evaluates the model.
        """
        strategy = self.choose_strategy(strategy_name)

        context = ModelContext(strategy)
        context.train(data)
        context.evaluate(data)


    def export_results(self, data, file_path, format_type):
        """
        Export results using the specified format.

        Args:
            data (dict): The results to export.
            file_path (str): Path for the exported file.
            format_type (str): Format type (e.g., "csv").
        """
        exporter = ExportFactory.get_exporter(format_type)
        exporter.export(data, file_path)


    def perform_modelling(self, data, df, model_name, export_path=None, export_format="csv",**kwargs):
        """
        Train and evaluate the selected model using the ModelFactory.

        Args:
            :param data: The processed Data object containing train/test splits.
            :param  (str): The name of the model to use.
            :param kwargs: Additional arguments for model initialization.
            :param export_format: Filetype (csv or json)
            :param export_path: File export path (match export format param)
        """
        print(f"Modelling | Initializing the {model_name} model using ModelFactory...")

        # passing data required for the model
        kwargs.update({
            "data": data,
            "embeddings": data.X_train,
            "df": df,
            "classifier_type": model_name,
            "y": data.y_train,
        })

        # getting the model instance from factory
        model = ClassifierFactory.get_classifier(**kwargs)

        # training
        print(f"Modelling | Training the {model_name} model")
        model.train(data)

        # making predictions
        print(f"Modelling | Making predictions with the {model_name} model")
        predictions = model.predict(data.X_test)

        # evaluating the model
        print(f"Modelling | Evaluating the {model_name} model")
        self.evaluate_model(predictions, data.y_test, export_path=export_path, export_format=export_format)