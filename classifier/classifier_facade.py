import random
import numpy as np

from model.strategies.neural_network_strategy import NeuralNetworkStrategy
from model.strategies.random_forest_strategy import RandomForestStrategy
from model.strategies.svm_strategy import SVMStrategy
from model.model_context import ModelContext
from observers.observer import Publisher
from observers.subscriber import ConsoleLogger
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
        self.publisher = Publisher(events=["data_loaded", "preprocessing", "embedding_generation", "training",
                                           "data_object_creation", "classification", "evaluation", "modelling"])

        # Create the subscriber to the publisher(Observer)
        # This subscriber will simply output events to the console.
        self.logger = ConsoleLogger(name="AppLogger")
        # Register the logger as a subscriber for all events  (Check Publisher class if you are confused)
        for event in self.publisher.subscribers:
            self.publisher.register(event=event, who=self.logger)

        self.random.seed(seed)
        self.np.random.seed(seed)

    def load_data(self, file_path):
        """
        Load the data from a filepath

        :param file_path:
        :return pandas Dataframe:
        """
        df = self.get_input_data(file_path)
        # Observer: Notify subscribers that data has been loaded.
        self.publisher.dispatch(event="data_loaded", message="Data successfully loaded from file.")
        return df

    def preprocess_data(self, df):
        """
        Preprocess the dataframe using the preprocess_data function.

        :param df: input dataframe to preprocess.
        :return: processed dataframe.
        """
        # Observer: Send out the message for Data Preprocessing
        self.publisher.dispatch(event="preprocessing", message="Data preprocessing completed.")
        return self.preprocess_data(df)

    def choose_strategy(self, strategy_name, **kwargs):
        """
        Choose the strategy to be used for modelling
        """
        if strategy_name == "RandomForest":
            return RandomForestStrategy(**kwargs)
        elif strategy_name == "SVM":
            return SVMStrategy()
        elif strategy_name == "NeuralNetwork":
            return NeuralNetworkStrategy()
        else:
            raise ValueError(f"Strategy not found: {strategy_name}")

    def get_embeddings(self, df):
        # Observer: Notify subscribers of the initialisation of embedding generation.
        self.publisher.dispatch(event="embedding_generation", message="Starting TF-IDF embedding generation...")
        X = self.get_tfidf_embd(df)
        # Observer: Send out the message for of embedding generation completion.
        self.publisher.dispatch(event="embedding_generation",
                                message=f"TF-IDF embeddings generated with shape: {X.shape}.")
        return X, df

    def get_data_object(self, X, df):
        # Observer: Notify subscribers about Data object creation
        self.publisher.dispatch(event="data_object_creation", message="Data object successfully created.")
        return self.Data(X, df)

    def train_and_evaluate(self, data, df, strategy_name, **kwargs):
        """
        Train and evaluate the selected model using the Strategy Pattern.

        Args:
            df: DataFrame used for training and evaluation.
            strategy_name: Name of the strategy/model.
            data: The processed Data object containing train/test splits.
            kwargs: Additional arguments for model initialization.
        """
        print(f"Modelling | Initializing the {strategy_name} model...")

        kwargs.update({
            "data": data,
            "embeddings": data.X_train,
            "df": df,
            "classifier_type": strategy_name,
            "y": data.y_train,
        })

        strategy = ClassifierFactory.get_classifier(**kwargs)
        #strategy = self.choose_strategy(strategy_name)
        # TODO ~ JOHNNY: ADD OBSERVER NOTIFS here
        context = ModelContext(strategy)

        # Training the model
        print(f"Modelling | Training the {strategy_name} model...")
        context.train(data)

        # Making predictions
        print(f"Modelling | Making predictions with the {strategy_name} model...")
        predictions = context.predict(data.X_test)

        # Evaluating the model
        print(f"Modelling | Evaluating the {strategy_name} model...")
        self.evaluate_model(predictions, data.y_test)


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
            data: The processed Data object containing train/test splits.
            model_name (str): The name of the model to use.
            kwargs: Additional arguments for model initialization.
        """
        # Observer: Notify subscribers about the performing modelling
        self.publisher.dispatch(event="modelling", message=f"Performing modelling with {model_name}.")

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
        model.train(data)

        # making predictions
        predictions = model.predict(data.X_test)

        # evaluating the model
        print(f"Modelling | Evaluating the {model_name} model")
        self.evaluate_model(predictions, data.y_test, export_path=export_path, export_format=export_format)