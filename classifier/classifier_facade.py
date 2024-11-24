import random
import numpy as np

from model.model_context import ModelContext
from observers.observer_setup import setup_observer
from utils.preprocess import get_input_data, de_duplication, noise_remover, translate_to_en, preprocess_data
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling import evaluate_model
from Config import Config
from classifier.classifier_factory import ClassifierFactory
from export.export_factory import ExportFactory

class ClassifierFacade:
    def __init__(self, seed=0):
        # Set up the publisher with subscribers
        self.publisher = setup_observer()
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

    def get_embeddings(self, df):
        # Observer: Notify subscribers of the initialisation of embedding generation.
        self.publisher.dispatch(event="embedding_generation", message="Starting TF-IDF embedding generation...")
        x = self.get_tfidf_embd(df)
        # Observer: Notify subscribers of embedding generation completion.
        self.publisher.dispatch(event="embedding_generation",
                                message=f"TF-IDF embeddings generated with shape: {x.shape}.")
        return x, df

    def get_data_object(self, x, df):
        # Observer: Notify subscribers about Data object creation
        self.publisher.dispatch(event="data_object_creation", message="Data object successfully created.")
        return self.Data(x, df)

    def train_and_evaluate(self, data, df, strategy_name, export_path, export_format, **kwargs):
        """
        Train and evaluate the selected model using the Strategy Pattern.

        Args:
            export_format: csv or json
            export_path: location for results export
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
        context = ModelContext(strategy)

        # Training the model
        self.publisher.dispatch(event="modelling", message=f"Training the {strategy_name} model...")
        context.train(data)

        # Making predictions
        print(f"Modelling | Making predictions with the {strategy_name} model...")
        predictions = context.predict(data.X_test)

        # Evaluating the model
        print(f"Modelling | Evaluating the {strategy_name} model...")
        self.evaluate_model(predictions, data.y_test, export_path, export_format)

    @staticmethod
    def export_results(data, file_path, format_type):
        """
        Export results using the specified format.

        Args:
            data (dict): The results to export.
            file_path (str): Path for the exported file.
            format_type (str): Format type (e.g., "csv").
        """
        exporter = ExportFactory.get_exporter(format_type)
        exporter.export(data, file_path)