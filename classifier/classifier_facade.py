import random
import numpy as np
from utils.preprocess import get_input_data, de_duplication, noise_remover, translate_to_en, preprocess_data
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling import evaluate_model
from Config import Config
from model.svm import SVMModel
from model.randomforest import RandomForestModel
from classifier.classifier_factory import ClassifierFactory  # Import your factory class


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

    def get_embeddings(self, df):
        X = self.get_tfidf_embd(df)
        return X, df

    def get_data_object(self, X, df):
        return self.Data(X, df)

    def perform_modelling(self, data, model_name, **kwargs):
        """
        Train and evaluate the selected model using the ModelFactory.

        Args:
            data: The processed Data object containing train/test splits.
            model_name (str): The name of the model to use.
            kwargs: Additional arguments for model initialization.
        """
        print(f"Modelling | Initializing the {model_name} model using ModelFactory...")

        # passing data required for the model
        kwargs.update({
            "embeddings": data.X_train,
            "y": data.y_train,
        })

        # getting the model instance from factory
        model = ClassifierFactory.get_classifier(model_name, **kwargs)

        # training
        print(f"Modelling | Training the {model_name} model")
        model.train(data)

        # making predictions
        print(f"Modelling | Making predictions with the {model_name} model")
        predictions = model.predict(data.X_test)

        # evaluating the model
        print(f"Modelling | Evaluating the {model_name} model")
        self.evaluate_model(predictions, data.y_test)

