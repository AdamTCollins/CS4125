# This is the main file: The controller. All methods will directly or indirectly be called here

# Imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import get_input_data, preprocess_data
from embeddings import get_tfidf_embd
from modelling.modelling import evaluate_model
from modelling.data_model import Data
from classifier.factory import ClassifierFactory
from Config import Config
import random
import numpy as np

# Setting random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

# Loading the input data from the CSV.
def load_data():
    file_path = "datasets/AppGallery.csv"
    df = get_input_data(file_path)
    return df


# Preprocessing the loaded data.
def preprocess_data_step(df):
    df = preprocess_data(df)
    return df


# Generating embeddings from the preprocessed data.
def get_embeddings(df):
    print("Main | Generating TF-IDF embeddings...")
    X = get_tfidf_embd(df)  # Generate TF-IDF embeddings
    return X, df


# Creating a data object for modelling.
def get_data_object(X, df):
    print("Main | Creating Data object...")
    return Data(X, df)


# Performing modelling using the chosen classifier.
def perform_modelling(data, df, classifier_type, **kwargs):

    # Instantiating the classifier using the factory.
    print(f"Main | Using classifier: {classifier_type}")
    if classifier_type == 'random_forest':
        classifier = ClassifierFactory.get_classifier(
            classifier_type=classifier_type,
            model_name="RandomForest",
            embeddings=data.X_train,
            y=data.y_train,
            **kwargs
        )
        execute_model_workflow(classifier)

    elif classifier_type == 'svm':
        classifier = ClassifierFactory.get_classifier(
            classifier_type="svm",
            model_name="SVMClassifier",
            kernel="linear",
            C=1.0
        )
        execute_model_workflow(classifier)

    elif classifier_type == 'neural_network':
        classifier = ClassifierFactory.get_classifier(
            classifier_type="neural_network",
            model_name="NeuralNet",
            hidden_layer_sizes=(50, 50),
            activation="relu",
            solver="adam"
        )
        execute_model_workflow(classifier)
    else:
        print(f"{classifier_type} is not a valid classifier type.")


# Executing the complete workflow for training, predicting and evaluation a classifier.
def execute_model_workflow(classifier):

    # Training the classifier.
    print("Main | Training the classifier...")
    classifier.train(data.X_train, data.y_train)

    # Making predictions.
    print("Main | Making predictions...")
    predictions = classifier.predict(data.X_test)

    # Evaluating the model.
    print("Main | Evaluating the model...")
    evaluate_model(predictions, data.y_test)

    # Logging predictions.
    print(f"Main | Model predictions:\n{predictions}")


# Executing Code.
if __name__ == '__main__':
    # CLI to ask user which model to use.
    print("Choose a classifier type: [random_forest, svm, neural_network]")
    classifier_type = input("Enter classifier type: ").strip()

    if classifier_type not in ['random_forest', 'svm', 'neural_network']:
        print("Invalid classifier type. Please choose from 'random_forest', 'svm', or 'neural_network'.")
        sys.exit(1)

    print(f"Main | Starting the email classification process with {classifier_type} classifier...")

    # Loading the raw data.
    print("Main | Loading raw data...")
    df = load_data()

    # Preprocessing the data.
    print("Main | Preprocessing data...")
    df = preprocess_data_step(df)

    # Ensuring columns are in the correct format.
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Generating the embeddings.
    print("Main | Generating embeddings...")
    X, group_df = get_embeddings(df)

    # Creating a data object.
    print("Main | Creating a Data object...")
    data = get_data_object(X, df)

    # Performing modelling.
    print("Main | Starting the modelling process...")
    perform_modelling(data, df, classifier_type, n_estimators=100)  # Pass additional classifier parameters if needed

    print("Main | Email classification process completed.")