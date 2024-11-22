# This is the main file: The controller. All methods will directly or indirectly be called here
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import get_input_data, preprocess_data  # Importing preprocessing functions
from embeddings import get_tfidf_embd  # Importing embedding generator
from modelling.modelling import evaluate_model  # Importing evaluation function
from modelling.data_model import Data  # Importing the Data class
from classifier.factory import ClassifierFactory  # Importing the Factory for classifiers
from Config import Config  # Importing the Config class
import random
import numpy as np

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """
    Load the input data from a CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    file_path = "datasets/AppGallery.csv"  # Adjust the path to match the dataset location
    df = get_input_data(file_path)
    return df


def preprocess_data_step(df):
    """
    Preprocess the loaded data.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df = preprocess_data(df)
    return df


def get_embeddings(df):
    """
    Generate embeddings from the preprocessed data.

    Args:
        df (pd.DataFrame): Preprocessed dataframe.

    Returns:
        tuple: Features (X) and the original dataframe (df).
    """
    print("Main | Generating TF-IDF embeddings...")
    X = get_tfidf_embd(df)  # Generate TF-IDF embeddings
    return X, df


def get_data_object(X, df):
    """
    Create a Data object for modelling.

    Args:
        X (np.ndarray): Features matrix.
        df (pd.DataFrame): Original dataframe.

    Returns:
        Data: Data object.
    """
    print("Main | Creating Data object...")
    return Data(X, df)


def perform_modelling(data, df, classifier_type, **kwargs):
    """
    Perform modelling using the specified classifier type.

    Args:
        data (Data): Data object containing features and labels.
        df (pd.DataFrame): Original data frame.
        classifier_type (str): Type of classifier (e.g., 'random_forest', 'svm', 'neural_network').
        kwargs: Additional parameters for the classifier.
    """
    # Instantiate the classifier using the Factory
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


def execute_model_workflow(classifier):
    """ Executes the complete workflow for training, predicting, and evaluating a classifier."""
    # Train the classifier
    print("Main | Training the classifier...")
    classifier.train(data.X_train, data.y_train)

    # Make predictions
    print("Main | Making predictions...")
    predictions = classifier.predict(data.X_test)

    # Evaluate the model
    print("Main | Evaluating the model...")
    evaluate_model(predictions, data.y_test)

    # Log predictions (optional, depending on your requirements)
    print(f"Main | Model predictions:\n{predictions}")


# Code execution starts here
if __name__ == '__main__':
    print("Main | Starting the email classification process...")

    # Step 1: Load raw data
    print("Main | Loading raw data...")
    df = load_data()

    # Step 2: Preprocess the data
    print("Main | Preprocessing data...")
    df = preprocess_data_step(df)

    # Ensure columns are in the correct format
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Step 3: Generate embeddings
    print("Main | Generating embeddings...")
    X, group_df = get_embeddings(df)

    # Step 4: Create a Data object
    print("Main | Creating a Data object...")
    data = get_data_object(X, df)

    # Step 5: Perform modelling
    print("Main | Starting the modelling process...")
    classifier_type = 'svm'  # Choose classifier dynamically ('random_forest', 'svm', 'neural_network')
    perform_modelling(data, df, classifier_type, n_estimators=100)  # Pass additional classifier parameters if needed

    print("Main | Email classification process completed.")
