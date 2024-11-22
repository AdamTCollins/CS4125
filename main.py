# This is the main file: The controller. All methods will directly or indirectly be called here

# Imports
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import get_input_data, preprocess_data  # Importing preprocessing functions
from embeddings import get_tfidf_embd  # Importing embedding generator
from modelling.modelling import evaluate_model  # Importing evaluation function
from modelling.data_model import Data  # Importing the Data class
from engine.modelengine import modelengine
from classifier.factory import ClassifierFactory  # Importing the Factory for classifiers
from Config import Config  # Importing the Config class
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

    # Instantiate the classifier using the Factory

    engine = modelengine()

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

    if classifier_type:
        print(f"Main | Using classifier_type")
        model = engine.get_model(classifier_type)

    if not model:
        print(f"Main | Initializing and adding {classifier_type} model to the ModelEngine...")
        classifier = ClassifierFactory.get_classifier(
            classifier_type=classifier_type,
            model_name="RandomForest",
            embeddings=data.X_train,
            y=data.y_train,
            **kwargs
        )
    else:
        print("Main | No classifier type provided. Using the active model from ModelEngine.")
        model = engine.get_active_model()

    if not model:
        raise ValueError("No active model is set in the ModelEngine. Please specify a classifier_type.")

    # Add new model to the engine
    engine.add_model(classifier_type, model)

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

    # Log predictions (optional, depending on your requirements)
    print(f"Main | Model {classifier_type or engine.active_model_name} achieved a score of {predictions}.")
    return {"model_name": classifier_type or engine.active_model_name, "model": model, "score": predictions}

def dynamically_switch_models(engine, data, df, available_classifiers, metric_threshold, **kwargs):
    """
        Dynamically switch between models based on evaluation scores or other criteria.

        Args:
            engine (ModelEngine): The model engine instance.
            data (Data): Data object for predictions and evaluation.
            df (pd.DataFrame): Original dataframe.
            available_classifiers (list): List of classifier types to evaluate.
            metric_threshold (float): Minimum score to retain the current active model.
            kwargs: Additional parameters for evaluation.

        Returns:
            str: The name of the newly active model.
        """
    print("DynamicSwitch | Evaluating models to decide active model...")

    best_model_name = None
    best_score = 0

    for classifier_type in available_classifiers:
        print(f"DynamicSwitch | Evaluating classifier: {classifier_type}")

        # Use perform_modelling for each classifier
        result = perform_modelling(data, df, classifier_type, **kwargs)
        score = result["score"]

        if score > best_score:
            best_score = score
            best_model_name = classifier_type
    # Logging predictions.
    print(f"Main | Model predictions:\n{predictions}")

    # If the best model exceeds the threshold, switch to it
    if best_score >= metric_threshold and best_model_name:
        engine.set_active_model(best_model_name)
        print(f"DynamicSwitch | Switching to model: {best_model_name} with score {best_score}.")
    else:
        print("DynamicSwitch | No model met the threshold; retaining the current model.")

    return best_model_name

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
    engine = modelengine()

    available_classifiers = ['random_forest', 'svm', 'neural_network']
    print("Main | Dynamically switching between models...")
    best_model_name = dynamically_switch_models(engine, data, df, available_classifiers, metric_threshold=0.9)

    print("Main | Using the active model for predictions...")
    active_model = engine.get_active_model()
    if active_model:
        predictions = active_model.predict(data.X_test)
        print(f"Main | Final predictions:\n{predictions}")

    # # Step 5: Perform modelling
    # print("Main | Starting the modelling process...")
    # classifier_type = 'random_forest'  # Choose classifier dynamically ('random_forest', 'svm', 'neural_network')
    # perform_modelling(data, df, classifier_type, n_estimators=100)  # Pass additional classifier parameters if needed
    engine = modelengine()

    available_classifiers = ['random_forest', 'svm', 'neural_network']
    print("Main | Dynamically switching between models...")
    best_model_name = dynamically_switch_models(engine, data, df, available_classifiers, metric_threshold=0.9)

    print("Main | Using the active model for predictions...")
    active_model = engine.get_active_model()
    if active_model:
        predictions = active_model.predict(data.X_test)
        print(f"Main | Final predictions:\n{predictions}")

    # # Step 5: Perform modelling
    # print("Main | Starting the modelling process...")
    # classifier_type = 'random_forest'  # Choose classifier dynamically ('random_forest', 'svm', 'neural_network')
    # perform_modelling(data, df, classifier_type, n_estimators=100)  # Pass additional classifier parameters if needed

    print("Main | Email classification process completed.")