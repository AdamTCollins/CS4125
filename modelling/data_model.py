import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Set a seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)


class Data:
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        """
        Initialize the Data object with features and a dataframe.

        Args:
            X (np.ndarray): Feature matrix.
            df (pd.DataFrame): Original dataframe containing labels and additional information.
        """
        self.X = X
        self.df = df

        # Perform a train-test split on the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        self.train_df = None
        self.test_df = None

    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: Training and testing data splits (X_train, X_test, y_train, y_test).
        """
        print("Data Model | Splitting data into train and test sets...")
        y = self.df["y2"]  # Assuming "y2" is the target label
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=test_size, random_state=random_state)
        print(f"Data Model | Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples.")
        return X_train, X_test, y_train, y_test

    def get_type(self):
        """
        Get the target labels (y) from the dataframe.

        Returns:
            pd.Series: Target labels (y).
        """
        return self.df["y2"]  # Assuming "y2" is the target column

    def get_X_train(self):
        """
        Get the training feature set.

        Returns:
            np.ndarray: Training features.
        """
        return self.X_train

    def get_X_test(self):
        """
        Get the testing feature set.

        Returns:
            np.ndarray: Testing features.
        """
        return self.X_test

    def get_type_y_train(self):
        """
        Get the training labels.

        Returns:
            np.ndarray: Training labels.
        """
        return self.y_train

    def get_type_y_test(self):
        """
        Get the testing labels.

        Returns:
            np.ndarray: Testing labels.
        """
        return self.y_test

    def get_train_df(self):
        """
        Get the dataframe for training samples.

        Returns:
            pd.DataFrame: Dataframe of training samples.
        """
        if self.train_df is None:
            self.train_df = self.df.iloc[self.X_train.index]
        return self.train_df

    def get_type_test_df(self):
        """
        Get the dataframe for testing samples.

        Returns:
            pd.DataFrame: Dataframe of testing samples.
        """
        if self.test_df is None:
            self.test_df = self.df.iloc[self.X_test.index]
        return self.test_df

    def get_embeddings(self):
        """
        Get the feature matrix (embeddings).

        Returns:
            np.ndarray: Feature matrix.
        """
        return self.X