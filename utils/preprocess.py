import pandas as pd
import re
from utils.translator import Translator  # Ensure this module exists and works correctly


class Config:
    """Configuration class for column names."""
    TICKET_SUMMARY = "Ticket Summary"
    INTERACTION_CONTENT = "Interaction content"
    TYPE_1 = "Type 1"
    TYPE_2 = "Type 2"
    TYPE_3 = "Type 3"
    TYPE_4 = "Type 4"


def get_input_data(file_path):
    """
    Load the input data from a CSV file.

    Args:
        file_path (str): Path to the dataset in CSV format.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    print(f"Preprocess | Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Preprocess | Loaded {len(df)} rows from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error | File not found: {file_path}")
        raise


def clean_text_column(column):
    """
    Clean a column in a dataframe by removing unnecessary characters, symbols, and extra spaces.

    Args:
        column (pd.Series): A single column from a pandas dataframe.

    Returns:
        pd.Series: The cleaned column with all text standardized.
    """
    return (
        column.fillna("")  # Replace NaN/None with empty strings
        .astype(str)  # Convert all values to strings
        .str.replace(r"\s+", " ", regex=True)  # Replace multiple spaces with a single space
        .str.replace(r"[^\w\s]", "", regex=True)  # Remove special characters and punctuation
        .str.strip()  # Remove leading and trailing whitespace
    )


def de_duplication(df):
    """
    Remove duplicate rows from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Deduplicated dataframe.
    """
    print("Preprocess | Removing duplicate rows...")
    df = df.drop_duplicates()
    print(f"Preprocess | Deduplicated dataframe has {len(df)} rows.")
    return df


def noise_remover(df):
    """
    Remove noise from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe with noise removed.
    """
    print("Preprocess | Removing noise...")
    # Add specific noise removal logic if applicable
    return df


def translate_to_en(column):
    """
    Translate text to English.

    Args:
        column (pd.Series): Column containing text to translate.

    Returns:
        pd.Series: Translated column.
    """
    print("Preprocess | Translating text to English...")
    translator = Translator()
    return column.apply(translator.translate)


def preprocess_data(df):
    """
    Preprocess the loaded dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    print("Preprocess | Starting data preprocessing")

    # Clean text columns
    print("Preprocess | Cleaning text fields...")
    df[Config.INTERACTION_CONTENT] = clean_text_column(df[Config.INTERACTION_CONTENT])
    df[Config.TICKET_SUMMARY] = clean_text_column(df[Config.TICKET_SUMMARY])

    # Translate text to English
    print("Preprocess | Translating ticket summaries...")
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY])

    # Deduplicate and remove noise
    df = de_duplication(df)
    df = noise_remover(df)

    # Rename columns for consistency
    print("Preprocess | Renaming columns for consistency...")
    df["x"] = df[Config.INTERACTION_CONTENT]
    df["y1"] = df[Config.TYPE_1]
    df["y2"] = df[Config.TYPE_2]
    df["y3"] = df[Config.TYPE_3]
    df["y4"] = df[Config.TYPE_4]

    # Filter rows with valid Type 2 labels
    print("Preprocess | Filtering rows with valid level 2 labels...")
    df = df.loc[df["y2"].fillna("").str.strip() != ""]
    print(f"Preprocess | Filtered down to {len(df)} rows.")

    return df