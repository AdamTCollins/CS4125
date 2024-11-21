import pandas as pd
import re
from utils.translator import Translator


def clean_text_column(column):
    """
    cleans a column in a dataframe by removing unnecessary characters, symbols,
    and extra spaces.

    Args:
        column (pd.series): A single column from a pandas dataframe (like a list of text values).

    Returns:
        pd.series: the cleaned column with all text standardized.
    """


    return (
        column.fillna("")  # replacing nan/none with empty strings
        .astype(str)  # turning all values to strings
        .str.replace(r"\s+", " ", regex=True)  # replace multiple spaces with a single space
        .str.replace(r"[^\w\s]", "", regex=True)  # removing special characters and punctuation
        .str.strip()  # removing whitespace
    )

def preprocess_data(file_path):
    """
    Preprocesses data by:
    1. loading the dataset from a csv file
    2. replacing none or nan type values with empty strings
    3. cleaning interaction content and ticket summary columns
    4. translating ticket summary to english if necessary
    5. renaming columns (we should change this later)

    Args:
        file_path (str): path to the dataset in csv format

    Returns:
        dataframe: preprocessed dataframe
    """

    print("Preprocess | Starting data preprocessing")

    # loading dataset
    df = pd.read_csv(file_path)
    print(f"Preprocess | Loaded {len(df)} rows from {file_path}.")

    # cleaning columns
    print("Preprocess | Cleaning text fields")
    df["Interaction content"] = clean_text_column(df["Interaction content"])
    df["Ticket Summary"] = clean_text_column(df["Ticket Summary"])

    # translating ticket summaries
    print("Preprocess | Translating ticket summaries")
    translator = Translator()
    df["Ticket Summary"] = df["Ticket Summary"].apply(translator.translate)

    # renaming columns
    print("Preprocess | Renaming columns for consistency...")
    df["x"] = df["Interaction content"]
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]

    # removing rows where column type 2 has missing or
    print("Preprocess | Filtering rows with valid level 2 labels...")
    df = df.loc[df["y2"].fillna("").str.strip() != ""]
    print(f"Preprocess | Filtered down to {len(df)} rows.")

    return df