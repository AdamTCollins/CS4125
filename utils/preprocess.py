# Data Preprocessing File.

# Imports
import pandas as pd
from utils.translator import Translator
import re


# Configuration class for column names.
class Config:
    TICKET_SUMMARY = "Ticket Summary"
    INTERACTION_CONTENT = "Interaction content"
    TYPE_1 = "Type 1"
    TYPE_2 = "Type 2"
    TYPE_3 = "Type 3"
    TYPE_4 = "Type 4"


# Loading the input data from the CSV file.
def get_input_data(file_path):
    print(f"Preprocess | Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Preprocess | Loaded {len(df)} rows from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error | File not found: {file_path}")
        raise


# CLeaning a column in the DataFrame by removing unnecessary characters, symbols & extra spaces.
def clean_text_column(column):
    return (
        column.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )


# Removing duplicated rows from the DataFrame.
def de_duplication(df):
    print("Preprocess | Removing duplicate rows...")
    df = df.drop_duplicates()
    print(f"Preprocess | Deduplicated dataframe has {len(df)} rows.")
    return df


# Removing noise from the DatFrame.
def noise_remover(df):
    print("Preprocess | Removing noise...")
    return df


# Translating all text to English.
def translate_to_en(column):
    print("Preprocess | Translating text to English...")
    translator = Translator()
    return column.apply(translator.translate)


# Preprocessing the loaded DataFrame.
def preprocess_data(df):
    print("Preprocess | Starting data preprocessing")

    # Cleaning the text columns.
    print("Preprocess | Cleaning text fields...")
    df[Config.INTERACTION_CONTENT] = clean_text_column(df[Config.INTERACTION_CONTENT])
    df[Config.TICKET_SUMMARY] = clean_text_column(df[Config.TICKET_SUMMARY])

    # Translating text to English.
    print("Preprocess | Translating ticket summaries...")
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY])

    # Deduplicating and removing noise.
    df = de_duplication(df)
    df = noise_remover(df)

    # Renaming columns for consistency.
    print("Preprocess | Renaming columns for consistency...")
    df["Interaction content"] = df[Config.INTERACTION_CONTENT]
    df["y1"] = df[Config.TYPE_1]
    df["y2"] = df[Config.TYPE_2]
    df["y3"] = df[Config.TYPE_3]
    df["y4"] = df[Config.TYPE_4]

    # Filtering rows with valid Type 2 labels.
    print("Preprocess | Filtering rows with valid level 2 labels...")
    df = df.loc[df["y2"].fillna("").str.strip() != ""]
    print(f"Preprocess | Filtered down to {len(df)} rows.")

    return df
