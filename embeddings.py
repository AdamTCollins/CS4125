# Embedding File

# Imports
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embd(df, column_name="Interaction content"):
    """
    Converting text data into TF-IDF embeddings.

    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Column to extract embeddings from.

    Returns:
        np.ndarray: TF-IDF feature matrix.
    """

    print(f"Embeddings | Generating TF-IDF embeddings for column: {column_name}...")
    print(f"Debug | Column names for given dataframe:  {list(df.columns.values)} ")
    vectorizer = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    embeddings = vectorizer.fit_transform(df[column_name])
    print(f"Embeddings | TF-IDF embeddings generated with shape: {embeddings.shape}.")
    return embeddings