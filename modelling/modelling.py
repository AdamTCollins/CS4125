from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(predictions, y_test):
    """
    Evaluate the performance of a model.

    Args:
        predictions (np.ndarray): Model predictions.
        y_test (np.ndarray): True labels.

    Prints:
        Classification metrics including accuracy, precision, recall, and F1-score.
    """
    print("Modelling | Evaluating model performance...")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average='weighted'))
    print("Recall:", recall_score(y_test, predictions, average='weighted'))
    print("F1 Score:", f1_score(y_test, predictions, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, predictions))