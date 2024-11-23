from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from export.export_factory import ExportFactory


def evaluate_model(predictions, y_test, export_path=None, export_format="csv"):
    """
    Evaluate the performance of a model.

    Args:
        predictions (np.ndarray): Model predictions.
        y_test (np.ndarray): True labels.

    Prints:
        Classification metrics including accuracy, precision, recall, and F1-score.
    """

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    class_report = classification_report(y_test, predictions)

    print("Modelling | Evaluating model performance...")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average='weighted'))
    print("Recall:", recall_score(y_test, predictions, average='weighted'))
    print("F1 Score:", f1_score(y_test, predictions, average='weighted'))
    print("\nClassification Report:\n", class_report)

    if export_path:
        metrics = {
            "Accuracy": f"{accuracy:.2%}",
            "Precision": f"{precision:.2%}",
            "Recall": f"{recall:.2%}",
            "F1-Score": f"{f1:.2%}",
        }

        data = {
            "metrics": metrics,
            "predictions": list(predictions),
            "classification_report": class_report
        }

        exporter = ExportFactory.get_exporter(export_format)
        exporter.export(data, export_path)

        print(f"Evaluation results exported to {export_path}")
