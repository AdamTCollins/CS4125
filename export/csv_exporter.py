import csv
from export.export_interface import ExportInterface

class CSVExporter(ExportInterface):
    def export(self, data, file_path):
        """
        Exports data to a csv file with a given filepath.

        Args:
            data (dict): data to be exported
            file_path (str): path to save the csv file to

        Raises:
            ValueError: if the data structure is not as expected.
        """

        try:
            super().export(data, file_path)

            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)

                if "metrics" in data:
                    writer.writerow(["Metric", "Value"])
                    for metric, value in data["metrics"].items():
                        writer.writerow([metric, value])

                if "classification_report" in data:
                    writer.writerow([])
                    writer.writerow(["Classification Report"])
                    writer.writerow([data["classification_report"]])

                if "predictions" in data:
                    writer.writerow([])
                    writer.writerow(["Predictions"])
                    for pred in data["predictions"]:
                        writer.writerow([pred])
                else:
                    print("Warning: No predictions found in the data.")

            print(f"CSV report has been successfully exported to {file_path}")

        except FileNotFoundError:
            print(f"Error: The specified file path '{file_path}' does not exist")
