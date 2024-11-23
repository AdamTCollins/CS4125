import json
from export.export_interface import ExportInterface

class JSONExporter(ExportInterface):
    def export(self, data, file_path):
        """
        Exports data to a json file with a given filepath.

        Args:
            data (dict): Data to export.
            file_path (str): Path to save the JSON file.
        """
        try:
            super().export(data, file_path)

            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"JSON report exported to {file_path}")
        except Exception as e:
            print(f"Failed to export JSON: {e}")