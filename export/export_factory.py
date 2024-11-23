from export.csv_exporter import CSVExporter
from export.json_exporter import JSONExporter

class ExportFactory:
    @staticmethod
    def get_exporter(format_type):
        if format_type == "csv":
            return CSVExporter()
        elif format_type == "json":
            return JSONExporter()
        else:
            raise ValueError(f"Unknown export format: {format_type}")
