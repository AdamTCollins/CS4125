from export.csv_exporter import CSVExporter

class ExportFactory:
    @staticmethod
    def get_exporter(format_type):
        if format_type == "csv":
            return CSVExporter()
        else:
            raise ValueError(f"Unknown export format: {format_type}")
