from abc import ABC, abstractmethod
import os

class ExportInterface(ABC):
    @abstractmethod
    def export(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pass