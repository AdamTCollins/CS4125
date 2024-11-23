from abc import ABC, abstractmethod

class ExportInterface(ABC):
    @abstractmethod
    def export(self, data, file_path):
        pass