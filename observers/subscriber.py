from abc import ABC, abstractmethod


class Subscriber(ABC):
    """ Interface for Subscriber """
    pass

    @abstractmethod
    def update(self, event: str, message: str) -> None:
        pass


class ConsoleLogger(Subscriber):
    """ A subscription to the publisher (Observer pattern) that outputs to the console"""

    def __init__(self, name: str):
        self.name = name

    def update(self, event: str, message: str) -> None:
        print(f"[{self.name}] Event: {event} | Message: {message}")


class MetricsCollector(Subscriber):
    """A subscriber to collect and store metrics."""

    def __init__(self):
        self.metrics = []

    def update(self, event, message):
        self.metrics.append((event, message))
        print(f"[MetricsCollector] Captured event: {event} with message: {message}")
