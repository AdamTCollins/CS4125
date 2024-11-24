from observers.observer import Publisher
from observers.subscriber import ConsoleLogger


def setup_observer():
    """
    Create and configure the Publisher and its subscribers (loggers).
    Returns:
        Publisher: Configured publisher instance.
    """
    # Initialize Publisher with events
    publisher = Publisher(events=[
        "data_loaded", "preprocessing", "embedding_generation", "training",
        "data_object_creation", "classification", "evaluation", "modelling"
    ])

    # Create loggers (subscribers)
    info_logger = ConsoleLogger(name="General Logger")
    metrics_logger = ConsoleLogger(name="Metrics Logger")

    # Map events to specific loggers
    event_subscriber_map = {
        "data_loaded": info_logger,
        "preprocessing": info_logger,
        "embedding_generation": info_logger,
        "evaluation": metrics_logger,
        "modelling": metrics_logger,
        "training": info_logger,
    }

    # Register subscribers with the publisher
    for event, subscriber in event_subscriber_map.items():
        publisher.register(event=event, who=subscriber)

    return publisher
