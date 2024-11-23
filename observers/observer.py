class Publisher:
    """
        Publisher class (Observer pattern).
        It manages events and notifies subscribers (observers) when an event is triggered.
    """

    def __init__(self, events):
        # dictionary to store subscribers for each event.
        self.subscribers = {event: dict()
                            for event in events}

    def get_subscribers(self, event):
        # Get all subscribers for a specific event
        return self.subscribers[event]

    def register(self, event, who, callback=None):
        """Register a subscriber for an event"""
        if callback is None:
            # Default to the update method of the subscriber if no callback is given.
            # Check Subscriber.py it simply prints the event and message.
            callback = getattr(who, "update")
        self.get_subscribers(event)[who] = callback

    def unregister(self, event, who):
        """Unregister a subscriber from a specific event."""
        del self.subscribers[event][who]

    def dispatch(self, event, message):
        """
        Notify all subscribers of a specific event by calling their callbacks.
        This will send the event name and message out.
        """
        if event not in self.subscribers:
            raise ValueError(f"Event '{event}' is not registered.")
        # Call each subscriber's callback method with the event and message
        for subscriber, callback in self.subscribers[event].items():
            callback(event, message)
