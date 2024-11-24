class modelengine:
    _instance = None

    def __new___(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(modelengine, cls).__new__(cls)
            return cls._instance

    #
    def __init__(self):
        self.models = {}

    def add_model(self, model_name, model_instance):
        self.models[model_name] = model_instance

    def get_model(self, model_name):
        return self.models.get(model_name)

    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]

    def list_models(self):
        return list(self.models.keys())

    def set_active_model(self, model_name):
        """Set the active model."""
        if model_name in self.models:
            self.active_model_name = model_name
            print(f"ModelEngine | Active model set to: {model_name}")
        else:
            print(f"ModelEngine | Model {model_name} does not exist.")

    def get_active_model(self):
        """Get the currently active model."""
        if self.active_model_name:
            return self.models.get(self.active_model_name)
        else:
            print("ModelEngine | No active model is set.")
            return None
