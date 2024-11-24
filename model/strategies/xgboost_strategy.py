from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from model.cnn_strategy import CNNStrategy

class XGBoostStrategy(CNNStrategy):
    def __init__(self, model_name="XGBoost", max_depth=6, learning_rate=0.1, n_estimators=100):
        self.model_name = model_name
        self.model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
        self.label_encoder = LabelEncoder()

    def train(self, data):
        print(f"Training {self.model_name}...")
        y_train_encoded = self.label_encoder.fit_transform(data.y_train)
        self.model.fit(data.X_train, y_train_encoded)

    def predict(self, X_test):
        print(f"Predicting with {self.model_name}...")
        predictions = self.model.predict(X_test)
        return self.label_encoder.inverse_transform(predictions)

    def evaluate(self, data):
        print(f"Evaluating {self.model_name}...")
        y_test_encoded = self.label_encoder.transform(data.y_test)
        accuracy = self.model.score(data.X_test, y_test_encoded)
        print(f"Test Accuracy: {accuracy}")
