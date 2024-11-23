from model.cnn_strategy import CNNStrategy

"""
Strategy class for SVM
"""
class SVMStrategy(CNNStrategy):
    def train(self, data):
        print("train svm model")

    def evaluate(self, data):
        print("evaluate svm model")