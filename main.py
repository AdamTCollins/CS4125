from classifier.classifier_facade import ClassifierFacade
from classifier.classifier_factory import ClassifierFactory

if __name__ == '__main__':
    facade = ClassifierFacade()

    file_path="datasets/AppGallery.csv"
    df = facade.load_data(file_path)

    df = facade.preprocess_data(df)
    df[facade.Config.INTERACTION_CONTENT] = df[facade.Config.INTERACTION_CONTENT].astype('U')
    df[facade.Config.TICKET_SUMMARY] = df[facade.Config.TICKET_SUMMARY].astype('U')

    X, group_df = facade.get_embeddings(df)
    data = facade.get_data_object(X, df)

    print("Using facade to create a RandomForest model...")
    facade.train_and_evaluate(data, strategy_name="RandomForest")
    print("Using facade to create a SVM model...")
    facade.train_and_evaluate(data, strategy_name="SVM")
    print("Using facade to create a NeuralNetwork model...")
    facade.train_and_evaluate(data, strategy_name="NeuralNetwork")

    model_name = "neural_network"
    facade.perform_modelling(data, df, model_name)

