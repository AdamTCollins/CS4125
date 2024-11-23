from datetime import datetime

from classifier.classifier_facade import ClassifierFacade
from classifier.classifier_factory import ClassifierFactory

if __name__ == '__main__':
    facade = ClassifierFacade()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path="datasets/AppGallery.csv"
    df = facade.load_data(file_path)

    df = facade.preprocess_data(df)
    df[facade.Config.INTERACTION_CONTENT] = df[facade.Config.INTERACTION_CONTENT].astype('U')
    df[facade.Config.TICKET_SUMMARY] = df[facade.Config.TICKET_SUMMARY].astype('U')

    X, group_df = facade.get_embeddings(df)
    data = facade.get_data_object(X, df)


    # Select from the following: random_forest, svm, neural_network, knn, xgboost
    model_name = "random_forest"
    # export format
    export_format = "csv"
    # export path
    export_path = f"output/result_{model_name}_{timestamp}.{export_format}"


    facade.perform_modelling(data, df, model_name, export_format=export_format, export_path=export_path)