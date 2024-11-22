from classifier.classifier_facade import ClassifierFacade

if __name__ == '__main__':
    facade = ClassifierFacade()

    file_path="datasets/AppGallery.csv"
    df = facade.load_data(file_path)

    df = facade.preprocess_data(df)
    df[facade.Config.INTERACTION_CONTENT] = df[facade.Config.INTERACTION_CONTENT].astype('U')
    df[facade.Config.TICKET_SUMMARY] = df[facade.Config.TICKET_SUMMARY].astype('U')

    X, group_df = facade.get_embeddings(df)
    data = facade.get_data_object(X, df)

    # select model to run here
    model_name = "random_forest"
    facade.perform_modelling(data, model_name)