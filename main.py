from datetime import datetime
from classifier.classifier_facade import ClassifierFacade


def display_menu(title, options):
    """
    Displays a menu and prompts the user for a choice.

    Args:
        title (str): Title of the menu.
        options (list): List of options to display.

    Returns:
        int: The user's selected option as an integer index (1-based).
    """
    print(f"\n=== {title} ===")
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    while True:
        choice = input("Enter your choice: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice)
        print("Invalid choice. Please try again.")


if __name__ == '__main__':
    facade = ClassifierFacade()

    print("\n=== Welcome to the Email Classification Tool ===\n")

    # Main Menu
    main_choice = display_menu("Choose an option", ["Start a new classification", "Quit"])
    if main_choice == 2:
        print("Goodbye!")
        exit()

    # Step 1: Select a Dataset
    datasets = ["AppGallery.csv", "Purchasing.csv"]  # Add more datasets as needed
    dataset_choice = display_menu("Step 1: Select a Dataset", datasets)
    file_path = f"datasets/{datasets[dataset_choice - 1]}"

    # Step 2: Select a Model
    models = ["Random Forest", "Neural Network", "SVM", "XGBoost", "KNN"]
    model_choice = display_menu("Step 2: Select a Model", models)
    model_name = models[model_choice - 1].replace(" ", "_").lower()

    # Step 3: Select a Result Format
    formats = ["CSV", "JSON"]
    format_choice = display_menu("Step 3: Select a Result Format", formats)
    export_format = formats[format_choice - 1].lower()

    # Summary
    print("\n--- Summary of Your Choices ---")
    print(f"Dataset: {file_path}")
    print(f"Model: {model_name}")
    print(f"Result Format: {export_format}\n")

    # Confirm the user's choices
    confirm = input("Proceed with these settings? (y/n): ").strip().lower()
    if confirm != "y":
        print("Exiting application.\n")
        exit()

    # Processing Request
    print("\nProcessing your request...")
    print(f"Running classification with dataset '{file_path}', model '{model_name}', and format '{export_format}'...\n")

    try:
        # Loading the dataset
        df = facade.load_data(file_path)

        # Preprocessing the data
        df = facade.preprocess_data(df)
        df[facade.Config.INTERACTION_CONTENT] = df[facade.Config.INTERACTION_CONTENT].astype('U')
        df[facade.Config.TICKET_SUMMARY] = df[facade.Config.TICKET_SUMMARY].astype('U')

        # Generating the embeddings
        X, group_df = facade.get_embeddings(df)
        data = facade.get_data_object(X, df)

        # Performing the modelling and exporting the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"output/result_{model_name}_{timestamp}.{export_format}"
        facade.perform_modelling(data, df, model_name, export_format=export_format, export_path=export_path)

        print(f"Classification completed successfully! Results exported to {export_path}.")
        print("Exiting application.\n")
        exit()

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print("Exiting application.\n")
        exit()