# Executable main.py file that loads the Command Line Interface.

# Imports
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
        choice = input("Please enter your choice: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice)
        print("Invalid choice. Please try again.")


def preload_dataset(facade):
    """
    Handles dataset selection and preprocessing.

    Args:
        facade (ClassifierFacade): The facade object for dataset handling.

    Returns:
        tuple: Preprocessed dataset (DataFrame) and dataset name (str).
    """
    datasets = ["AppGallery.csv", "Purchasing.csv"]
    dataset_choice = display_menu("Step 1: Select a Dataset to Classify", datasets)
    file_path = f"datasets/{datasets[dataset_choice - 1]}"

    print("\nPreloading and Preprocessing the Dataset...\n")
    try:
        # Loading and preprocessing the dataset.
        df = facade.load_data(file_path)
        df = facade.preprocess_data(df)
        df[facade.Config.INTERACTION_CONTENT] = df[facade.Config.INTERACTION_CONTENT].astype('U')
        df[facade.Config.TICKET_SUMMARY] = df[facade.Config.TICKET_SUMMARY].astype('U')
        print("Dataset Preloaded and Preprocessed Successfully!\n")
        return df, datasets[dataset_choice - 1]
    except Exception as e:
        print(f"Failed to load or preprocess the dataset: {e}")
        exit()


def run_classification(facade, df, dataset_name):
    """
    Handles the classification workflow.

    Args:
        facade (ClassifierFacade): The facade object to run the classification.
        df (DataFrame): Preprocessed dataset.
        dataset_name (str): Name of the selected dataset.
    """
    # Step 2: Selecting a Classification Model.
    models = ["Random Forest", "Neural Network", "SVM", "XGBoost", "KNN"]
    model_choice = display_menu("Step 2: Please select a Classification Model", models)
    model_name = models[model_choice - 1].replace(" ", "_").lower()

    # Step 3: Selecting a Result Format.
    formats = ["CSV", "JSON"]
    format_choice = display_menu("Step 3: Please select your preferred Result Format", formats)
    export_format = formats[format_choice - 1].lower()

    # Summary of Choices.
    print("\n--- Summary of Your Choices ---")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Result Format: {export_format}\n")

    confirm = input("Proceed with these settings? (y/n): ").strip().lower()
    if confirm != "y":
        print("Returning to the main menu...\n")
        return

    # Generating Embeddings.
    print("\nProcessing your request...")
    print(
        f"Running Classification with Dataset '{dataset_name}', Model '{model_name}', and Format '{export_format}'...\n")

    try:
        X, group_df = facade.get_embeddings(df)
        data = facade.get_data_object(X, df)

        # Performing modelling and exporting the results.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"output/result_{model_name}_{timestamp}.{export_format}"
        facade.perform_modelling(data, df, model_name, export_format=export_format, export_path=export_path)

        print(f"Classification completed successfully! Results exported to {export_path}.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    facade = ClassifierFacade()

    print("\n=== Welcome to the Email Classification Tool ===")

    # Preloading the initial dataset.
    df, dataset_name = preload_dataset(facade)

    # Main menu loop.
    while True:
        main_choice = display_menu("Please Choose an Option:", ["Run a Classification", "Change the Dataset", "Quit"])

        if main_choice == 1:
            run_classification(facade, df, dataset_name)
        elif main_choice == 2:
            # Reloading the dataset.
            df, dataset_name = preload_dataset(facade)
        elif main_choice == 3:
            print("Thank you. Goodbye!")
            break