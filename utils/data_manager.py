import pandas as pd

class DataManager:
    def __init__(self, translations_csv="translations.csv"):
        self.translations_csv = translations_csv

    def save_translations(self, translations):
        """
        Save translations to a CSV file. Translations should be a dictionary.
        """
        translations_df = pd.DataFrame(list(translations.items()), columns=["original", "translated"])
        translations_df.to_csv(self.translations_csv, index=False)
        print(f"Translations saved to {self.translations_csv}")

    def load_translations(self):
        """
        Load translations from a CSV file. Returns a dictionary.
        """
        try:
            translations_df = pd.read_csv(self.translations_csv)
            translations = dict(zip(translations_df["original"], translations_df["translated"]))
            print(f"Translations loaded from {self.translations_csv}")
            return translations
        except FileNotFoundError:
            print(f"No translations file found at {self.translations_csv}")
            return {}
