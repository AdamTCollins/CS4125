# Translator File

# Imports
import stanza
from stanza import DownloadMethod
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from stanza.pipeline.core import DownloadMethod
import pandas as pd

'''
Supressing future warnings, It currently throws a warning for a potential security issue for loading untrusted models.
We are using trusted models from hugging face and stanza, so it's not a concern (*^▽^*)
'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Translator:
    """
    Manages the translation of text to English.
    Uses the Facebook M2M100 model for translation and Stanza for language detection.
    """

    _instance = None
    translations_csv = "translations.csv"  # Path for saving translations
    translations = {}  # In-memory cache for translations

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Translator, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        private method to initialize the translation and language detection model
        """

        print("Translator | Initializing models")

        # loading translation model and tokenizer
        self.model_name = "facebook/m2m100_418M"
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)

        # loading stanza
        self.lang_identifier = stanza.Pipeline(lang="multilingual", processors="langid", download_method=DownloadMethod.REUSE_RESOURCES)
        print("Translator | Models loaded successfully")
        self._load_translations()

    def detect_language(self, text):
        """
        Detects the language of the given text using Stanza.
        """

        if not text.strip():
            return "unknown text given"

        doc = self.lang_identifier(text)
        return doc.lang

    def translate(self, text):

        if text in self.translations:
            return self.translations[text]

        if not isinstance(text, str) or not text.strip():
            return text

        detected_lang = self.detect_language(text)
        if detected_lang == "en":
            return text  # text is already in english

        '''
        This is similar to something we had in the labs. There was an if statement that handled these special cases.
        This is basically mapping unsupported languages to their modern equivelants. 
        '''
        lang_mappings = {
            "fro": "fr",
            "la": "it",
            "nn": "no",
            "kmr": "tr",
            "zh": "zh",
            "zh-hans": "zh"
        }
        detected_lang = lang_mappings.get(detected_lang, detected_lang)

        self.tokenizer.src_lang = detected_lang
        encoded_input = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_input, forced_bos_token_id=self.tokenizer.get_lang_id("en")
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        self.translations[text] = translated_text
        self._save_translations()
        return translated_text

    def _load_translations(self):
        """Load existing translations from a CSV file into memory."""
        try:
            translations_df = pd.read_csv(self.translations_csv)
            self.translations = dict(zip(translations_df["original"], translations_df["translated"]))
            print(f"Loaded {len(self.translations)} translations from {self.translations_csv}")
        except FileNotFoundError:
            print(f"No translations file found at {self.translations_csv}. Starting fresh.")
            self.translations = {}

    def _save_translations(self):
        """Save current translations to a CSV file."""
        translations_df = pd.DataFrame(list(self.translations.items()), columns=["original", "translated"])
        translations_df.to_csv(self.translations_csv, index=False)
        print(f"Saved {len(self.translations)} translations to {self.translations_csv}")
