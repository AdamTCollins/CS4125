import stanza
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class Translator:
    """
    Manages the translation of text to English.
    Uses the Facebook M2M100 model for translation and Stanza for language detection.
    """

    _instance = None

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

    def detect_language(self, text):
        """
        Detects the language of the given text using Stanza.
        """

        if not text.strip():
            return "unknown text given"

        doc = self.lang_identifier(text)
        return doc.lang

    def translate(self, text):
        """
        translates the given text to english if it is not in english
        """

        if not text.strip():
            return text  # return empty text

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
            "kmr": "tr"
        }
        detected_lang = lang_mappings.get(detected_lang, detected_lang)

        self.tokenizer.src_lang = detected_lang
        encoded_input = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_input, forced_bos_token_id=self.tokenizer.get_lang_id("en")
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text