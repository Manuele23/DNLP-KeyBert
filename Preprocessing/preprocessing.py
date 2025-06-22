import re
import spacy # type: ignore
from autocorrect import Speller # type: ignore
from wordfreq import zipf_frequency # type: ignore

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")


class Preprocessor:
    """
    Lightweight text preprocessing class with:
    1. Typo correction (repetition handling + autocorrect)
    2. Punctuation-spacing normalization
    3. Nonsense filtering
    4. Lemmatization using spaCy (preserves original casing and punctuation)
    """

    def __init__(self):
        self.spell = Speller(lang="en")

    def correct_typos(self, text: str):
        """
        Correct typos with refined logic:
        1. Skip correction if word is already valid (high frequency in English).
        2. For words with 3+ repeated letters:
        - Try reducing to 2 repeated characters first.
        - If invalid, reduce to 1 and retry.
        3. If still invalid, use autocorrect as fallback.
        """
        def is_valid(word):
            return zipf_frequency(word, 'en') > 3.5

        corrected_words = []
        for word in text.split():
            original = word

            if is_valid(original.lower()):
                corrected_words.append(original)
                continue

            # Step 1: reduce 3+ repetitions to 2
            reduced_to_2 = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', word)
            if is_valid(reduced_to_2.lower()):
                corrected_words.append(reduced_to_2)
                continue

            # Step 2: further reduce double letters to single
            reduced_to_1 = re.sub(r'([a-zA-Z])\1', r'\1', reduced_to_2)
            if is_valid(reduced_to_1.lower()):
                corrected_words.append(reduced_to_1)
            else:
                # Fallback to autocorrect
                corrected_words.append(self.spell(original))

        return " ".join(corrected_words)



    def normalize_spacing(self, text: str) -> str:
        """
        Add a space after punctuation only if:
        - It's directly followed by an alphanumeric character (no whitespace)
        - It's NOT followed by another punctuation mark (e.g., '!!!' is preserved)
        Examples:
            'Hello.This' → 'Hello. This'
            'Wow!!!Great' → 'Wow!!! Great'
        """
        return re.sub(r'([.,!?;:])(?=[^\s\W\d_])', r'\1 ', text)

    def is_nonsense(self, text: str) -> bool:
        """
        Flag a review as nonsense if:
        - It is too short (<10 characters)
        - It contains mostly non-alphabetic characters (less than 30% alpha)
        """
        text = text.strip()
        if len(text) < 10:
            return True
        alpha_ratio = sum(c.isalpha() for c in text) / (len(text) + 1e-5)
        return alpha_ratio < 0.3

    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize the input text using spaCy, with:
        - Casing preservation
        - No space added before punctuation
        """
        doc = nlp(text)
        tokens = []

        for token in doc:
            # Avoid strange lemmatization for pronouns
            lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text

            # Preserve original casing
            if token.text[0].isupper():
                lemma = lemma.capitalize()
            elif token.text.isupper():
                lemma = lemma.upper()

            # Attach punctuation to previous token
            if token.is_punct and tokens:
                tokens[-1] += lemma
            else:
                tokens.append(lemma)

        return " ".join(tokens)

    def preprocess_review(self, text: str) -> str:
        """
        Full preprocessing pipeline:
        - Correct typos (repetitions + autocorrect)
        - Normalize punctuation spacing
        - Filter out nonsense reviews
        - Lemmatize the final cleaned text
        """
        if not isinstance(text, str):
            return None

        text = self.correct_typos(text)
        text = self.normalize_spacing(text)

        if self.is_nonsense(text):
            return None

        text = self.lemmatize_text(text)
        return text
