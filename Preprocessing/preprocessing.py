import re
import spacy # type: ignore
from autocorrect import Speller #type: ignore
from wordfreq import zipf_frequency # type: ignore

# Load English spaCy language model
nlp = spacy.load("en_core_web_sm")


class Preprocessor:
    """
    Lightweight text preprocessing class for movie reviews.
    The goal is to improve input quality while preserving structure for Transformer-based models (e.g., KeyBERT).
    It performs:
        1. Typo correction with repetition handling and autocorrect fallback
        2. Spacing normalization around punctuation
        3. Filtering of non-informative or nonsensical reviews
        4. Lemmatization with spacing and casing preserved
    """

    def __init__(self):
        # Initialize the autocorrect spell checker
        self.spell = Speller(lang="en")

    def correct_typos(self, text: str):
        """
        Fixes typos while preserving punctuation, numbers, and spacing.
        Strategy:
        - Uses spaCy to tokenize (preserving whitespace)
        - Leaves non-alphabetic tokens (numbers, punctuation) unchanged
        - Preserves proper nouns (capitalized tokens not at sentence start)
        - Reduces repeated letters (e.g., 'baaad' → 'bad') in a controlled way
        - Falls back to autocorrect when needed
        """

        # Clean out unknown/non-printable characters
        text = re.sub(r'[^\x20-\x7E]', ' ', text)

        # Function to check if a word is valid based on its frequency
        def is_valid(word):
            # A word is valid if its frequency is above a threshold
            return zipf_frequency(word, 'en') > 3.5

        doc = nlp(text)
        corrected = []
        prev_token = ""

        for i, token in enumerate(doc):
            token_text = token.text
            space = token.whitespace_

            # Leave punctuation, numbers, and symbols untouched
            if not token.is_alpha:
                corrected.append(token_text + space)
                prev_token = token_text
                continue

            # Preserve capitalized words (likely named entities)
            if token_text[0].isupper() and i > 0 and prev_token not in [".", "!", "?"]:
                corrected.append(token_text + space)
                prev_token = token_text
                continue

            lower_token = token_text.lower()
            if is_valid(lower_token):
                # Keep valid words unchanged
                corrected.append(token_text + space)
                prev_token = token_text
                continue

            # Try reducing 3+ repeated letters to 2
            reduced_2 = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', token_text)
            if is_valid(reduced_2.lower()):
                corrected.append(reduced_2 + space)
                prev_token = token_text
                continue

            # Try reducing 2 repeated letters to 1
            reduced_1 = re.sub(r'([a-zA-Z])\1', r'\1', reduced_2)
            if is_valid(reduced_1.lower()):
                corrected.append(reduced_1 + space)
            else:
                # Fallback: use autocorrect
                corrected.append(self.spell(token_text) + space)

            prev_token = token_text

        return "".join(corrected).strip()

    def normalize_spacing(self, text: str):
        """
        Ensures a space follows punctuation if it is directly followed by a word character.
        Example: 'Wow!!!Great' → 'Wow!!! Great'
        It avoids affecting things like '$200,000.00'
        """
        return re.sub(r'([.,!?;:])(?=[a-zA-Z_])', r'\1 ', text)

    def is_nonsense(self, text: str):
        """
        Flags a review as nonsense if:
        - It is shorter than 10 characters
        - Less than 30% of characters are alphabetic (to filter out gibberish or symbols)
        """
        text = text.strip()
        if len(text) < 10:
            return True
        alpha_ratio = sum(c.isalpha() for c in text) / (len(text) + 1e-5)
        return alpha_ratio < 0.3

    def lemmatize_text(self, text: str):
        """
        Lemmatizes each token while preserving:
        - Original spacing using spaCy's whitespace handling
        - Proper casing (e.g., 'NASA' remains 'NASA', 'Harry' → 'Harry')
        This ensures readability and compatibility with downstream NLP models.
        """
        doc = nlp(text)
        lemmatized = []

        for token in doc:
            if token.is_space:
                continue

            lemma = token.lemma_

            # Preserve casing
            if token.text.istitle():
                lemma = lemma.capitalize()
            elif token.text.isupper():
                lemma = lemma.upper()

            # List of contractions to split if used as AUX (verb)
            contractions = {
                "'s": "be",
                "'re": "be",
                "'ve": "have",
                "'ll": "will",
                "'d": "would",
                "'m": "be"
            }

            # Handle contractions as separate tokens
            if token.text in contractions and token.pos_ == "AUX":
                lemmatized.append(token.whitespace_ + contractions[token.text] + token.whitespace_)
            else:
                lemmatized.append(lemma + token.whitespace_)

        return "".join(lemmatized).strip()


    def preprocess_review(self, text: str):
        """
        Full preprocessing pipeline for a single review.
        Steps:
        1. Typo correction (repetition + spellcheck)
        2. Normalize spacing around punctuation
        3. Filter out nonsense (short or unintelligible reviews)
        4. Lemmatize while preserving spacing and casing
        Returns:
            - Cleaned string or None if the review is considered invalid
        """
        if not isinstance(text, str):
            return None

        text = self.correct_typos(text)          # Fix typos and repeated letters
        text = self.normalize_spacing(text)      # Fix punctuation spacing

        if self.is_nonsense(text):
            return None                          # Filter meaningless input

        text = self.lemmatize_text(text)         # Final lemmatization
        return text
