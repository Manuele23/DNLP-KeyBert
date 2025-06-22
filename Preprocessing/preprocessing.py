import re
import spacy  # type: ignore
from autocorrect import Speller  # type: ignore
from wordfreq import zipf_frequency  # type: ignore

# Load English spaCy language model
nlp = spacy.load("en_core_web_sm")


class Preprocessor:
    """
    Lightweight text preprocessing class for movie reviews.

    Purpose:
    To improve the quality of noisy, user-generated text before feeding it into
    transformer-based models like KeyBERT. The focus is on cleaning up typos,
    repeated characters, and punctuation spacing—while preserving as much structure
    and semantics as possible.
    """

    def __init__(self):
        # Initialize spell checker
        self.spell = Speller(lang="en")

    def is_valid(self, word):
        """
        Returns True if the word is common enough in English (Zipf frequency > 3.5),
        otherwise returns False. Used to determine if a candidate correction is acceptable.
        """
        return zipf_frequency(word, 'en') > 3.5

    def correct_typos(self, text: str):
        """
        Fixes repeated letters and spelling errors in words while preserving punctuation and spacing.

        Strategy:
        - Tokenize text using spaCy while preserving spaces.
        - Leave non-alphabetic tokens (numbers, punctuation, emojis) unchanged.
        - For words, reduce repeated letters:
            1. First reduce 3+ repeated letters → 2 (e.g., "soooo" → "soo")
            2. Then reduce double letters → single (e.g., "baad" → "bad")
        - Accept the first candidate that passes the frequency check.
        - If no candidate passes, use autocorrect.
        - Normalize all extra whitespace at the end.
        """
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E]', ' ', text)

        # Tokenize the input text
        doc = nlp(text)
        corrected = []

        for token in doc:
            token_text = token.text
            space = token.whitespace_

            # Leave non-alphabetic tokens unchanged
            if not token.is_alpha:
                corrected.append(token_text + space)
                continue

            # Try to reduce repeated characters
            reduced_2 = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', token_text)
            reduced_1 = re.sub(r'([a-zA-Z])\1', r'\1', reduced_2)

            # Use first valid form based on Zipf frequency
            for candidate in [reduced_2, reduced_1]:
                if self.is_valid(candidate.lower()):
                    corrected.append(candidate + space)
                    break
            else:
                # Fallback to autocorrect if no valid candidate found
                corrected.append(self.spell(token_text) + space)

        # Normalize multiple spaces to a single space
        cleaned_text = re.sub(r'\s+', ' ', "".join(corrected))
        return cleaned_text.strip()

    def normalize_spacing(self, text: str):
        """
        Fixes spacing issues around punctuation and parentheses to improve tokenizer behavior.

        Specifically:
        - Ensures a space appears after punctuation if it's immediately followed by a word character
        - Adds a space before '(' if it's not already spaced
        """
        # Add space after punctuation if followed by a word
        text = re.sub(r'([.,!?;:])(?=[a-zA-Z_])', r'\1 ', text)

        # Add space before '(' if needed
        text = re.sub(r'(?<=[^\s])\(', r' (', text)

        return text

    def is_nonsense(self, text: str):
        """
        Determines if a review is too short or meaningless to be useful.

        A review is discarded if:
        - It's shorter than 10 characters
        - Fewer than 30% of characters are alphabetic (i.e., mostly symbols or numbers)
        """
        text = text.strip()
        if len(text) < 10:
            return True
        alpha_ratio = sum(c.isalpha() for c in text) / (len(text) + 1e-5)
        return alpha_ratio < 0.3

    def preprocess_review(self, text: str):
        """
        Full preprocessing pipeline for a single review.

        Steps:
        1. Fix spacing around punctuation
        2. Correct spelling and repeated letters
        3. Filter out reviews that are too short or meaningless

        Returns:
        - The cleaned review string, or None if the review is discarded
        """
        if not isinstance(text, str):
            return None

        text = self.normalize_spacing(text)
        text = self.correct_typos(text)

        if self.is_nonsense(text):
            return None

        return text
