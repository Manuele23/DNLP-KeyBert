# You need to install the following packages:
# pip install keybert
# pip install scikit-learn
# pip install numpy
# pip install torch
# pip install sentence-transformers

import sys

import sys
sys.path.append("../KeyBERTSentimentAware")  # Add parent directory to import custom modules

import numpy as np  # Fundamental package for numerical computing in Python
from typing import Tuple  # Used for type hinting tuples in function signatures

import torch  # Core PyTorch library for tensor computations
import torch.nn.functional as F  # Functional interface for activation functions, etc.

from sklearn.feature_extraction.text import CountVectorizer  # Extract text n-gram candidates
from sklearn.metrics.pairwise import cosine_similarity  # Compute cosine similarity between embeddings

# KeyBERT keyword extraction base class
from keybert import KeyBERT as KB  # type: ignore 

# Sentence transformer for generating sentence embeddings
from sentence_transformers import SentenceTransformer   # type: ignore

# Custom sentiment model wrapper (generalized)
from models.SentimentModel import SentimentModel

# KeyBERT extension for sentiment-aware keyword extraction
# This class extends KeyBERT to include sentiment-aware keyword extraction
# by defining a subclass of KeyBERT that modifies the scoring phase to incorporate
# sentiment alignment in a post-processing step. The goal is to boost keywords
# that are both semantically relevant and emotionally aligned with the overall
# sentiment of the input review.
class KeyBERTSentimentAware(KB):
    """
    Extension of KeyBERT to integrate sentiment analysis in keyword extraction.

    This class overrides and extends parts of KeyBERT's pipeline to:
    - Extract a larger candidate pool using CountVectorizer.
    - Calculate sentiment polarity scores for the document and candidates,
      using a pretrained sentiment classification model with continuous outputs.
    - Combine semantic similarity and sentiment alignment scores via a weighting factor alpha.
    - Filter candidate keywords based on this combined score before final ranking.

    Parameters:
    -----------
    model : SentenceTransformer
        Semantic embedding model used by KeyBERT.

    sentiment_model_name : str, optional (default: "nlptown/bert-base-multilingual-uncased-sentiment")
        Identifier of pretrained sentiment model on HuggingFace Hub.

    alpha : float, optional (default: 0.5)
        Weight to balance sentiment alignment vs semantic similarity.
        alpha=1.0 means only sentiment alignment is considered.
        alpha=0.0 means only semantic similarity is considered.

    candidate_pool_size : int, optional (default: 100)
        Maximum number of initial candidate keywords to extract.

    device : str, optional (default: "cpu")
        Device to run embedding and sentiment models on ("cpu" or "cuda").
    """

    def __init__(
        self,
        model,
        sentiment_model_name: str ="cardiffnlp/twitter-roberta-base-sentiment", # or "nlptown/bert-base-multilingual-uncased-sentiment"
        alpha: float = 0.5,
        candidate_pool_size: int = 100,
        device: str = "cpu",
    ):
        # Validate that the specified device is either 'cpu' or 'cuda'
        valid_devices = {"cpu", "cuda"}
        if device not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}.")
        
        # Check CUDA availability if 'cuda' is requested
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please use 'cpu' instead.")

        # Validate input types to ensure correct usage
        if not isinstance(model, SentenceTransformer):
            raise TypeError("model must be an instance of SentenceTransformer.")
        if not isinstance(sentiment_model_name, str):
            raise TypeError("sentiment_model_name must be a string.")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float.")
        if not isinstance(candidate_pool_size, int):
            raise TypeError("candidate_pool_size must be an integer.")

        # Validate value ranges to prevent logical errors
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1 inclusive.")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be a positive integer.")

        # Initialize the superclass (KeyBERT) with the semantic embedding model
        super().__init__(model)

        # Assign validated parameters to instance variables
        self._alpha = None
        self.alpha = alpha
        self.candidate_pool_size = candidate_pool_size
        self.device = device

        # Store the semantic embedding model for embedding computation
        self.embedder = model

        # Initialize the sentiment model wrapper with the given model name and device
        self.sentiment_model = SentimentModel(sentiment_model_name, device=device)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {value}")
        self._alpha = value

    def _get_doc_polarity_continuous(self, doc: str) -> float:
        """
        Compute the document's continuous sentiment polarity score as the weighted sum of
        predicted class probabilities multiplied by their numeric mappings.

        This method overrides and replaces any default sentiment handling in the base class.

        Parameters:
        -----------
        doc : str
            The document text.

        Returns:
        --------
        float
            Continuous sentiment polarity score between 0 (very negative) and 1 (very positive).
        """
        # Get probability distribution over sentiment classes for the document
        probs = self.sentiment_model.predict_proba([doc])[0]

        # Compute continuous polarity as weighted average of class scores
        polarity = sum(
            p * self.sentiment_model.label_to_score[label]
            for p, label in zip(probs, self.sentiment_model.labels_ordered)
        )
        return polarity

    def _get_candidate_polarities(self, candidates) -> np.ndarray:
        """
        Compute continuous sentiment polarity scores for each candidate keyword.

        This method extends candidate scoring with sentiment, overriding base candidate processing.

        Parameters:
        -----------
        candidates : iterable of str
            List of candidate keywords.

        Returns:
        --------
        np.ndarray
            Array of polarity scores for each candidate keyword.
        """
        candidates = list(candidates)  # ensure correct input format for tokenizer
        
        # Batch predict probabilities for all candidates
        probs_list = self.sentiment_model.predict_proba(candidates)
        
        polarities = []
        for probs in probs_list:
            # Weighted average as continuous polarity score
            polarity = sum(
                p * self.sentiment_model.label_to_score[label]
                for p, label in zip(probs, self.sentiment_model.labels_ordered)
            )
            polarities.append(polarity)
        return np.array(polarities)

    def _select_candidates(
        self, 
        doc: str, 
        ngram_range: Tuple[int, int] = (1, 2), 
        threshold: float = 0.4,
        stop_words: str = 'english'
    ):
        """
        Extract initial candidates with CountVectorizer and filter them based on combined
        semantic similarity and sentiment alignment scores.

        This method replaces the default candidate generation and filtering steps of KeyBERT,
        incorporating sentiment filtering before final keyword ranking.

        Parameters:
        -----------
        doc : str
            Document text.

        ngram_range : tuple of int
            N-gram size range for candidate extraction.

        threshold : float
            Minimum combined score for candidate retention.

        Returns:
        --------
        list of str
            Filtered list of candidate keywords.
        """
        # Extract candidates with CountVectorizer (statistical n-grams)
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words,
            max_features=self.candidate_pool_size
        )
        candidates = vectorizer.fit([doc]).get_feature_names_out()

        # Compute semantic embeddings for doc and candidates
        doc_emb = self.model.embed([doc])
        cand_emb = self.model.embed(candidates)

        # Compute continuous sentiment polarity scores
        doc_pol = self._get_doc_polarity_continuous(doc)
        cand_pols = self._get_candidate_polarities(candidates)

        # Calculate cosine semantic similarity scores
        sim_scores = cosine_similarity(doc_emb, cand_emb)[0]

        # Calculate sentiment alignment scores
        sentiment_scores = 1 - np.abs(cand_pols - doc_pol)
        sentiment_scores_mapped = 2 * sentiment_scores - 1

        # Combine semantic and sentiment scores with alpha weighting
        combined_scores = self.alpha * sentiment_scores_mapped + (1 - self.alpha) * sim_scores

        # Filter candidates that meet threshold on combined score
        filtered_candidates = [c for c, s in zip(candidates, combined_scores) if s >= threshold]

        return filtered_candidates

    def extract_keywords(
        self,
        doc: str,
        top_n: int = 5,
        candidate_threshold: float = 0.4,
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        print_doc_polarity: bool = False,
        stop_words: str = 'english',
        **kwargs
    ):
        """
        Extract top keywords from a document by combining semantic similarity and sentiment alignment.

        This method overrides the `extract_keywords` method from KeyBERT base class,
        adding sentiment-aware candidate filtering and scoring.

        Parameters:
        -----------
        doc : str
            Input document text.

        top_n : int
            Number of keywords to return.

        candidate_threshold : float
            Threshold score to filter candidate keywords.

        keyphrase_ngram_range : tuple of int
            N-gram range for candidate keyword extraction.

        print_doc_polarity : bool
            Whether to print the document's sentiment polarity score.

        Returns
        -------
        list of tuples
            List of (keyword, score, keyword_sentiment) tuples sorted by descending combined score.

        """

        # Select candidates filtered by combined semantic+sentiment scoring
        candidates = self._select_candidates(
            doc,
            ngram_range=keyphrase_ngram_range,
            threshold=candidate_threshold,
            stop_words=stop_words
        )
        if not candidates:
            print("No candidates passed the sentiment-semantic filter.")
            return []

        # Compute semantic embeddings for document and filtered candidates
        doc_emb = self.model.embed([doc])
        cand_emb = self.model.embed(candidates)

        # Compute continuous sentiment polarity for the document
        doc_pol = self._get_doc_polarity_continuous(doc)

        # Print document polarity if requested
        if print_doc_polarity:
            # Scale polarity from [0,1] to [0,10]
            scaled_pol = doc_pol * 10

            # Determine polarity label with neutral zone between 4 and 6 on 0-10 scale
            if scaled_pol < 5.5:
                polarity_label = "Negative"
            elif scaled_pol > 6.5:
                polarity_label = "Positive"
            else:
                polarity_label = "Neutral"

            print(f"\n=== Document Polarity Score: {scaled_pol:.2f} ({polarity_label}) ===\n")

        # Compute sentiment polarities for candidates
        cand_pols = self._get_candidate_polarities(candidates)

        # Calculate cosine semantic similarity scores (range [-1,1])
        sim_scores = cosine_similarity(doc_emb, cand_emb)[0]

        # Calculate sentiment alignment scores in [0,1] and map to [-1,1]
        sentiment_scores = 1 - np.abs(cand_pols - doc_pol)
        sentiment_scores_mapped = 2 * sentiment_scores - 1

        # Final combined score with weighting factor alpha in [-1,1]
        final_scores = self.alpha * sentiment_scores_mapped + (1 - self.alpha) * sim_scores

        # Select top_n keywords sorted by combined score descending
        top_indices = np.argsort(final_scores)[-top_n:][::-1]

        return [(candidates[i], final_scores[i], cand_pols[i]) for i in top_indices]
