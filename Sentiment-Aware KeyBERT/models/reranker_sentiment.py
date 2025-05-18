# You need to install the following packages:
# pip install keybert
# pip install sentence-transformers
# pip install transformers

from typing import List, Sequence, Tuple, Union
from keybert import KeyBERT  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sentiment_model import SentimentModel  

# KeyBERT Post-hoc Sentiment-Aware Re-ranking

# This module extends KeyBERT to include sentiment-aware keyword extraction by defining 
# a subclass of KeyBERT that modifies the scoring phase to incorporate sentiment alignment 
# in a post-processing step. The goal is to boost keywords that are both semantically relevant 
# and emotionally aligned with the overall sentiment of the input review.

# Sentiment-Aware KeyBERT Class
class KeyBERTSentimentReranker(KeyBERT):
    """
    A KeyBERT extension that performs sentiment-aware post-hoc re-ranking.

    Parameters
    ----------
    model : str or SentenceTransformer
        The embedding model used to generate document and keyword embeddings.
        By default, we use 'all-MiniLM-L6-v2', a lightweight and efficient SBERT model
        that balances speed and semantic accuracy.

    alpha : float, default=0.5
        This parameter controls the trade-off between semantic similarity and sentiment alignment.
        When alpha = 0, only cosine similarity is considered (i.e., standard KeyBERT).
        When alpha = 1, only sentiment alignment is used. 
        A mid-value like 0.5 allows semantic meaning to dominate while still incorporating sentiment consistency.

    sentiment_model_name : str, optional
        Identifier of the pretrained HuggingFace sentiment model to be used.

    device : str, optional
        Computation device ("cpu" or "cuda").
    """

    def __init__(
        self,
        model: Union[str, "SentenceTransformer"] = "sentence-transformers/all-MiniLM-L6-v2",  # type: ignore
        *,
        alpha: float = 0.5,
        sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in the interval [0, 1]")

        if not isinstance(model, SentenceTransformer):
            raise TypeError("model must be a SentenceTransformer")

        self._alpha = None
        self.alpha = alpha
        self.device = device
        self.sentiment_model = SentimentModel(sentiment_model_name, device=self.device)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {value}")
        self._alpha = value

    # Compute continuous polarity score ∈ [0,1]
    def get_sentiment_score(self, text: str) -> float:
        probs = self.sentiment_model.predict_proba([text])[0]
        return sum(
            p * self.sentiment_model.label_to_score[label]
            for p, label in zip(probs, self.sentiment_model.labels_ordered)
        )

    # Forward all additional keyword arguments (e.g., keyphrase_ngram_range, stop_words, use_maxsum)
    # to the original KeyBERT extract_keywords() method. This allows the user to customize the
    # underlying behavior of KeyBERT without redefining all parameters here.
    # This is possible because KeyBERT uses **kwargs to accept a wide range of parameters.
    def extract_keywords(
        self,
        docs: Union[str, Sequence[str]],
        *,
        top_n: int = 10,
        **kwargs,
    ) -> List[List[Tuple[str, float]]]:
        """
        Returns top-n keywords using sentiment-aware re-ranking.

        Steps:
        1. Use standard KeyBERT to extract candidate keywords and cosine similarity scores.
        2. Compute sentiment polarity scores for the full review and each keyword.
        3. For each keyword, calculate an alignment score with the review's sentiment.
        4. Fuse cosine similarity and alignment using the alpha parameter.
        5. Return the top-n keywords sorted by the new combined score.
        """

        # Normalize input to a list of reviews
        if isinstance(docs, str):
            docs = [docs]
            single_doc = True
        else:
            single_doc = False

        # Step 1: extract keywords from KeyBERT
        # Output: base_keywords[review_index] = List[Tuple[keyword, cosine_sim]]
        base_keywords = super().extract_keywords(docs, top_n=top_n, **kwargs)

        # Ensure base_keywords is always List[List[Tuple[str, float]]] even if there is only 1 review
        if isinstance(base_keywords, list) and all(isinstance(k, tuple) for k in base_keywords):
            base_keywords = [base_keywords]

        # Step 2: compute sentiment for each review
        # Output: doc_sentiments[review_index] = float in [0, +1]
        doc_sentiments = [self.get_sentiment_score(doc) for doc in docs]

        # Step 3–4: re-rank using sentiment alignment
        # Output: reranked_all[review_index] = List[Tuple[keyword, fused_score]]
        reranked_all: List[List[Tuple[str, float]]] = []

        for kws, s_doc in zip(base_keywords, doc_sentiments):
            # One list of re-ranked keywords for this review
            reranked_doc: List[Tuple[str, float]] = []

            # Normalize single-tuple edge case into list
            if isinstance(kws, tuple) and len(kws) == 2 and isinstance(kws[0], str):
                kws = [kws]

            for kw_data in kws:
                try:
                    if isinstance(kw_data, tuple) and len(kw_data) == 2:
                        kw, cos_sim = kw_data
                    else:
                        raise ValueError("Expected a (keyword, score) tuple")

                    if not isinstance(kw, str) or not isinstance(cos_sim, (float, int)):
                        raise ValueError("Malformed keyword-score pair")

                    # Step 3: Compute sentiment for keyword and alignment score
                    s_kw = self.get_sentiment_score(kw)
                    align = 1.0 - abs(s_doc - s_kw)      # alignment ∈ [0, 1]
                    align_mapped = 2 * align - 1         # mapped to [-1, 1]

                    # Step 4: Combine cosine and alignment via convex combination
                    final_score = round((1.0 - self.alpha) * cos_sim + self.alpha * align_mapped, 4)

                    # Store keyword with new adjusted score
                    reranked_doc.append((kw, final_score))

                except Exception as e:
                    print(f"Skipping invalid keyword data {kw_data}: {e}")

            # Sort keywords for this review by adjusted score
            reranked_doc.sort(key=lambda x: x[1], reverse=True)
            reranked_all.append(reranked_doc[:top_n])

        return [reranked_all[0]] if single_doc else reranked_all
