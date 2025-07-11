import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from math import log


from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import CountVectorizer

from keybert import KeyBERT # type: ignore
from keybert.backend._base import BaseEmbedder # type: ignore


class KeyBERTMetadata(KeyBERT):
    """
    A minimal class for keyword extraction based on KeyBERT

    This class overrides the embedding extraction method to emphasize the 
    importance of metadata associated with the text.

    """

    def __init__(
        self,
        model="all-MiniLM-L6-v2",
        **kwargs
    ):
        """KeyBERTMetadata initialization

        Arguments:
            model: selected BERT model (by default all-MiniLM-L6-v2)
        """
        super().__init__(model=model, **kwargs)

    
    def extract_embeddings_mean(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
        metadata: Optional[List[List[float]]] = None,
        optional_pruning: bool = True,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract document and word embeddings for the input documents and the
        generated candidate keywords/keyphrases respectively.

        Note that all potential keywords/keyphrases are not returned but only their
        word embeddings. This means that the values of `candidates`, `keyphrase_ngram_range`,
        `stop_words`, and `min_df` need to be the same between using `.extract_embeddings` and
        `.extract_keywords`.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            metadata: Metadata values for each document
            optional_pruning: If True, removes selected columns from the embeddings.

        Returns:
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.

        Usage:

        To generate the word and document embeddings from a set of documents:

        ```python
        from keybert import KeyBERT
        from KeyBertMetadata import KeyBERTMetadata
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        kw_model = KeyBERTMetadata(model=embedding_model)

        doc_embeddings, word_embeddings = kw_model.extract_embeddings(docs)

        keywords = kw_model.extract_keywords(docs, doc_embeddings=doc_embeddings, word_embeddings=word_embeddings)
        ```

        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        # Extract potential words using a vectorizer / tokenizer
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []

       
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()

        doc_embeddings = self.model.embed(docs)
        word_embeddings = self.model.embed(words)

        # Optional pruning of useless embeddings
        if optional_pruning:
            cols_to_remove = [127, 223, 319]

            # Remove specified columns from document embeddings
            doc_embeddings = np.delete(doc_embeddings, cols_to_remove, axis=1)

            # Remove specified columns from word embeddings
            word_embeddings = np.delete(word_embeddings, cols_to_remove, axis=1)


        # Enrich doc embeddings with metadata if provided
        if metadata is not None:
            if len(metadata) != len(doc_embeddings):
                raise ValueError("The number of metadata entries must match the number of documents")

            # Convert each embedding to np.array if not already
            doc_embeddings = [
                np.concatenate([np.asarray(doc_emb), np.asarray(meta)], axis=0)
                for doc_emb, meta in zip(doc_embeddings, metadata)
            ]

            # Compute document-term matrix
            X = count.transform(docs)
            metadata = np.asarray(metadata)

            enriched_word_embeddings = []
            for idx, word_emb in enumerate(word_embeddings):
                doc_indices = X[:, idx].nonzero()[0]
                if len(doc_indices) > 0:
                    meta_mean = np.mean(metadata[doc_indices], axis=0)
                else:
                    meta_mean = np.zeros_like(metadata[0])  # fallback for isolated words
                #uncomment to show computed values
                #print(f"Word: {words[idx]}\nmeta_mean: {meta_mean}\n")
                
                enriched = np.concatenate([np.asarray(word_emb), meta_mean], axis=0)
                enriched_word_embeddings.append(enriched)
            
            word_embeddings = enriched_word_embeddings



        return np.vstack(doc_embeddings), np.vstack(word_embeddings)

    

    @staticmethod
    def extract_metadata(df: pd.DataFrame, alpha: float = 0.3) -> list:
        """
        Extracts and normalizes metadata scores per review, treating each movie separately.
        Normalization maps each score into the range [-alpha, alpha].
        Constant features are assigned 0.0.

        Parameters:
            df (pd.DataFrame): Input DataFrame
            alpha (float): Half the desired output range (e.g., 0.3 for [-0.3, 0.3])

        Returns:
            List of lists: [utility, length, polarity, recency, controversy, rating_deviation]
        """
        df = df.copy()
        df['Review_Date'] = pd.to_datetime(df['Review_Date'])
        df['Release_Date'] = df.groupby('Movie_Title')['Review_Date'].transform('min')

        # Precompute per-movie rating means for deviation score
        df['mean_rating'] = df.groupby('Movie_Title')['Rating'].transform('mean')

        def compute_scores(row):
            # If helpful votes or total votes are NaN, set likes and total votes to 0
            if pd.isna(row['Helpful_Votes']) or pd.isna(row['Total_Votes']):
                likes = 0
                total_votes = 0
            else:
                likes = row['Helpful_Votes']
                total_votes = row['Total_Votes']
            
            dislikes = total_votes - likes

            utility_score = likes / total_votes if total_votes > 0 else 0.0
            length_score = len(str(row['Preprocessed_Review']))
            polarity_score = (likes - dislikes) / (total_votes + 1)
            days_since_release = (row['Review_Date'] - row['Release_Date']).days
            recency_score = 1 / log(days_since_release + 2)

            controversy_score = 1 - abs(polarity_score)
            
            # Handle NaN ratings
            if pd.isna(row['Rating']):
                rating_deviation = 0.0
            else:
                # Calculate the absolute deviation from the mean rating for the movie
                rating_deviation = abs(row['Rating'] - row['mean_rating'])

            return pd.Series([
                utility_score, length_score, polarity_score,
                recency_score, controversy_score, rating_deviation
            ])

        score_cols = ['utility', 'length', 'polarity', 'recency', 'controversy', 'rating_deviation']
        df[score_cols] = df.apply(compute_scores, axis=1)

        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([0.0] * len(series), index=series.index)
            normalized = (series - min_val) / (max_val - min_val)
            return (normalized - 0.5) * (2 * alpha)

        for col in score_cols:
            df[col] = df.groupby('Movie_Title')[col].transform(normalize)

        return df[score_cols].values.tolist()
