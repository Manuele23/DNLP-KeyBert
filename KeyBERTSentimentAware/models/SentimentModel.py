from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore
import torch
import torch.nn.functional as F
import numpy as np
import re

# Class definition
class SentimentModel:
    """
    A flexible sentiment analysis wrapper supporting multiple HuggingFace models.

    This class dynamically adapts to the label schema of the specified model,
    allowing for consistent polarity scoring across different sentiment models.
    It also supports automatic truncation and chunking for long texts that exceed
    the model's token limit, ensuring robust performance on lengthy inputs.

    For models with a token limit (e.g., BERT-based with 512 tokens), the class
    splits long texts into chunks, computes predictions for each, and returns
    the averaged sentiment probabilities.
    """

    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment", device="cpu"):
        """
        Initialize the sentiment model.

        Parameters:
        ----------
        model_name : str
            HuggingFace model identifier.
            Default is "cardiffnlp/twitter-roberta-base-sentiment".

        device : str
            Computation device. Should be either 'cpu' or 'cuda'.
        """

        # Validate the selected device
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please use 'cpu' instead.")

        self.device = device
        self.model_name = model_name

        # Load tokenizer and model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        # Determine label mapping based on the model
        self._set_label_mapping()

    def _set_label_mapping(self):
        """
        Set the label to score mapping based on the model's label schema.
        """

        # Retrieve the model's configuration to get label mappings
        id2label = self.model.config.id2label

        # Sort labels by their IDs to maintain order
        self.labels_ordered = [id2label[i] for i in range(len(id2label))]



        # Define label to score mapping based on known models

        # Using a weighted average of class probabilities mapped to sentiment scores (e.g., 0.0, 0.25, ..., 1.0)
        # is generally better than relying on a single compound score (if available). This approach:
        # - Provides full control over the sentiment scale and allows consistent interpretation.
        # - Aligns directly with the model's output (probabilities per class), rather than applying post-hoc rules.
        # - Works with any number of sentiment classes and generalizes well across models.
        # - Produces a continuous, interpretable score that is ideal for averaging, thresholding, and ranking.

        if self.model_name == "cardiffnlp/twitter-roberta-base-sentiment":
            # Labels: ['negative', 'neutral', 'positive']
            self.label_to_score = {
                'LABEL_0': 0.0, # negative
                'LABEL_1': 0.5, # neutral
                'LABEL_2': 1.0  # positive
            }
        elif self.model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
            # Labels: ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
            self.label_to_score = {
                '1 star': 0.0,
                '2 stars': 0.25,
                '3 stars': 0.5,
                '4 stars': 0.75,
                '5 stars': 1.0
            }
        else:
            # For unknown models, assign scores evenly across labels
            num_labels = len(self.labels_ordered)
            self.label_to_score = {
                label: idx / (num_labels - 1) for idx, label in enumerate(self.labels_ordered)
            }

    def _split_into_chunks(self, text, max_length=512):
        """
        Split long texts into smaller chunks that do not exceed the token limit.

        Parameters:
        ----------
        text : str
            The input text to split.

        max_length : int
            Maximum token length allowed by the model.

        Returns:
        -------
        List[str]
            A list of text chunks within the token limit.
        """

        # Naively split on sentence delimiters
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current = [], ""

        for sent in sentences:
            # Check tokenized length with current buffer
            if len(self.tokenizer.encode(current + " " + sent, add_special_tokens=True)) <= max_length:
                current += " " + sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())

        return chunks

    def predict_proba(self, texts, max_length=512):
        """
        Compute the probability distribution over sentiment classes for one or more input texts.

        Parameters:
        ----------
        texts : List[str]
            List of text strings to analyze.

        max_length : int
            Maximum token length per chunk (default: 512).

        Returns:
        -------
        np.ndarray
            A 2D array of shape (len(texts), num_classes), where each row represents
            the predicted softmax probabilities for the corresponding input.
        """

        all_probs = []

        for text in texts:
            # Always chunk to avoid loss of info on long inputs
            chunks = self._split_into_chunks(text, max_length=max_length)
            chunk_probs = []

            for chunk in chunks:
                # Tokenize and infer sentiment per chunk
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True,
                                        padding=True, max_length=max_length).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
                    chunk_probs.append(probs)

            # Average the probabilities over all chunks
            avg_probs = np.mean(chunk_probs, axis=0)
            all_probs.append(avg_probs)

        return np.array(all_probs)
    
    
    def predict_score(self, text):
        """
        Compute the continuous sentiment score for a single input text.

        Parameters:
        ----------
        text : str
            The input text to analyze.

        Returns:
        -------
        float
            The sentiment score in the range [0, 1].
        """

        probs = self.predict_proba([text])[0]
        score = sum(
            prob * self.label_to_score[label]
            for prob, label in zip(probs, self.labels_ordered)
        )
        return score
