# Required imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

class SentimentModel:
    """
    A flexible sentiment analysis wrapper supporting multiple HuggingFace models.

    This class dynamically adapts to the label schema of the specified model,
    allowing for consistent polarity scoring across different sentiment models.
    """

    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment", device="cpu"):
        """
        Initialize the sentiment model.

        Parameters:
        ----------
        model_name : str
            HuggingFace model identifier.

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
        if self.model_name == "cardiffnlp/twitter-roberta-base-sentiment":
            # Labels: ['negative', 'neutral', 'positive']
            self.label_to_score = {
                'negative': 0.0,
                'neutral': 0.5,
                'positive': 1.0
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

    def predict_proba(self, texts):
        """
        Compute the probability distribution over sentiment classes for one or more input texts.

        Parameters:
        ----------
        texts : List[str]
            List of text strings to analyze.

        Returns:
        -------
        np.ndarray
            A 2D array of shape (len(texts), num_classes), where each row represents
            the predicted softmax probabilities for the corresponding input.
        """

        # Tokenize and encode the input text(s) with padding and truncation
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Forward pass through the model without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # raw prediction scores before softmax

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=1).cpu().numpy()

        return probs

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
