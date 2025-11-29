import torch
import torch.nn as nn

from bbert.core.utils import setup_logger
from bbert.core.config import SEQUENCE_LENGTH

verbose = True  # or use argparse to pass --verbose
logger = setup_logger(verbose)


class BertClassifier(nn.Module):
    """
    Simple linear classifier for BBERT embeddings.

    Takes flattened BBERT embeddings and applies a linear layer for classification.
    """
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        # Linear layer to classify the entire sequence
        # Multiply by seq_length (102) to get flattened embedding size
        self.classifier = nn.Linear(hidden_size * SEQUENCE_LENGTH, num_classes)

    def forward(self, x):
        # Flatten the sequence embeddings for each sample to [batch_size, hidden_size * seq_length]
        x = x.view(x.size(0), -1)  # Flatten along the sequence length dimension

        # Apply the classifier to the flattened embeddings
        logits = self.classifier(x)  # Shape: [batch_size, num_classes]

        return logits
