"""
BBERT: BERT for Bacterial DNA Classification

A BERT-based transformer model for DNA sequence analysis, specifically designed
for bacterial sequence classification and genomic feature prediction.

The model performs three key classification tasks:
- Bacterial vs. non-bacterial classification
- Reading frame prediction (6 frames: +1,+2,+3,-1,-2,-3)
- Coding vs. non-coding sequence classification

Quick Start:
    >>> from bbert import load_bbert_model, BertClassifier
    >>> import torch
    >>>
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model, tokenizer, collate_fn = load_bbert_model("default", device)

For more information, see: https://github.com/AmirErez/BBERT
"""

from bbert._version import __version__, __version_info__

# Import core functionality
from bbert.core.config import (
    SEQUENCE_LENGTH,
    HIDDEN_SIZE_384,
    HIDDEN_SIZE_768,
    NUM_BACTERIAL_CLASSES,
    NUM_READING_FRAMES,
    NUM_CODING_CLASSES,
)
from bbert.core.dataset import FastqIterableDataset
from bbert.core.collator import CollateFnWithTokenizer

# Import models
from bbert.models.classifier import BertClassifier

# Import utilities
from bbert.utils.common import setup_logging, get_predicted_frame, FRAME_MAPPING
from bbert.utils.inference import (
    get_device,
    load_bbert_model,
    load_classifier,
    get_output_filename,
)

__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    # Core configuration
    "SEQUENCE_LENGTH",
    "HIDDEN_SIZE_384",
    "HIDDEN_SIZE_768",
    "NUM_BACTERIAL_CLASSES",
    "NUM_READING_FRAMES",
    "NUM_CODING_CLASSES",
    # Core classes
    "FastqIterableDataset",
    "CollateFnWithTokenizer",
    # Models
    "BertClassifier",
    # Utilities
    "setup_logging",
    "get_predicted_frame",
    "FRAME_MAPPING",
    "get_device",
    "load_bbert_model",
    "load_classifier",
    "get_output_filename",
]

# Package metadata
__author__ = "Amir Erez et al."
__email__ = "amir.erez@mail.huji.ac.il"
__license__ = "MIT"
