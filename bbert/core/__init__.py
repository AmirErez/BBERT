"""
Core functionality for BBERT.

This module contains the core configuration, dataset handling, and collation functionality.
"""

from bbert.core.config import *
from bbert.core.collator import CollateFnWithTokenizer, collate_fn
from bbert.core.dataset import FastqIterableDataset
from bbert.core import utils

__all__ = [
    # From config.py (all constants)
    "SEQUENCE_LENGTH",
    "HIDDEN_SIZE_384",
    "HIDDEN_SIZE_768",
    "NUM_BACTERIAL_CLASSES",
    "NUM_READING_FRAMES",
    "NUM_CODING_CLASSES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHUNK_MULTIPLIER",
    "LOG_INTERVAL",
    "DEFAULT_BBERT_MODEL_PATH_768",
    "DEFAULT_BBERT_MODEL_PATH_384",
    "DEFAULT_BACTERIAL_CLASSIFIER_PATH",
    "DEFAULT_FRAME_CLASSIFIER_PATH",
    "DEFAULT_CODING_CLASSIFIER_PATH",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_DROPOUT",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_PREFETCH_FACTOR",
    # From collator.py
    "CollateFnWithTokenizer",
    "collate_fn",
    # From dataset.py
    "FastqIterableDataset",
    # From utils.py (module)
    "utils",
]
