"""
Utility functions for inference.py.
This module contains testable helper functions extracted from inference.py.
"""

import os
import torch
from transformers import BertForMaskedLM, AutoTokenizer


def get_device(logger=None):
    """
    Detect and return the best available device for inference.

    Args:
        logger: Optional logger instance for logging device selection

    Returns:
        tuple: (device, use_half_precision) where device is torch.device
               and use_half_precision is bool
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if logger:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info(f"Using CUDA GPU: {gpu_name}")
        use_half_precision = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if logger:
            logger.info("Using Apple MPS (Metal Performance Shaders)")
        use_half_precision = False  # MPS doesn't support float16 well
    else:
        device = torch.device('cpu')
        if logger:
            logger.info("Using CPU (no GPU acceleration available)")
        use_half_precision = False  # CPU doesn't support float16 efficiently

    return device, use_half_precision


def get_output_filename(file_path, output_dir, emb_out=False):
    """
    Generate output filename from input file path.

    Args:
        file_path: Input file path
        output_dir: Output directory
        emb_out: Whether embeddings are included

    Returns:
        str: Output file path

    Examples:
        >>> get_output_filename("/path/to/file.fasta", "/output", False)
        '/output/file_scores_len.parquet'
        >>> get_output_filename("/path/to/file.fasta.gz", "/output", True)
        '/output/file_scores_len_emb.parquet'
    """
    # Extract base filename for output (without directory and extensions)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Handle .gz files (remove .gz and then the next extension)
    if base_name.endswith('.fasta') or base_name.endswith('.fastq'):
        base_name = os.path.splitext(base_name)[0]

    if emb_out:
        return os.path.join(output_dir, f"{base_name}_scores_len_emb.parquet")
    else:
        return os.path.join(output_dir, f"{base_name}_scores_len.parquet")


def load_bbert_model(model_path, tokenizer_path, device, use_half_precision, logger=None):
    """
    Load BBERT model and tokenizer.

    Args:
        model_path: Path to BBERT model
        tokenizer_path: Path to tokenizer
        device: torch.device to load model on
        use_half_precision: Whether to use float16
        logger: Optional logger instance

    Returns:
        tuple: (model, tokenizer, collate_fn)
    """
    from BERT_model.collator import CollateFnWithTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    collate_fn_instance = CollateFnWithTokenizer(tokenizer)

    model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    if use_half_precision:
        model.half()
    model.to(device)

    if logger:
        logger.info(f"BBERT model loaded from {model_path}")

    return model, tokenizer, collate_fn_instance


def load_classifier(model_path, hidden_size, num_classes, device, use_half_precision, logger=None, model_name="Classifier"):
    """
    Load a BertClassifier model.

    Args:
        model_path: Path to classifier checkpoint
        hidden_size: Hidden layer size
        num_classes: Number of output classes
        device: torch.device to load model on
        use_half_precision: Whether to use float16
        logger: Optional logger instance
        model_name: Name for logging

    Returns:
        BertClassifier model
    """
    from emb_model.architecture import BertClassifier

    classifier = BertClassifier(hidden_size, num_classes)
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    if use_half_precision:
        classifier.half()
    classifier.to(device)

    if logger:
        logger.info(f"{model_name} model loaded from {model_path}")

    return classifier
