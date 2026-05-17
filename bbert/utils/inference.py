"""
Utility functions for inference.
This module contains testable helper functions for model loading and inference.
"""

import os
from typing import Optional, Tuple
import torch
from transformers import BertForMaskedLM, AutoTokenizer
from logging import Logger


def get_device(logger: Optional[Logger] = None) -> Tuple[torch.device, bool]:
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


def get_output_filename(file_path: str, output_dir: str, emb_out: bool = False) -> str:
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


def load_bbert_model(
    model_path: str,
    tokenizer_path: str,
    device: torch.device,
    use_half_precision: bool,
    logger: Optional[Logger] = None
) -> Tuple[BertForMaskedLM, AutoTokenizer, 'CollateFnWithTokenizer']:
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

    Raises:
        FileNotFoundError: If model files don't exist
        RuntimeError: If model loading fails
    """
    from bbert.core.collator import CollateFnWithTokenizer

    # Validate model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"BBERT model not found at: {model_path}\n"
            "Run 'bbert download' to download models from Hugging Face."
        )

    # Check for required model files (safetensors or legacy pytorch_model.bin)
    has_weights = (
        os.path.exists(os.path.join(model_path, 'model.safetensors')) or
        os.path.exists(os.path.join(model_path, 'pytorch_model.bin'))
    )
    has_config = os.path.exists(os.path.join(model_path, 'config.json'))
    if not has_weights or not has_config:
        missing = []
        if not has_weights:
            missing.append('model.safetensors (or pytorch_model.bin)')
        if not has_config:
            missing.append('config.json')
        raise FileNotFoundError(
            f"Missing required BBERT model files: {missing}\n"
            f"Model directory: {model_path}\n"
            "Run 'bbert download' to download models from Hugging Face."
        )

    try:
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

    except Exception as e:
        raise RuntimeError(
            f"Failed to load BBERT model from {model_path}: {str(e)}\n"
            "This might be a corrupted model file. Try re-downloading with:\n"
            "bbert download"
        ) from e


def load_classifier(
    model_path: str,
    hidden_size: int,
    num_classes: int,
    device: torch.device,
    use_half_precision: bool,
    logger: Optional[Logger] = None,
    model_name: str = "Classifier"
) -> 'BertClassifier':
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

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    from bbert.models.classifier import BertClassifier

    # Validate checkpoint file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_name} checkpoint not found at: {model_path}\n"
            "Run 'bbert download' to download models from Hugging Face."
        )

    try:
        classifier = BertClassifier(hidden_size, num_classes)
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)

        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise KeyError(
                f"Invalid checkpoint format: missing 'model_state_dict' key.\n"
                f"Checkpoint file: {model_path}"
            )

        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        if use_half_precision:
            classifier.half()
        classifier.to(device)

        if logger:
            logger.info(f"{model_name} model loaded from {model_path}")

        return classifier

    except Exception as e:
        raise RuntimeError(
            f"Failed to load {model_name} from {model_path}: {str(e)}\n"
            "This might be a corrupted checkpoint file. Try re-downloading with:\n"
            "bbert download"
        ) from e
