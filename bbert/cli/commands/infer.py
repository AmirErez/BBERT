"""
BBERT Inference Command

Run BBERT inference on DNA sequences from FASTA/FASTQ files.
"""

import os
import sys
import multiprocessing as mp
import time
import gc
import platform
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Suppress torch.load warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Import from bbert package
from bbert.core.dataset import FastqIterableDataset
from bbert.core.utils import clear_GPU, setup_logger, get_resources_msg, get_slurm_cpus
from bbert.core.config import (
    SEQUENCE_LENGTH,
    HIDDEN_SIZE_768,
    HIDDEN_SIZE_384,
    NUM_BACTERIAL_CLASSES,
    NUM_READING_FRAMES,
    NUM_CODING_CLASSES,
    LOG_INTERVAL,
    DEFAULT_BBERT_MODEL_PATH_768,
    DEFAULT_BBERT_MODEL_PATH_384,
    DEFAULT_BACTERIAL_CLASSIFIER_PATH,
    DEFAULT_FRAME_CLASSIFIER_PATH,
    DEFAULT_CODING_CLASSIFIER_PATH,
    DEFAULT_BACTERIAL_CLASSIFIER_PATH_384,
    DEFAULT_FRAME_CLASSIFIER_PATH_384,
    DEFAULT_CODING_CLASSIFIER_PATH_384,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_NUM_WORKERS
)
from bbert.utils.inference import get_device, get_output_filename, load_bbert_model, load_classifier

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


def add_arguments(parser):
    """Add inference command arguments to parser."""
    parser.add_argument(
        "files",
        nargs='+',
        type=str,
        help="Input file paths (FASTA/FASTQ/GZ). Supports wildcards and multiple files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output Parquet files (required)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for processing (default: 1024)"
    )
    parser.add_argument(
        "--emb-out",
        action='store_true',
        help="Include sequence embeddings in output (warning: slow and large files, requires --max-reads)"
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        help="Maximum number of reads to process per file (default: process all reads)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        choices=[384, 768],
        default=384,
        help="Hidden size of BBERT model to use (default: 384)"
    )


def get_model_paths(hidden_size, bbert_dir):
    """Get model paths based on hidden size and BBERT directory."""
    if hidden_size == HIDDEN_SIZE_384:
        bbert_model_path = os.path.join(bbert_dir, DEFAULT_BBERT_MODEL_PATH_384)
        bact_class_model_path = os.path.join(bbert_dir, DEFAULT_BACTERIAL_CLASSIFIER_PATH_384)
        frame_class_model_path = os.path.join(bbert_dir, DEFAULT_FRAME_CLASSIFIER_PATH_384)
        class_model_path = os.path.join(bbert_dir, DEFAULT_CODING_CLASSIFIER_PATH_384)
        return bbert_model_path, bact_class_model_path, frame_class_model_path, class_model_path
    elif hidden_size == HIDDEN_SIZE_768:
        bbert_model_path = os.path.join(bbert_dir, DEFAULT_BBERT_MODEL_PATH_768)
        bact_class_model_path = os.path.join(bbert_dir, DEFAULT_BACTERIAL_CLASSIFIER_PATH)
        frame_class_model_path = os.path.join(bbert_dir, DEFAULT_FRAME_CLASSIFIER_PATH)
        class_model_path = os.path.join(bbert_dir, DEFAULT_CODING_CLASSIFIER_PATH)
        return bbert_model_path, bact_class_model_path, frame_class_model_path, class_model_path
    else:
        raise ValueError(f"Unsupported hidden size: {hidden_size}")


def process_file(
    file_path,
    output_dir,
    batch_size,
    emb_out,
    max_reads,
    bbert_model,
    tokenizer,
    collate_fn_instance,
    bact_classifier,
    frame_classifier,
    coding_classifier,
    device,
    hidden_size,
    num_workers,
    prefetch_factor,
    logger
):
    """Process a single input file."""
    chunk_size = batch_size * 2
    seq_len = SEQUENCE_LENGTH
    frame_classes = NUM_READING_FRAMES

    # Validate input file exists
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        logger.error("Please check that the file path is correct.")
        sys.exit(1)

    # Generate output filename
    output_path = get_output_filename(file_path, output_dir, emb_out)

    logger.info(f"Processing file: {file_path}")
    try:
        dataset = FastqIterableDataset(file_path, chunk_size=chunk_size, max_reads=max_reads)
        seq_lens, data_len = dataset.get_stats()
    except PermissionError:
        logger.error(f"Permission denied when reading file: {file_path}")
        logger.error("Please check that you have read permissions for this file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {str(e)}")
        logger.error("Please check that the file is a valid FASTA/FASTQ file.")
        sys.exit(1)

    logger.info(f"{len(seq_lens)} reads, {min(seq_lens)} <= len <= {max(seq_lens)}")
    if max_reads and data_len >= max_reads:
        logger.info(f"Limited to first {data_len} reads (--max-reads {max_reads})")

    # Configure DataLoader parameters
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'collate_fn': collate_fn_instance,
        'num_workers': num_workers,
        'pin_memory': True
    }

    # Only add multiprocessing parameters if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = True

    dataloader = DataLoader(**dataloader_kwargs)
    batch_num = data_len // batch_size + 1 if data_len % batch_size != 0 else data_len // batch_size

    # Reserve memory for results
    seq_ids = np.empty(data_len, dtype=object)
    bact_probs = torch.zeros(data_len, dtype=torch.float32, device=device)
    loss_out = torch.zeros(data_len, dtype=torch.float32, device=device)
    frame_probs = torch.zeros(data_len, frame_classes, dtype=torch.float32, device=device)
    coding_probs = torch.zeros(data_len, dtype=torch.float32, device=device)

    if emb_out:
        emb_array = np.empty((data_len, seq_len, hidden_size), dtype=np.float16)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    batch_start = 0
    inference_start_time = time.time()
    iter_reads = 0
    iter_start_time = time.time()

    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            batch_len = len(batch[0])
            seq_ids[batch_start:batch_start+batch_len] = batch[0]
            input_ids = batch[1].to(device=device)
            attention_mask = batch[2].to(device=device)

            outputs = bbert_model(input_ids=input_ids, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
            bbert_logits = outputs.logits

            if emb_out:
                batch_embs = embeddings.detach().cpu().numpy().astype(np.float16)
                emb_array[batch_start:batch_start + batch_len, :, :] = batch_embs

            bact_class_logits = bact_classifier(embeddings).to(device)
            bact_probs[batch_start:batch_start+batch_len] = F.softmax(bact_class_logits, dim=1)[:, 1]

            frame_logits = frame_classifier(embeddings).to(device)
            frame_probs[batch_start:batch_start+batch_len] = F.softmax(frame_logits, dim=1)

            coding_class_logits = coding_classifier(embeddings).to(device)
            coding_probs[batch_start:batch_start+batch_len] = F.softmax(coding_class_logits, dim=1)[:, 1]

            # Compute loss
            logits_flat = bbert_logits.view(-1, bbert_logits.size(-1))
            targets_flat = input_ids.view(-1)
            token_losses = loss_fn(logits_flat, targets_flat)
            token_losses = token_losses.view(bbert_logits.size(0), -1)
            loss_per_read = (token_losses * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            loss_out[batch_start:batch_start+batch_len] = loss_per_read

            iter_reads += batch_len

            if (batch_id + 1) % LOG_INTERVAL == 0 or batch_id + 1 == batch_num:
                iter_time = time.time() - iter_start_time
                mean_speed = iter_reads / iter_time
                msg = f"[{batch_id + 1}/{batch_num}] Mean speed: {mean_speed:.2f} reads/sec over {LOG_INTERVAL} batches"
                resources_msg = get_resources_msg()
                logger.info(f'{msg} \t{resources_msg}')
                iter_reads = 0
                iter_start_time = time.time()

            batch_start += batch_len

    speed = data_len / (time.time() - inference_start_time)
    logger.info(f"mean speed: {speed:.2f} reads/sec")

    # Prepare output
    if emb_out:
        flat_emb_array = emb_array.reshape(-1, hidden_size)
        flat_pa = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_emb_array.ravel(), type=pa.float16()),
            list_size=hidden_size
        )
        emb_pa = pa.FixedSizeListArray.from_arrays(pa.array(flat_pa), list_size=seq_len)
        emb_shape = (len(emb_pa), emb_pa.type.list_size, emb_pa.type.value_type.list_size)
        logger.info(f"Embeddings Arrow shape: {emb_shape}")

    frame_array = frame_probs.cpu().numpy().astype(np.float32)
    frame_pa = pa.FixedSizeListArray.from_arrays(
        pa.array(frame_array.ravel(), type=pa.float32()),
        list_size=frame_classes
    )

    columns = {
        "id": pa.array(seq_ids, type=pa.string()),
        "len": pa.array(seq_lens, type=pa.uint16()),
        "loss": pa.array(loss_out.cpu().numpy()),
        "bact_prob": pa.array(bact_probs.cpu().numpy()),
        "frame_prob": frame_pa,
        "coding_prob": pa.array(coding_probs.cpu().numpy()),
    }
    if emb_out:
        columns["embedding"] = emb_pa
    output_table = pa.table(columns)

    # Write output
    try:
        pq.write_table(output_table, output_path, compression="zstd")
        logger.info(output_table.slice(0, 5).to_pandas())
        logger.info(f"Results saved to {output_path}")
    except PermissionError:
        logger.error(f"Permission denied when writing to {output_path}")
        logger.error("Please check that you have write permissions for this location.")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Failed to write results to {output_path}")
        logger.error(f"Reason: {str(e)}")
        if "No space left on device" in str(e):
            logger.error("You may have run out of disk space.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error writing results to {output_path}: {str(e)}")
        sys.exit(1)

    # Cleanup
    del dataset, dataloader, input_ids, outputs, embeddings
    del bbert_logits, bact_class_logits, frame_logits, coding_class_logits
    del bact_probs, frame_probs, coding_probs, loss_out, token_losses
    del loss_per_read, seq_ids

    torch.cuda.empty_cache()
    gc.collect()


def main(args):
    """Main inference function."""
    input_files = args.files
    output_dir = args.output_dir
    batch_size = args.batch_size
    emb_out = args.emb_out
    max_reads = args.max_reads
    hidden_size = args.hidden_size

    # Safety check
    if emb_out and max_reads is None:
        print("Error: --emb-out requires --max-reads to prevent creating huge files.", file=sys.stderr)
        print("Example: bbert infer file.fasta --output-dir results --emb-out --max-reads 1000", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        print(f"Error: Permission denied when creating output directory: {output_dir}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Failed to create output directory: {output_dir}", file=sys.stderr)
        print(f"Reason: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    verbose = True
    try:
        logger = setup_logger(verbose, log_file=os.path.join(output_dir, "inference.log"))
    except (PermissionError, OSError) as e:
        print(f"Warning: Failed to create log file in {output_dir}: {str(e)}", file=sys.stderr)
        print("Continuing without file logging...", file=sys.stderr)
        logger = setup_logger(verbose, log_file=None)

    logger.info(f'batch size = {batch_size//1024}K, chunk size = {(batch_size*2)//1024}K')

    # Configure workers
    slurm_cpus = get_slurm_cpus()
    if platform.system() in ["Darwin", "Windows"]:
        num_workers = DEFAULT_NUM_WORKERS
    else:
        num_workers = max(1, min(slurm_cpus, mp.cpu_count()) - 1)
    prefetch_factor = DEFAULT_PREFETCH_FACTOR
    logger.info(f"CPUs allocated by SLURM: {slurm_cpus}, workers_num = {num_workers}")

    # Device selection
    device, use_half_precision = get_device(logger)

    # Get BBERT directory (assumes installed package or running from repo)
    # Try to find models in package directory or current directory
    import bbert
    bbert_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(bbert.__file__)))
    if not os.path.exists(os.path.join(bbert_pkg_dir, "models")):
        # Fall back to current directory
        bbert_dir = os.getcwd()
    else:
        bbert_dir = bbert_pkg_dir

    # Get model paths
    try:
        bbert_model_path, bact_class_model_path, frame_class_model_path, class_model_path = get_model_paths(
            hidden_size, bbert_dir
        )
    except (NotImplementedError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Load models
    tokenizer_path = bbert_model_path
    bbert_model, tokenizer, collate_fn_instance = load_bbert_model(
        bbert_model_path, tokenizer_path, device, use_half_precision, logger
    )

    bact_classifier = load_classifier(
        bact_class_model_path, hidden_size, NUM_BACTERIAL_CLASSES, device, use_half_precision, logger, "Bacterial classifier"
    )

    frame_classifier = load_classifier(
        frame_class_model_path, hidden_size, NUM_READING_FRAMES, device, use_half_precision, logger, "Frame classifier"
    )

    coding_classifier = load_classifier(
        class_model_path, hidden_size, NUM_CODING_CLASSES, device, use_half_precision, logger, "Coding classifier"
    )

    # Process each file
    for file_path in input_files:
        process_file(
            file_path, output_dir, batch_size, emb_out, max_reads,
            bbert_model, tokenizer, collate_fn_instance,
            bact_classifier, frame_classifier, coding_classifier,
            device, hidden_size, num_workers, prefetch_factor, logger
        )

    # Final cleanup
    clear_GPU()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    logger.info("Inference complete!")
