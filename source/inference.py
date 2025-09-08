import os
import sys
import multiprocessing as mp

import time
import gc
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BERT_model.dataset import FastqIterableDataset
from BERT_model.utils import clear_GPU, setup_logger, label_to_frame, get_resources_msg
from BERT_model.collator import CollateFnWithTokenizer
from emb_class_frame.architecture import BertClassifier

os.environ["WANDB_DISABLED"] = "true"

slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
# Mac has multiprocessing issues with DataLoader, use single worker
import platform
if platform.system() == "Darwin":  # macOS
    num_workers = 0
else:
    num_workers = max(1, min(slurm_cpus, mp.cpu_count()) - 1)
prefetch_factor = 2

data_len = None
seq_len = 102
hidden_size = 768
bact_classes = 2
frame_classes = 6
coding_classes = 2
log_interval = 100

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up (..)
bbert_dir = os.path.abspath(os.path.join(script_dir, ".."))

if hidden_size == 384:
    bbert_model_path = f'{bbert_dir}/models/diverse_bact_3_384_6_50000Ks/checkpoint-7000'
elif hidden_size == 768:
    bbert_model_path = f'{bbert_dir}/models/diverse_bact_12_768_6_20000/checkpoint-32500'
    bact_class_model_path = f'{bbert_dir}/emb_class_bact/models/emb_class_model_768H_3906K_80e/epoch_80.pt'
    frame_class_model_path = f'{bbert_dir}/emb_class_frame/models/classifier_model_2000K_37e.pth'
    class_model_path = f'{bbert_dir}/emb_class_coding/models/emb_coding_model_768_3906K_50e/epoch_46.pt'
    
# Argument parsing with detailed help
description = """
BBERT - BERT for Bacterial DNA Classification

BBERT is a BERT-based transformer model for DNA sequence analysis that performs:
- Bacterial vs. non-bacterial classification  
- Reading frame prediction (6 frames: +1,+2,+3,-1,-2,-3)
- Coding vs. non-coding sequence classification

Supports FASTA, FASTQ, and compressed (.gz) input files.
"""

epilog = """
EXAMPLES:
  # Single file
  python source/inference.py example/sample.fasta --output_dir results
  
  # Multiple files  
  python source/inference.py file1.fasta file2.fastq.gz --output_dir results --batch_size 512
  
  # Using wildcards
  python source/inference.py example/*.fasta.gz --output_dir results
  
  # With embeddings (warning: large files)
  python source/inference.py example/Pseudomonas_*.fasta.gz --output_dir results --emb_out
  
  # Process limited reads for testing
  python source/inference.py large_file.fasta.gz --output_dir test --max_reads 1000
  
  # All example files with embeddings and read limit
  python source/inference.py example/Pseudomonas_*.fasta.gz example/Saccharomyces_*.fasta.gz --output_dir results --emb_out --max_reads 5000

OUTPUT FILES:
  - Without --emb_out: {filename}_scores_len.parquet
  - With --emb_out: {filename}_scores_len_emb.parquet (much larger)

OUTPUT COLUMNS:
  - id: Sequence identifier
  - len: Sequence length  
  - loss: Cross-entropy loss value
  - bact_prob: Bacterial classification probability (0-1)
  - frame_prob: Reading frame probabilities (array of 6 values)
  - coding_prob: Coding sequence probability (0-1)  
  - embedding: Sequence embeddings (only with --emb_out)

SYSTEM REQUIREMENTS:
  - Python 3.10+
  - PyTorch, Transformers, BioPython, pandas, numpy, pyarrow
  - GPU recommended (CUDA/MPS) but CPU supported
  - Git LFS for model files

For more information: https://github.com/AmirErez/BBERT
"""

parser = argparse.ArgumentParser(
    description=description,
    epilog=epilog,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument("files", nargs='+', type=str, 
                   help="Input file paths (FASTA/FASTQ/GZ). Supports wildcards and multiple files.")
parser.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save output Parquet files (required)")
parser.add_argument("--batch_size", type=int, default=1024,
                   help="Batch size for processing (default: 1024)")
parser.add_argument("--emb_out", action='store_true',
                   help="Include sequence embeddings in output (warning: slow and large files)")
parser.add_argument("--max_reads", type=int,
                   help="Maximum number of reads to process per file (default: process all reads)")

args = parser.parse_args()

input_files = args.files
output_dir = args.output_dir
batch_size = args.batch_size
emb_out = args.emb_out
max_reads = args.max_reads
chunk_size = batch_size * 2

os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":

    verbose = True  # or use argparse to pass --verbose
    logger = setup_logger(verbose, log_file=os.path.join(output_dir, "inference.log"))
    
    logger.info(f'batch size = {batch_size//1024}K, chunk size = {chunk_size//1024}K')
    if slurm_cpus:
        slurm_cpus = int(slurm_cpus)
    else:
        slurm_cpus = 1  # Default to 1 if not running under SLURM
    num_cpus = mp.cpu_count()
    logger.info(f"CPUs allocated by SLURM: {slurm_cpus}, workers_num = {num_workers}")

    # Device selection with Mac MPS support
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        logger.info(f"Using CUDA GPU: {gpu_name}")
        use_half_precision = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS (Metal Performance Shaders)")
        use_half_precision = False  # MPS doesn't support float16 well
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (no GPU acceleration available)")
        use_half_precision = False  # CPU doesn't support float16 efficiently
    
    ## BBERT tokenizer and model
    tokenizer_path = bbert_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    collate_fn_instance = CollateFnWithTokenizer(tokenizer)
    
    bbert_model = BertForMaskedLM.from_pretrained(bbert_model_path, local_files_only=True)
    bbert_model.eval()
    if use_half_precision:
        bbert_model.half()
    bbert_model.to(device)
    logger.info(f"BBERT model loaded from {bbert_model_path}")

    bact_classifier = BertClassifier(hidden_size, bact_classes)
    bact_checkpoint = torch.load(bact_class_model_path, weights_only=True, map_location=device)
    bact_classifier.load_state_dict(bact_checkpoint['model_state_dict'])
    bact_classifier.eval()
    if use_half_precision:
        bact_classifier.half()
    bact_classifier.to(device)
    logger.info(f"Bacterial classifier model loaded from {bact_class_model_path}")
    
    ## frame classifier model
    frame_classifier = BertClassifier(hidden_size, frame_classes)
    frame_checkpoint = torch.load(frame_class_model_path, weights_only=True, map_location=device)
    frame_classifier.load_state_dict(frame_checkpoint['model_state_dict'])
    frame_classifier.eval()
    if use_half_precision:
        frame_classifier.half()
    frame_classifier.to(device)
    logger.info(f"Frame classifier model loaded from {frame_class_model_path}")
    
    ## coding classifier model
    coding_classifier = BertClassifier(hidden_size, coding_classes)
    coding_checkpoint = torch.load(class_model_path, weights_only=True, map_location=device)
    coding_classifier.load_state_dict(coding_checkpoint['model_state_dict'])
    coding_classifier.eval()
    if use_half_precision:
        coding_classifier.half()
    coding_classifier.to(device)
    logger.info(f"Coding classifier model loaded from {class_model_path}")
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    
    
    for file_path in input_files:
        ## dataset loading
        dataset_path = file_path
        
        # Extract base filename for output (without directory and extensions)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Handle .gz files (remove .gz and then the next extension)
        if base_name.endswith('.fasta') or base_name.endswith('.fastq'):
            base_name = os.path.splitext(base_name)[0]
        
        if emb_out:
            output_path = os.path.join(output_dir, f"{base_name}_scores_len_emb.parquet")
        else:
            output_path = os.path.join(output_dir, f"{base_name}_scores_len.parquet")
            
        logger.info(f"Processing file: {dataset_path}")
        dataset = FastqIterableDataset(dataset_path, chunk_size=chunk_size, max_reads=max_reads)
        
        seq_lens, data_len = dataset.get_stats()
        logger.info(f"{len(seq_lens)} reads, {min(seq_lens)} <= len <= {max(seq_lens)}")
        if max_reads and data_len >= max_reads:
            logger.info(f"Limited to first {data_len} reads (--max_reads {max_reads})")
        batches_num = data_len//batch_size
        
        # Configure DataLoader parameters based on platform
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
        
        ## reserve memory for results on GPU
        seq_ids = np.empty(data_len, dtype=object)  # To store sequence IDs
        bact_probs      = torch.zeros(data_len, dtype=torch.float32, device=device)
        # bact_pred_class = torch.zeros(data_len, dtype=torch.bool, device=device)
        loss_out        = torch.zeros(data_len, dtype=torch.float32, device=device)
        frame_probs     = torch.zeros(data_len, frame_classes, dtype=torch.float32, device=device)
        # frame_pred_class = torch.zeros(data_len, dtype=torch.int8, device=device)
        coding_probs      = torch.zeros(data_len, dtype=torch.float32, device=device)
        # coding_pred_class = torch.zeros(data_len, dtype=torch.bool, device=device)
        
        if emb_out:
            emb_array = np.empty((data_len, seq_len, hidden_size), dtype=np.float16)
        
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
                embeddings = outputs.hidden_states[-1]          # Last hidden layer
                bbert_logits = outputs.logits                   # [batch_size, seq_length, vocab_size]
                # logger.info(embeddings.shape)
                
                if emb_out:
                    # Get [CLS] token embedding or mean over tokens
                    batch_embs = embeddings.detach().cpu().numpy().astype(np.float16)
                    # logger.info(batch_embs.shape)
                    emb_array[batch_start:batch_start + batch_len, :, :] = batch_embs

                                
                bact_class_logits = bact_classifier(embeddings).to(device)
                bact_probs[batch_start:batch_start+batch_len] = F.softmax(bact_class_logits, dim=1)[:, 1]
                # bact_pred_class[batch_start:batch_start+batch_len] = (torch.argmax(bact_class_logits, dim=1) == 1)
                
                frame_logits = frame_classifier(embeddings).to(device)
                # frame_predicted_class = torch.argmax(frame_logits, dim=1)
                frame_probs[batch_start:batch_start+batch_len] = F.softmax(frame_logits, dim=1)
                # frame_pred_class[batch_start:batch_start+batch_len] = torch.argmax(frame_logits, dim=1)

                coding_class_logits = coding_classifier(embeddings).to(device)
                coding_probs[batch_start:batch_start+batch_len] = F.softmax(coding_class_logits, dim=1)[:, 1]
                # coding_pred_class[batch_start:batch_start+batch_len] = (torch.argmax(coding_class_logits, dim=1) == 1)
                            
                # Mean loss per sample (still on CUDA)
                # Flatten logits and targets
                logits_flat = bbert_logits.view(-1, bbert_logits.size(-1))     # [batch_size * seq_len, vocab_size]
                targets_flat = input_ids.view(-1)                              # [batch_size * seq_len]
                # Compute per-token loss
                token_losses = loss_fn(logits_flat, targets_flat)              # [batch_size * seq_len]
                # Reshape back to [batch_size, seq_len]
                token_losses = token_losses.view(bbert_logits.size(0), -1)     # [batch_size, seq_len]
                # Now compute mean loss for each read
                loss_per_read = (token_losses * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                loss_out[batch_start:batch_start+batch_len] = loss_per_read
                
                iter_reads += batch_len

                if (batch_id + 1) % log_interval == 0 or batch_id + 1 == batch_num:
                    iter_time = time.time() - iter_start_time
                    mean_speed = iter_reads / iter_time
                    msg = f"[{batch_id + 1}/{batch_num}] Mean speed: {mean_speed:.2f} reads/sec over {log_interval} batches"
                    resources_msg = get_resources_msg()
                    logger.info(f'{msg} \t{resources_msg}')
                    iter_reads = 0
                    iter_start_time = time.time() 

                batch_start += batch_len
                
        
        speed = data_len / (time.time() - inference_start_time)
        logger.info(f"mean speed: {speed:.2f} reads/sec")

        if emb_out:
            # Convert [data_len, seq_len, hidden_size] → [data_len * seq_len, hidden_size]
            flat_emb_array = emb_array.reshape(-1, hidden_size)

            # First level: fixed-size lists of hidden_size (vectors)
            flat_pa = pa.FixedSizeListArray.from_arrays(pa.array(flat_emb_array.ravel(), type=pa.float16()),list_size=hidden_size)

            # Second level: fixed-size lists of length seq_len (token sequences)
            emb_pa = pa.FixedSizeListArray.from_arrays(pa.array(flat_pa), list_size=seq_len)

            emb_shape = (len(emb_pa), emb_pa.type.list_size, emb_pa.type.value_type.list_size)
            logger.info(f"Embeddings Arrow shape: {emb_shape}")
        
        frame_array = frame_probs.cpu().numpy().astype(np.float32)  # No copy on GPU, just moves to CPU
        frame_pa = pa.FixedSizeListArray.from_arrays(pa.array(frame_array.ravel(), type=pa.float32()), list_size=frame_classes)

        # define the cols to write to output parquet
        columns = {
            "id": pa.array(seq_ids, type=pa.string()),
            "len": pa.array(seq_lens, type=pa.uint16()),  # store lengths
            "loss": pa.array(loss_out.cpu().numpy()),
            "bact_prob": pa.array(bact_probs.cpu().numpy()),
            "frame_prob": frame_pa,
            "coding_prob": pa.array(coding_probs.cpu().numpy()),
        }
        if emb_out:
            columns["embedding"] = emb_pa
        output_table = pa.table(columns)
        
        pq.write_table(output_table, output_path, compression="zstd")
        logger.info(output_table.slice(0, 5).to_pandas())
    
        if os.path.exists(output_path):
            logger.info(f"Results saved to {output_path}")
        else:
            logger.error(f"Failed to save results to {output_path}")
                  
        # Explicit deletion
        del dataset, dataloader, input_ids, outputs, embeddings
        del bbert_logits, bact_class_logits, frame_logits, coding_class_logits
        del bact_probs, frame_probs, coding_probs, loss_out, token_losses
        # del bact_pred_class, coding_pred_class, frame_pred_class
        del loss_per_read, seq_ids
        dataloader = None
        dataset = None

        # Clean memory
        torch.cuda.empty_cache()
        gc.collect()
        
clear_GPU()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()
sys.exit(0)
    