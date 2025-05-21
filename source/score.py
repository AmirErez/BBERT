import os
import argparse

import torch
from torch.utils.data import DataLoader, IterableDataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertForMaskedLM

import multiprocessing as mp
import time
import logging
import gc
import re
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

from BERT_model.dataset import FastqIterableDataset
from BERT_model.utils import get_true_label, log_resources, clear_GPU, setup_logger
from BERT_model.collator import CollateFnWithTokenizer
from emb_model.architecture import BertClassifier

nvmlInit()
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger()
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.DEBUG,                    # Set logging level to INFO or DEBUG
        format='%(asctime)s - %(message)s',     # Customize format
        handlers=[logging.StreamHandler()],     # Output to console
        datefmt='%Y-%m-%d %H:%M:%S'             # Custom date format (excluding milliseconds)
    )

slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus:
    slurm_cpus = int(slurm_cpus)
else:
    slurm_cpus = 1  # Default to 1 if not running under SLURM

slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

parser = argparse.ArgumentParser(description="Run scoring on a FASTA, FASTQ or GZIP file and save the output.")
parser.add_argument("model_path", type=str, help="Path to the model")
parser.add_argument("file_path", type=str, help="Path to the FASTA, FASTQ or GZIP input file")
parser.add_argument("scores_filename", type=str, help="Path to save the score output CSV file")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing (default: 1024)")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (debug level).")
    
args = parser.parse_args()
model_path = args.model_path
file_path = args.file_path
scores_filename = args.scores_filename
batch_size = args.batch_size

chunk_size = 10 * batch_size
max_length = 102
data_len = None
num_workers = 1
prefetch_factor = 2

if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)  # Verbose logging enabled
else:
    logging.getLogger().setLevel(logging.INFO)  # Default logging

os.makedirs(os.path.dirname(scores_filename), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_cpus = mp.cpu_count()
model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True).eval().half().to(device)

tokenizer_path = model_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

loss = torch.nn.CrossEntropyLoss()

def collate_fn(batch):
    seq = [re.sub('[^ACTGN]', 'N', r['seq'].upper()) for r in batch]
    seq_len = [len(r['seq']) for r in batch]
    id = [r['id'] for r in batch]
    encoded_seq = tokenizer(seq, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')['input_ids']
    return id, encoded_seq, seq_len

if __name__ == "__main__":
    
    # Set logging configuration based on verbosity flag
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,  # Debug level enables both debug and info logs
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,  # Info level will suppress debug logs
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    logging.info(f"SLURM CPUs: \t{slurm_cpus}")
    logging.info(f"Model path: \t{model_path}")
    logging.info(f"Input file: \t{file_path}")
    logging.info(f"Score output:\t{scores_filename}")
    logging.info(f"Batch size: \t{batch_size}")
    logging.debug(model.config)
    
    scores_dir = os.path.dirname(scores_filename)

    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    dataset = FastqIterableDataset(file_path, chunk_size=chunk_size, max_reads=data_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    scores_list = []
    speed = []
    iter = 1
        
    with torch.no_grad():
        
        for batch in dataloader:
            id = batch[0]
            input_ids = batch[1].to(device)
            seq_lens = batch[2]
                        
            iter_start_time = time.time()
            
            input_size = input_ids.shape[0]
            loss_input = model(input_ids, labels=input_ids, output_hidden_states=False).logits

            loss_out = [loss(loss_input[i], input_ids[i]).item() for i in range(input_size)]

            for i, score, read_len  in zip(id, loss_out, seq_lens):
                batch_scores = {'id': i, 'score': score, 'len': read_len}
                scores_list.append(batch_scores)
            
            iter_time = time.time() - iter_start_time
            speed.append(input_size / iter_time)
            
            logging.debug(f'{iter} - {input_size} - {speed[-1]}')
            log_resources()
            
            # del input_ids, loss_input, loss_out  # Delete unused variables
            # torch.cuda.empty_cache()
            # gc.collect()
            
            iter += 1
    
    del input_ids, loss_input, loss_out  # Delete unused variables
    torch.cuda.empty_cache()
    gc.collect()
    
    log_resources()
    logging.info(f'Mean perfomance: \t{np.mean(speed):.1f} reads/sec' )
    scores = pd.DataFrame(scores_list)
    logging.debug(scores)
    
    short_reads_num = sum(scores['len'] < 100)
    logging.info(f'Short reads: \t{short_reads_num}/{len(scores)}')
    logging.debug(scores.loc[scores['len']<100])
    
    scores.to_csv(scores_filename, index=False, float_format="%.6f")
    if os.path.exists(scores_filename) and os.path.getsize(file_path) > 0:
        logging.info(f"Scores CSV file \t'{scores_filename}' created successfully, {os.path.getsize(scores_filename)//(1024)} Kb")
    else:
        logging.info("Error: CSV file not found or empty.")

    