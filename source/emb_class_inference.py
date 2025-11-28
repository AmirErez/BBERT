import os
import sys
import multiprocessing as mp
import subprocess
import time
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BERT_model.dataset import FastqIterableDataset
from BERT_model.utils import get_true_label, log_resources, clear_GPU, setup_logger, label_to_frame, get_slurm_cpus
from BERT_model.collator import CollateFnWithTokenizer
from BERT_model.config import HIDDEN_SIZE_768, NUM_READING_FRAMES
from emb_model.architecture import BertClassifier

os.environ["WANDB_DISABLED"] = "true"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run inference with reading frame classifier on BBERT embeddings')
parser.add_argument('--dataset', type=str, required=True,
                    help='Path to input dataset (FASTA/FASTQ file)')
parser.add_argument('--model_path', type=str, default='models/diverse_bact_12_768_6_20000/checkpoint-32500',
                    help='Path to pretrained BBERT model (default: %(default)s)')
parser.add_argument('--classifier_path', type=str, default='cnn_emb_model/models/classifier_model_2000K_37e.pth',
                    help='Path to trained classifier model (default: %(default)s)')
parser.add_argument('--output', type=str, default=None,
                    help='Path to save results CSV (default: input filename with _frames.csv suffix)')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='Batch size for inference (default: %(default)s)')
parser.add_argument('--max_reads', type=int, default=None,
                    help='Maximum number of reads to process (default: all reads)')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='Enable verbose logging (default: %(default)s)')
args = parser.parse_args()

logger = setup_logger(args.verbose)

slurm_cpus = get_slurm_cpus()
logger.info(f"CPUs allocated by SLURM: {slurm_cpus}")

# Use argument values
BBERT_model_path = args.model_path
class_model_path = args.classifier_path
dataset_path = args.dataset

# Determine output path
if args.output:
    output_filepath = args.output
else:
    # Auto-generate output path from input filename
    base_name = os.path.basename(dataset_path)
    output_dir = os.path.dirname(dataset_path)
    output_filename = base_name.replace('.fasta', '_frames.csv').replace('.fastq', '_frames.csv')
    output_filepath = os.path.join(output_dir, output_filename)

batch_size = args.batch_size
chunk_size = batch_size * 50
data_len = args.max_reads
num_workers = 0
prefetch_factor = None

hidden_size = HIDDEN_SIZE_768
num_classes = NUM_READING_FRAMES

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_cpus = mp.cpu_count()

    model = BertForMaskedLM.from_pretrained(BBERT_model_path, local_files_only=True).eval().to(device)
    # model.half()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer_path = BBERT_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    collate_fn_instance = CollateFnWithTokenizer(tokenizer)

    dataset = FastqIterableDataset(dataset_path, chunk_size=chunk_size, max_reads=data_len)
    data_len = len(dataset)
    logger.info(f"Dataset length: {data_len}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_instance,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    classifier = BertClassifier(hidden_size, num_classes)
    checkpoint = torch.load(class_model_path, weights_only=True, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device).eval()
    
    result = pd.DataFrame(index=range(data_len), columns=['id', 'pred_frame', 'true_frame'])
    batch_num = data_len // batch_size + 1 if data_len % batch_size != 0 else data_len // batch_size
    batch_start = 0

    for batch_id, batch in enumerate(dataloader):
        
        iter_start_time = time.time()

        batch_len = len(batch[0])
        input_ids = batch[1].to(device)
        true_labels = torch.tensor([get_true_label(d) for d in batch[0]], dtype=torch.long, device=device)

        with torch.no_grad():
            BBERT_outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
            embeddings = BBERT_outputs.hidden_states[-1]  # [batch, 102, 768]
        
        logits = classifier(embeddings)
        predicted_class = torch.argmax(logits, dim=1)
       
        pred_frame = [label_to_frame(c.item()) for c in predicted_class]
        true_frame = [label_to_frame(c.item()) for c in true_labels]

        print(sum(p==t for p, t in zip(pred_frame, true_frame)) / batch_len)
        result.loc[batch_start:batch_start + batch_len - 1] = list(zip(batch[0], pred_frame, true_frame))
        batch_start += batch_len
        iter_end_time = time.time()
        dt = iter_end_time - iter_start_time
        # logger.info(f"batch {batch_id + 1}/{batch_num}, {batch_len} reads, {iter_end_time - iter_start_time:.2f} sec")
        msg = f"batch {batch_id + 1}/{batch_num}, {batch_len} reads, {dt:.2f}sec, {batch_len/dt:.2f} reads/sec"
        log_resources(msg)

    # abs_rate = result[result['pred_frame'].apply(abs) == result['true_frame'].apply(abs)].shape[0] / (batch_start + 1)
    rate = result[result['pred_frame'] == result['true_frame']].shape[0] / (batch_start + 1)
    
    print(rate)
    with pd.option_context('display.max_colwidth', None):
        print(result)
    
    result.to_csv(output_filepath, index=False)
    