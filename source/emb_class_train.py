import os
import sys
import multiprocessing as mp
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BERT_model.dataset import FastqIterableDataset
from BERT_model.utils import get_true_label, log_resources, clear_GPU, setup_logger, get_slurm_cpus
from BERT_model.collator import CollateFnWithTokenizer
from BERT_model.config import SEQUENCE_LENGTH, HIDDEN_SIZE_768, NUM_READING_FRAMES
from emb_model.architecture import BertClassifier

os.environ["WANDB_DISABLED"] = "true"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train reading frame classifier on BBERT embeddings')
parser.add_argument('--dataset', type=str, required=True,
                    help='Path to training dataset (FASTA/FASTQ file)')
parser.add_argument('--model_path', type=str, default='models/diverse_bact_12_768_6_20000/checkpoint-32500',
                    help='Path to pretrained BBERT model (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='Batch size for training (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=37,
                    help='Number of training epochs (default: %(default)s)')
parser.add_argument('--max_reads', type=int, default=None,
                    help='Maximum number of reads to use (default: all reads)')
parser.add_argument('--output_model', type=str, default=None,
                    help='Path to save trained model (default: auto-generated based on dataset size)')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='Enable verbose logging (default: %(default)s)')
args = parser.parse_args()

logger = setup_logger(args.verbose)

slurm_cpus = get_slurm_cpus()
logger.info(f"CPUs allocated by SLURM: {slurm_cpus}")

# Use argument values
model_path = args.model_path
dataset_path = args.dataset
batch_size = args.batch_size
epochs_num = args.epochs

chunk_size = batch_size * 20
max_length = SEQUENCE_LENGTH
data_len = args.max_reads if args.max_reads else batch_size * 1000
num_workers = 0
prefetch_factor = None

seq_length = SEQUENCE_LENGTH
hidden_size = HIDDEN_SIZE_768
num_classes = NUM_READING_FRAMES

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_cpus = mp.cpu_count()

    model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True).eval().to(device)
    # model.half()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    collate_fn_instance = CollateFnWithTokenizer(tokenizer)

    dataset = FastqIterableDataset(dataset_path, chunk_size=chunk_size, max_reads=data_len)
    data_len = len(dataset)
    batches_num = data_len//batch_size
    logger.info(f"Dataset length: {data_len} reads - {batches_num} batches of {batch_size} reads")

    # Use provided output path or auto-generate one
    if args.output_model:
        class_model_path = args.output_model
    else:
        class_model_path = f'cnn_emb_model/models/classifier_model_{data_len//1024}K_{epochs_num}e.pth'
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_instance,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    classifier = BertClassifier(hidden_size=hidden_size, num_classes=num_classes).to(device)  # Move to GPU if needed
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)  # Adjust the learning rate as needed
    loss_fn = nn.CrossEntropyLoss().to(device) # Move to GPU if needed
    
    for epoch in range(epochs_num):
        iter_start_time = time.time()
        total_loss = 0
        clear_GPU()

        for batch_id, batch in enumerate(dataloader):
            
            input_ids = batch[1].to(device)
            # Prepare classification labels
            class_labels = torch.tensor([get_true_label(d) for d in batch[0]], dtype=torch.long, device=device)

            with torch.no_grad():
                BBERT_outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
                embeddings = BBERT_outputs.hidden_states[-1]  # [batch, 102, 768]
            
            logits = classifier(embeddings)
            train_loss = loss_fn(logits, class_labels)

            optimizer.zero_grad()  # Clear previous gradients
            train_loss.backward()  # Compute gradients
            optimizer.step()  # Update the model parameters
            total_loss += train_loss.item()  # Accumulate loss for this batch
            print(f"\rBatch: {batch_id+1}/{batches_num}", end='', flush=True)
            
        iter_time = time.time() - iter_start_time
        msg = f'epoch={epoch+1}/{epochs_num} | loss={total_loss/batches_num:.4f} | {(time.time() - iter_start_time):.2f}sec/it'
        log_resources(msg)

    torch.save({
        'epoch': epoch,  # Store the current epoch number
        'model_state_dict': classifier.state_dict(),  # The model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state (useful for resuming training)
        'loss': train_loss.item(),  # Last loss value, if useful
    }, class_model_path)

print(f"Model saved to {class_model_path}")
                        