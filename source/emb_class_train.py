import os
import sys
import multiprocessing as mp
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BERT_model.dataset import FastqIterableDataset
from BERT_model.utils import get_true_label, log_resources, clear_GPU, setup_logger
from BERT_model.collator import CollateFnWithTokenizer
from emb_model.architecture import BertClassifier

os.environ["WANDB_DISABLED"] = "true"
slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

verbose = True  # or use argparse to pass --verbose
logger = setup_logger(verbose)

if slurm_cpus:
    slurm_cpus = int(slurm_cpus)
else:
    slurm_cpus = 1  # Default to 1 if not running under SLURM
logger.info(f"CPUs allocated by SLURM: {slurm_cpus}")

model_path = 'models/diverse_bact_12_768_6_20000/checkpoint-32500'
dataset_path = 'D:/data/frame_bact_train_dataset/train_frame_5000K_R1.fasta'

batch_size = 2048
chunk_size = batch_size * 20
max_length = 102
data_len = batch_size * 1000         # Set to None to read the entire dataset
num_workers = 0
prefetch_factor = None
epochs_num = 37

seq_length = 102
hidden_size = 768
num_classes = 6

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
                        