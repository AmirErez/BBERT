

import os
# os.environ["WANDB_DISABLED"] = "false"
import torch
import numpy as np
from transformers import BertConfig, BertForMaskedLM, TrainerCallback
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
import argparse
from torchinfo import summary
# from pytorch_lamb import Lamb  # Import the LAMB optimizer
import sys
import logging

att_heads = 6 		# if not resuming from checkpoint
hidden_size = 384 	# if not resuming from checkpoint
hidden_layers = 12 	# if not resuming from checkpoint 
batch_size = 2048	# set even if loading from checkpoont
train_len = 50_000_000
eval_factor = 0.3
num_train_epochs = 2
 
model_name = f'diverse_bact_{att_heads}_{hidden_size}_{hidden_layers}_{train_len//1000}Ks'

project_dir = '~/projects/BBERT'
data_dir = '~/projects/BBERT/data/diverse_bacteria_R1_R2_100Ms'

parser = argparse.ArgumentParser(description='BERT for Bacteria OOD sequence detection')
parser.add_argument('--dataset_path', type=str, default=f'{data_dir}')
parser.add_argument('--save_model_path', type=str, default=f'{project_dir}/models/{model_name}')
parser.add_argument('--tokenizer_path', type=str, default=f'{project_dir}/tokenizers/base_tokenizer') #if not resuming from checkpoint
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,                 # Set logging level to INFO or DEBUG
    format='%(asctime)s - %(message)s', # Customize format if needed   
    handlers=[logging.StreamHandler()]  # Output to console
)

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of GPUs
    num_devices = torch.cuda.device_count()
    logging.info(f"Number of CUDA devices: {num_devices}")
    
    # Print details for each device
    for i in range(num_devices):
        logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        logging.info(f"Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
        logging.info(f"Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
else:
    print("CUDA is not available.")

last_checkpoint = None
if os.path.exists(args.save_model_path):
    last_checkpoint = get_last_checkpoint(args.save_model_path)

if last_checkpoint:
    logging.info(f'Found last checkpoint: {last_checkpoint}')
else:
    logging.info('Checkpoint not found')

class BaseSeqCollator(object):
    def __init__(self, tokenizer, mask_perc=0.3):
        self.tokenizer = tokenizer
        self.mask_perc = mask_perc

    def __call__(self, features):
        seqs = [f['seq'] for f in features]
        batch = self.tokenizer(seqs, padding='max_length', truncation=True, max_length=102, return_tensors='pt')
        batch['labels'] = torch.ones(batch['input_ids'].shape, dtype=torch.int8)*(-100)
        mask = torch.tensor(np.random.binomial(1, self.mask_perc, batch['input_ids'].shape), dtype=torch.int64)
        mask = mask*batch['attention_mask']
        mask = mask==1
        batch['labels'] = torch.where(mask, batch['input_ids'], -100)
        batch['input_ids'] = torch.where(~mask, batch['input_ids'], self.tokenizer.convert_tokens_to_ids('<msk>'))
        return batch

class MemoryUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Only log if CUDA is available (GPU training)
        if torch.cuda.is_available():
            # sys.stdout.flush()  # Flush the output to force it to display immediately
            
            allocated_memory = torch.cuda.memory_allocated() / 1024**2  # In MB
            reserved_memory = torch.cuda.memory_reserved() / 1024**2  # In MB
            
            # Print on the same line by using '\r' and flush the output
            sys.stdout.write(f"\rStep {state.global_step}/{state.max_steps} | "
                             f"Allocated Memory: {allocated_memory:.2f} MB | "
                             f"Reserved Memory: {reserved_memory:.2f} MB")

logging.info(f'Dataset: {args.dataset_path}')
logging.info(f'Model: {args.save_model_path}')

dataset = load_from_disk(args.dataset_path)
dataset.set_format(type='torch')
logging.info(dataset)
logging.info(dataset['train'].select(range(100))['ids'])

tokenizer = AutoTokenizer.from_pretrained(last_checkpoint if last_checkpoint else args.tokenizer_path)
logging.info(tokenizer)

bert_config = BertConfig(vocab_size=len(tokenizer), hidden_size=hidden_size, num_attention_heads=att_heads, num_hidden_layers=hidden_layers, intermediate_size=1536)
# model = BertForMaskedLM(bert_config).cuda()
# model = BertForMaskedLM.from_pretrained(args.load_model_path, local_files_only=True).cuda()
# model = BertForMaskedLM(bert_config).from_pretrained(args.load_model_path, config=bert_config, local_files_only=True).cuda()
# model = BertForMaskedLM(bert_config).from_pretrained(args.load_model_path, local_files_only=True).cuda()

if last_checkpoint:
    logging.info(f"Resuming from checkpoint: {last_checkpoint}")
    model = BertForMaskedLM.from_pretrained(last_checkpoint).cuda()
else:
    logging.info("No checkpoint found, starting from scratch.")
    model = BertForMaskedLM(bert_config).cuda()
    
logging.info(summary(model))
logging.info(model.config)
logging.info("Model - Num of parameters: ", model.num_parameters())

if last_checkpoint:
    logging.info(f"Loading Training_Args from checkpoint: {last_checkpoint}")
    training_args = torch.load(os.path.join(last_checkpoint, 'training_args.bin'))
    # training_args.resume_from_checkpoint = True
    # training_args.num_train_epochs=num_train_epochs
    training_args.per_device_train_batch_size=batch_size
    training_args.per_device_eval_batch_size=batch_size

else:
    logging.info(f"Loading Training_Args from scratch:")
    training_args = TrainingArguments(
        report_to=None,
        output_dir=f"{args.save_model_path}",
        warmup_steps=2000,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        weight_decay=0.01,
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=100,
        num_train_epochs=num_train_epochs,
        label_smoothing_factor=0.0,
        load_best_model_at_end=True,
        save_steps=200,
        save_strategy='steps',
        save_total_limit=100,
        logging_first_step=True,
        disable_tqdm=False,
        remove_unused_columns=False,
    )
logging.info(training_args)

#optimizer = Lamb(model.parameters(), lr=5e-4, weight_decay=0.01) # Set learning rate and weight decay

train_dataset=dataset['train'].shuffle(seed=56).select(range(train_len))
eval_dataset=dataset['val'].shuffle(seed=56).select(range(int(train_len * eval_factor)))

# train_dataset=dataset['train'].shuffle(seed=56)
# eval_dataset=dataset['val'].shuffle(seed=56)

logging.info(f'Train dataset:\n {train_dataset}')
logging.info(f'Eval dataset:\n {eval_dataset}')

trainer = Trainer(  model=model,
                    args=training_args,
                    train_dataset=train_dataset,
		            eval_dataset=eval_dataset,
                    data_collator=BaseSeqCollator(tokenizer),
		            tokenizer=tokenizer,
                #   callbacks=[MemoryUsageCallback()]
			    #   optimizers=(optimizer, None),  # (optimizer, scheduler)
                #   compute_metrics=metrics,
)

if last_checkpoint:
    logging.info(f"Resuming from checkpoint {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    logging.info("Starting training from scratch")
    trainer.train()

trainer.save_model(os.path.join(args.save_model_path, 'best_model'))
