import os
import subprocess
import tensorflow as tf
import time
import logging
import sys

DATASET_ID=sys.argv[1]
source_dir = '/sci/home/alekhin_dm_81/data/soil/test_datasets_3'

ds_len_label = '300K'
input_dir = os.path.join(source_dir, f'datasets_{ds_len_label}_R1_R2/datasets_{ds_len_label}_R1_R2')
output_dir = os.path.join(source_dir, f'bertax_{ds_len_label}_R1_R2')
os.makedirs(output_dir, exist_ok=True)
os.environ["WANDB_DISABLED"] = "true"

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        # logging.FileHandler("output.log"),  # Log to file
        logging.StreamHandler(sys.stdout)   # Log to console
    ]
)

# Example usage
logging.info(tf.sysconfig.get_build_info()["cuda_version"])
physical_devices = tf.config.list_physical_devices('GPU')
logging.info(f'Num GPUs Available: {len(physical_devices)}')

if physical_devices:
    logging.info("TensorFlow is using the GPU")
else:
    logging.info("TensorFlow is NOT using the GPU")


for R_idx in ['_R1', '_R2']:
    
    input_filepath = os.path.join(input_dir, f'ds_{DATASET_ID}{R_idx}.fasta')
    output_filepath = os.path.join(output_dir, f'ds_{DATASET_ID}{R_idx}_bertax.txt')
    
    logging.info(f'{DATASET_ID}\t{input_filepath} -> {output_filepath}')
    start_time = time.time()
    logging.info(start_time)

    subprocess.run(['bertax', '--batch_size', '1024',
                    input_filepath,
                    '-o', output_filepath,
                    '-s', '100',
                    '--output_ranks', 'superkingdom',
                    '--no_confidence'])

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")
