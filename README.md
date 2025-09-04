# BBERT: BERT for Bacterial DNA Classification

BBERT is a BERT-based transformer model fine-tuned for DNA sequence analysis, specifically designed for bacterial sequence classification and genomic feature prediction. The model performs three key classification tasks:

- **Bacterial Classification**: Distinguishes bacterial DNA from non-bacterial sequences
- **Reading Frame Prediction**: Identifies the correct reading frame (1 of 6 possible frames)
- **Coding Sequence Classification**: Determines whether sequences are protein-coding or non-coding

The model processes short DNA sequences (100bp or longer) and outputs classification probabilities along with sequence embeddings for downstream analysis.

## System Requirements

- **Python**: 3.10+
- **GPU**: CUDA-compatible GPU recommended (tested with CUDA 12.4)
- **Memory**: Minimum 8GB RAM, 4GB+ GPU memory recommended
- **Storage**: ~2GB for model files (requires Git LFS)
- **Dependencies**: PyTorch, Transformers, PyArrow, pandas, scikit-learn, seaborn

## 1. Installation.  
### 1.1. Download
#### Option 1: From github
First, download this repository,
```bash
git clone https://github.com/AmirErez/BBERT.git
cd BBERT
```
You will need github's large file storage feature to download the model
```bash
sudo apt-get install git-lfs  # This or a different way to install git-lfs
git lfs install
git lfs pull # Downloads the model
```
#### Option 2: From Zenodo
TBA

### 1.2.  Create BBERT environment from .yml file:
```bash
conda env create -f BBERT_env.yml  
```
or  
```bash
   conda create -n BBERT python=3.10  
   conda activate BBERT  
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  
   conda install -c conda-forge transformers=4.30.2  
   conda install seaborn  
   conda install scikit-learn  
```
### 1.3.  Activate env and check the installation 

#### Option 1: From the python command line
Run the following in python, in the BBERT environment:

```python
   import torch  
   print("PyTorch CUDA available:", torch.cuda.is_available()) 
```
then run on the example file
```bash
   python source/inference.py --input_dir ../example --input_files example.fasta --output_dir ../example --batch_size 1024 
```
The output will be in the file ../example/example_scores_len.parquet

You can read it using python pandas,
```python
   import pandas as pd
   df = pd.read_parquet('../example/example_scores_len.parquet')
```
#### Option 2: From a job manager
Here is an example how to execute the script on a gpu node in our SLURM setup.

```bash  
   #!/bin/bash  
   #SBATCH --output=BBERT_venv_check_%A.txt  
   #SBATCH --gres=gpu:1  
   #SBATCH --time=01:00:00  
   #SBATCH --mem=8G  
   echo "Checking CUDA availability for PyTorch and TensorFlow..."  
   python3 - <<END  
   import torch  
   print("PyTorch CUDA available:", torch.cuda.is_available())  
   END
   python source/inference.py --input_dir ../example --input_files example.fasta --output_dir ../example --batch_size 1024   
```
You may then examine the output in ../example/scores_len.parquet as described in Option 1 above.

## 2. Inference.
 `source/inference.py` — Scoring Script for Sequence Files

This script runs inference on DNA sequencing data using the BBERT model and multiple downstream classifiers (bacterial classification, frame prediction, coding classification).  
It processes FASTA/FASTQ/GZIP input files, computes probabilities, loss values, and optionally embeddings, and writes results to Parquet files for downstream analysis.

### Features
- Loads a pretrained BBERT model and three classification heads:
  - Bacterial classifier (bacteria vs. non-bacteria)
  - Frame classifier (6 possible reading frames)
  - Coding classifier (coding vs. non-coding DNA)
- Supports input formats: `.fasta`, `.fastq`, `.gz`
- Outputs results to `.parquet` with:
  - Sequence IDs
  - Sequence lengths
  - Cross-entropy loss per read
  - Predicted probabilities for each classifier
  - Optional: sequence embeddings (`--emb_out`).*Warning* Slow and takes up a lot of space.
- Uses **PyArrow** for efficient storage and compression (`zstd`)
- Supports GPU acceleration and multi-core CPUs (SLURM-friendly)

### Usage
```bash
python source/inference.py \
    --input_dir /path/to/input \
    --input_files sample1.fasta sample2.fq.gz \
    --output_dir /path/to/output \
    --batch_size 1024
```

### Arguments
- `--input_dir`: Directory where input files are located (required)
- `--input_files`: List of input filenames to process (required, can specify multiple files)
- `--output_dir`: Directory to save output Parquet files (required)
- `--batch_size`: Batch size for processing (default: 1024)
- `--emb_out`: Include sequence embeddings in output (optional, warning: slow and large files)

## 3. Labeling scores.  
Script:  `/source/label_scores_R1_R2.py <R1.fasta> <R2.fasta> <labels.csv>`
Output  .csv file:  
- `base_id`   - read id (without /1 and /2 suffix)  
- `bact`      - true bacteria label  
- `score`     - mean score for two reads from R1.fasta and R2.fasta  

## 4. Cut point calculation.  
Script:  `/source/cut_point_calc_mult.py`  
Calculating the cut points and accuracy for a set of labeled scores.  
Plotting the results.  

## 5. Benchmarks.  
Script:  `/source/bertax_comparison.py`  
Comparison of classification performance between BBERT and BERTax on a set of testing datasets.  

## 6. Test datasets preparation.
### 6.1. Downloading and preprocessing
Script: `/source/ncbi-fna-iss-fastq-fasta.py`  
NCBI -> .fna files -> ISS processing -> fastq files -> conversion to .fasta:  
- obtaining a list of relevant bacterial and eukaryotic .fna files from NCBI.  
- filtering out .fna which genus intersects with BERTax training datasets.  
- downloading zip -> extracting .fna  
- using 'iss generate' tool to generate .fastq files  
- converting .fastq to .fasta and trimming reads to 100 bases
  
### 6.2. Datasets generation
Script:  `/source/gen_datasets_R1_R2.py`
Generation of 20 datasets, each containing 50 bact and 50 euk samples from generated .fasta files, with a 50/50 bact/euk ratio and a lognormal distribution.  


## 7. Training.  
Script:  `/source/train.py`  
To start the training process:  
- If training the model from scratch, set all base parameters.  
- If loading the model from a checkpoint, also specify the model name, batch size, and number of epochs (if needed).  
