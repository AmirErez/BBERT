# BERT-DNA-classification
## 1. Environment.  
First, download this repository,
```bash
git clone https://github.com/AmirErez/BBERT.git
```
### 1.1.  Create BBERT environment from .yml file:
```bash
conda env create -f BBERT_env.yml --yes  
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
### 1.2.  Activate env and check the installation using script like that (sbatch):  

```bash  
   #!/bin/bash  
   #SBATCH --output=BBERT_venv_ckeck_%A.txt  
   #SBATCH --gres=gpu:1  
   #SBATCH --time=01:00:00  
   #SBATCH --mem=8G  
   echo "Checking CUDA availability for PyTorch and TensorFlow..."  
   python3 - <<END  
   import torch  
   print("PyTorch CUDA available:", torch.cuda.is_available())  
   END  
```

## 2. Training.  
Script:  `/source/train.py`  
To start the training process:  
- If training the model from scratch, set all base parameters.  
- If loading the model from a checkpoint, also specify the model name, batch size, and number of epochs (if needed).  

## 3. Inference.
 `inference.py` — Scoring Script for Sequence Files

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
  - Optional: sequence embeddings (`--emb_out`)
- Uses **PyArrow** for efficient storage and compression (`zstd`)
- Supports GPU acceleration and multi-core CPUs (SLURM-friendly)

### Usage
```bash
python inference.py \
    --input_dir /path/to/input \
    --input_files sample1.fasta sample2.fq.gz \
    --output_dir /path/to/output \
    --batch_size 1024 \
    --emb_out
```

## 4. Labeling scores.  
Script:  `/source/label_scores_R1_R2.py <R1.fasta> <R2.fasta> <labels.csv>`
Output  .csv file:  
- `base_id`   - read id (without /1 and /2 suffix)  
- `bact`      - true bacteria label  
- `score`     - mean score for two reads from R1.fasta and R2.fasta  

## 5. Cut point calculation.  
Script:  `/source/cut_point_calc_mult.py`  
Calculating the cut points and accuracy for a set of labeled scores.  
Plotting the results.  

## 6. Benchmarks.  
Script:  `/source/bertax_comparison.py`  
Comparison of classification performance between BBERT and BERTax on a set of testing datasets.  

## 7. Test datasets preparation.
### 7.1. Downloading and preprocessing
Script: `/source/ncbi-fna-iss-fastq-fasta.py`  
Ncbi -> .fna files -> iss pricessing -> fastq files -> conversion to .fasta:  
- obtaining a list of relevant bacterial and eukaryotic .fna files from NCBI.  
- filtering out .fna wich genus intersects with Bertax trining datasets.  
- downloading zip -> extracting .fna  
- using 'iss generate' tool to generate .fastq files  
- converting .fastq to .fasta and trimming reads to 100 bases
  
### 7.2. Datasets generation
Script:  `/source/gen_datasets_R1_R2.py`
Generation of 20 datasets, each containing 50 bact and 50 euk samples from generated .fasta files, with a 50/50 bact/euk ratio and a lognormal distribution.  
