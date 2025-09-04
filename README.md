# BBERT: BERT for Bacterial DNA Classification

BBERT is a BERT-based transformer model fine-tuned for DNA sequence analysis, specifically designed for bacterial sequence classification and genomic feature prediction. The model performs three key classification tasks:

- **Bacterial Classification**: Distinguishes bacterial DNA from non-bacterial sequences
- **Reading Frame Prediction**: Identifies the correct reading frame (1 of 6 possible frames)
- **Coding Sequence Classification**: Determines whether sequences are protein-coding or non-coding

The model processes short DNA sequences (100bp or longer) and outputs classification probabilities along with sequence embeddings for downstream analysis.

## System Requirements

- **Python**: 3.10+
- **GPU**: 
  - CUDA-compatible GPU recommended (tested with CUDA 12.4)
  - Apple Silicon Macs: MPS acceleration supported
  - CPU-only: Supported but slower
- **Memory**: Minimum 8GB RAM, 4GB+ GPU memory recommended  
- **Storage**: ~2GB for model files (requires Git LFS)
- **Dependencies**: PyTorch, Transformers, PyArrow, pandas, scikit-learn, seaborn

### Mac Users
For Apple Silicon Macs, the model will automatically use MPS (Metal Performance Shaders) for acceleration. Install PyTorch with MPS support:
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

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

### 1.2.  Create BBERT environment:

#### For Linux/Windows with CUDA:
```bash
conda env create -f BBERT_env.yml  
```

#### For Mac (recommended):
```bash
conda env create -f BBERT_env_mac.yml
conda activate BBERT_mac
```

#### Manual installation for Mac/CPU-only systems:
```bash
conda create -n BBERT python=3.10  
conda activate BBERT  

# Core PyTorch (Mac with Apple Silicon gets MPS acceleration automatically)
conda install pytorch torchvision torchaudio -c pytorch

# Core dependencies
conda install -c conda-forge transformers=4.30.2 pyarrow pandas scikit-learn seaborn tqdm pyyaml
conda install biopython psutil
conda install "numpy<2"  # Fix compatibility issues

# Additional packages
pip install datasets huggingface_hub safetensors tokenizers torchinfo pynvml
```
### 1.3.  Test installation and GPU support

#### GPU acceleration test:
**For Mac users:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False); device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'); print('BBERT will use:', device)"
```

**For Linux/Windows users:**
```bash
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
```

#### Test with example data:
```bash
python source/inference.py --input_dir example --input_files example.fasta --output_dir ./ --batch_size 64 
```

**Expected output on Mac:**
- `Using Apple MPS (Metal Performance Shaders)` - if you have Apple Silicon
- `Using CPU (no GPU acceleration available)` - if you have Intel Mac

The output will be in `example_scores_len.parquet`. View results:
```python
import pandas as pd
df = pd.read_parquet('example_scores_len.parquet')
print(df.head())
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

## 3. Output Format

The inference script outputs results to a Parquet file containing:

| Column | Description |
|--------|-------------|
| `id` | Sequence identifier |
| `len` | Sequence length |
| `loss` | Cross-entropy loss value |
| `bact_prob` | Bacterial classification probability (0-1) |
| `frame_prob` | Reading frame probabilities (array of 6 values for frames +1,+2,+3,-1,-2,-3) |
| `coding_prob` | Coding sequence probability (0-1) |

### Reading Results
```python
import pandas as pd
df = pd.read_parquet('example_scores_len.parquet')
print(df.head())

# Get sequences predicted as bacterial (>50% probability)
bacterial_seqs = df[df['bact_prob'] > 0.5]

# Get most likely reading frame for each sequence
import numpy as np
df['predicted_frame'] = df['frame_prob'].apply(lambda x: np.argmax(x))
```
