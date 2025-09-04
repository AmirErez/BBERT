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
**For Linux/Windows users:**
```bash
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
```

**For Mac users:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False); device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'); print('BBERT will use:', device)"
```

#### Test with example data:
```bash
python source/inference.py --input_dir example --input_files example.fasta --output_dir ./ --batch_size 64 
```

**Expected output on Mac:**
- `Using Apple MPS (Metal Performance Shaders)` - if you have Apple Silicon
- `Using CPU (no GPU acceleration available)` - Uses CPU instead

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

## 4. Post-Processing BBERT Outputs

BBERT inference produces Parquet files with classification scores. Depending on your data type, use the appropriate post-processing script to convert to a consistent TSV format.

### Single-End Data Processing

For single-end sequencing data, convert Parquet to TSV format:

```bash
python source/convert_scores_to_tsv.py \
    --input example_scores_len.parquet \
    --output_dir ./ \
    --output_prefix example
```

**Output:**
- `example_good_long_scores.tsv.gz` - Reads ≥100bp with scores
- `example_good_short_scores.tsv.gz` - Reads <100bp (excluded from analysis)

### Paired-End Data Processing

For paired-end sequencing data (R1/R2 files), merge scores from both reads:

```bash
python source/merge_paired_scores.py \
    --r1 /path/to/SRR8100008-good_1_scores_len.parquet \
    --r2 /path/to/SRR8100008-good_2_scores_len.parquet \
    --output_dir /path/to/output \
    --output_prefix SRR8100008
```

**Output:**
- `SRR8100008_good_long_scores.tsv.gz` - Combined scores for read pairs ≥100bp
- `SRR8100008_good_short_scores.tsv.gz` - Filtered short read pairs

**Score combination logic:**
- Both R1,R2 ≥100bp: Average their `loss` and `bact_prob`
- Only one read ≥100bp: Use that read's scores  
- Both reads <100bp: Exclude from long scores file

### Batch Processing with SLURM

For processing many files, use SLURM array jobs. Choose the appropriate batch script:

**Single-end data conversion:**
```bash
# Create accessions.csv with one identifier per line
echo -e "SRR8100008\nSRR8100009\nSRR8100010" > accessions.csv

# Submit batch job for single-end conversion
sbatch --array=1-100 scripts/batch_convert_scores.sh accessions.csv /path/to/scores
```

**Paired-end data merging:**
```bash
# Submit batch job for paired-end processing  
sbatch --array=1-100 scripts/batch_merge_scores.sh accessions.csv /path/to/scores
```

**Monitor jobs:**
```bash
squeue -u $USER
tail -f logs/convert_scores_JOBID_TASKID.log
```

### Final Output Format

Both post-processing scripts produce consistent TSV.GZ files:

**Long scores file** (`*_good_long_scores.tsv.gz`):
| Column | Description |
|--------|-------------|
| `id` | Sequence identifier |
| `loss` | Cross-entropy loss value |
| `bact_prob` | Bacterial classification probability (0-1) |

**Short scores file** (`*_good_short_scores.tsv.gz`):
Contains metadata for reads/pairs excluded due to length filtering.

## 5. Testing BBERT Accuracy

Test BBERT's classification performance using known ground truth sequences:

```bash
# Run accuracy tests
python tests/test_inference_accuracy.py
```

**Test data:** `example/example.fasta` contains 10 sequences:
- Sequences 1-5: *E. coli* K-12 (should classify as bacterial, bact_prob > 0.5)
- Sequences 6-10: *Saccharomyces cerevisiae* (should classify as non-bacterial, bact_prob < 0.5)

**Expected results:**
- **Perfect classification**: All 10 sequences correctly classified using 0.5 threshold
- *E. coli* mean bacterial probability > 0.5
- *S. cerevisiae* mean bacterial probability < 0.5

**What the test validates:**
- ✅ Model predictions are accurate for known organisms
- ✅ Probabilities are in valid ranges [0,1]  
- ✅ Output format is consistent
- ✅ All sequences processed correctly

The test provides detailed output showing individual sequence predictions for model validation.
