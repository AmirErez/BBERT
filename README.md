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

## 1. Installation.  
### 1.1. Download
#### Option 1: From github
First, download this repository,
```bash
git clone https://github.com/AmirErez/BBERT.git
cd BBERT
```

#### Installing Git Large File Storage
You will need GitHub's large file storage feature to download the model files.

**On Unix/Linux:**
```bash
sudo apt-get install git-lfs  # Ubuntu/Debian
# OR
sudo yum install git-lfs      # CentOS/RHEL/Fedora
# OR use your distribution's package manager
```

**On Mac:**
Download and install Git LFS from https://git-lfs.com or use a package manager:
```bash
brew install git-lfs         # Using Homebrew
# OR
sudo port install git-lfs    # Using MacPorts
```

**Initialize Git LFS (all platforms):**
```bash
git lfs install
git lfs pull # Downloads the model files
```
#### Option 2: From Zenodo
TBA

### 1.2. Install using Conda

**Prerequisites:** You need conda or mamba installed on your system:
- **Conda:** Download from https://conda.io/miniconda.html or https://www.anaconda.com/
- **Mamba:** Faster alternative, install with `conda install mamba -n base -c conda-forge`

#### For Linux/Windows with CUDA:
```bash
conda env create -f BBERT_env.yml  
# OR for faster installation:
mamba env create -f BBERT_env.yml
```

#### For Mac (recommended):
```bash
conda env create -f BBERT_env_mac.yml
conda activate BBERT_mac
# OR for faster installation:
mamba env create -f BBERT_env_mac.yml
mamba activate BBERT_mac
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
### 1.3. Verify Installation

#### Step 1: Run System Diagnostics (Required)
Immediately after installation, verify that everything is set up correctly:

**Unix/Linux/Mac:**
```bash
./bbert --check
```

**Windows:**
```cmd
bbert.bat --check
```

**Cross-platform (Python):**
```bash
python bbert.py --check
```

This will verify:
- ✅ Python 3.10+ installation
- ✅ Required packages (PyTorch, Transformers, BioPython, etc.)  
- ✅ Model files downloaded via Git LFS
- ✅ GPU availability (CUDA/MPS)
- ✅ Conda environment status

**Expected output on Mac:**
- `Using Apple MPS (Metal Performance Shaders)` - if you have Apple Silicon
- `Using CPU (no GPU acceleration available)` - Uses CPU instead

#### Step 2: Run Accuracy Tests (Recommended)
After system checks pass, validate BBERT's classification accuracy:

```bash
python source/test_inference_accuracy.py
```

This test uses known ground truth sequences:
- Sequences 1-5: *E. coli* K-12 (should classify as bacterial, bact_prob > 0.5)
- Sequences 6-10: *Saccharomyces cerevisiae* (should classify as non-bacterial, bact_prob < 0.5)

**Expected results:**
Perfect classification: All 10 sequences correctly classified

#### Step 3: Test with Example Data
Once tests pass, try processing example data:

**Unix/Linux/Mac:**
```bash
./bbert example/example.fasta --output_dir example/ --batch_size 64
```

**Windows:**
```cmd
bbert.bat example\example.fasta --output_dir example/ --batch_size 64
```

**Cross-platform (Python):**
```bash
python bbert.py example/example.fasta --output_dir example/ --batch_size 64
```

The output will be in `example_scores_len.parquet`. View results in the python console:
```python
import pandas as pd
df = pd.read_parquet('example/example_scores_len.parquet')
print(df.head())
```


## 2. Running BBERT

### 2.1. Using the Convenient Executables (Recommended)

BBERT provides user-friendly executables that automatically check your system and provide helpful error messages:

**Unix/Linux/Mac:** `./bbert`
**Windows:** `bbert.bat`  
**Cross-platform:** `python bbert.py`

#### System Diagnostics
Before running analysis, check that everything is set up correctly:

```bash
# Unix/Linux/Mac
./bbert --check

# Windows  
bbert.bat --check

# Cross-platform
python bbert.py --check
```

#### Usage Examples

**Single file:**
```bash
./bbert example/sample.fasta --output_dir example
```

**Multiple files:**
```bash
./bbert example/Pseudo*.fasta.gz --output_dir example/ --batch_size 512
```

**With embeddings (warning: large files):**
```bash
./bbert \
    example/Pseudomonas_aeruginosa_R1.fasta.gz \
    example/Pseudomonas_aeruginosa_R2.fasta.gz \
    example/Saccharomyces_paradoxus_R1.fasta.gz \
    example/Saccharomyces_paradoxus_R2.fasta.gz \
    --output_dir example --emb_out
```

#### What the Executables Check:
- ✅ Python 3.10+ installation
- ✅ Required packages (PyTorch, Transformers, BioPython, etc.)  
- ✅ Model files downloaded via Git LFS
- ✅ GPU availability (CUDA/MPS)
- ✅ Input files exist
- ✅ Conda environment status

### 2.2. Direct Script Usage
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

#### Single File
```bash
python source/inference.py example/sample.fasta --output_dir example --batch_size 1024
```

#### Using Wildcards
```bash
# All .fasta.gz files in example directory
python source/inference.py example/*.fasta.gz --output_dir example

#### With Embeddings (Warning: Large Output Files)
```bash
python source/inference.py \
    example/Pseudomonas_aeruginosa_R1.fasta.gz \
    example/Pseudomonas_aeruginosa_R2.fasta.gz \
    example/Saccharomyces_paradoxus_R1.fasta.gz \
    example/Saccharomyces_paradoxus_R2.fasta.gz \
    --output_dir example \
    --emb_out
```

### Arguments
- `files`: List of input file paths to process (required, can be relative or absolute paths)
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
df = pd.read_parquet('example/example_scores_len.parquet')
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
    --output_dir example \
    --output_prefix example
```

**Output:**
- `example_good_long_scores.tsv.gz` - Reads ≥100bp with scores
- `example_good_short_scores.tsv.gz` - Reads <100bp (excluded from analysis)

### Paired-End Data Processing

For paired-end sequencing data (R1/R2 files), merge scores from both reads using the files from our usage examples:

```bash
# Merge P. aeruginosa R1/R2 scores
python source/merge_paired_scores.py \
    --r1 example/Pseudomonas_aeruginosa_R1_scores_len_emb.parquet \
    --r2 example/Pseudomonas_aeruginosa_R2_scores_len_emb.parquet \
    --output_dir example \
    --output_prefix Pseudomonas_aeruginosa

# Merge S. paradoxus R1/R2 scores  
python source/merge_paired_scores.py \
    --r1 example/Saccharomyces_paradoxus_R1_scores_len_emb.parquet \
    --r2 example/Saccharomyces_paradoxus_R2_scores_len_emb.parquet \
    --output_dir example \
    --output_prefix Saccharomyces_paradoxus
```

**Output:**
- `Pseudomonas_aeruginosa_good_long_scores.tsv.gz` - Combined scores for read pairs ≥100bp
- `Pseudomonas_aeruginosa_good_short_scores.tsv.gz` - Filtered short read pairs
- `Saccharomyces_paradoxus_good_long_scores.tsv.gz` - Combined scores for read pairs ≥100bp  
- `Saccharomyces_paradoxus_good_short_scores.tsv.gz` - Filtered short read pairs

**Score combination logic:**
- Both R1,R2 ≥100bp: Average their `loss` and `bact_prob`
- Only one read ≥100bp: Use that read's scores  
- Both reads <100bp: Exclude from long scores file


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

## 5. Visualizing BBERT Embeddings

BBERT can output high-dimensional embeddings that capture sequence features learned by the transformer model. These embeddings can be visualized using t-SNE to explore how BBERT groups sequences by organism type, coding status, and reading frame.

### Prerequisites

The visualization requires embeddings to be generated during inference using the `--emb_out` flag:

```bash
# Generate embeddings for visualization (if not done already)
./bbert \
    example/Pseudomonas_aeruginosa_R1.fasta.gz \
    example/Pseudomonas_aeruginosa_R2.fasta.gz \
    example/Saccharomyces_paradoxus_R1.fasta.gz \
    example/Saccharomyces_paradoxus_R2.fasta.gz \
    --output_dir example --emb_out --batch_size 512
```

**⚠️ Important**: Embedding files (`*_scores_len_emb.parquet`) are much larger than regular output files and processing is slower.

### Creating t-SNE Visualizations

Once embeddings are generated, create interactive visualizations:

```bash
# Check that embedding files exist
ls example/*_scores_len_emb.parquet

# If no embedding files found, you'll see:
# ls: example/*_scores_len_emb.parquet: No such file or directory
# Run the --emb_out command above first!

# Generate t-SNE visualization 
python example/visualize_embeddings.py --results_dir example --output_dir example --max_samples 500 --perplexity 30
```

### What the Visualization Shows

The script creates a 4-panel t-SNE plot (`example/bbert_tsne_visualization.png`) that reveals:

1. **Organism Separation**: How well BBERT separates bacterial (Pseudomonas) vs. eukaryotic (Saccharomyces) sequences
2. **Coding Classification**: Distinction between protein-coding and non-coding DNA sequences  
3. **Reading Frame Grouping**: Clustering of sequences by predicted reading frames (+1,+2,+3,-1,-2,-3)
4. **Sample Distribution**: Comparison between R1 and R2 paired-end reads

### Interpreting Results

**Expected patterns:**
- **Clear organism separation**: Pseudomonas and Saccharomyces should form distinct clusters
- **Coding vs. non-coding**: Protein-coding sequences often cluster separately from non-coding regions
- **Frame consistency**: Sequences in the same reading frame may group together
- **R1/R2 similarity**: Paired-end reads from the same organism should cluster near each other

**Troubleshooting visualization:**

If embeddings are missing:
```bash
# Error: No embedding parquet files found in example
# Solution: Re-run BBERT with --emb_out flag
./bbert example/*.fasta.gz --output_dir example --emb_out
```

If visualization fails:
```bash
# Install additional dependencies if needed
pip install matplotlib seaborn scikit-learn
```

## 6. Troubleshooting

### Installing Git

If you don't have Git installed on your system, you'll need it to clone the repository and access Git LFS files.

**On Unix/Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install git

# CentOS/RHEL/Fedora
sudo yum install git
# OR on newer versions
sudo dnf install git

# Arch Linux
sudo pacman -S git
```

**On Mac:**
```bash
# Using Homebrew (recommended)
brew install git

# Using MacPorts
sudo port install git

# Or download from: https://git-scm.com/download/mac
```

**On Windows:**
- Download Git for Windows from: https://git-scm.com/download/win
- Or use Windows Subsystem for Linux (WSL) and follow Linux instructions

### Downloading Without Git (Alternative Methods)

If you cannot install Git, here are alternative approaches:

#### Option 1: Manual File Download
**⚠️ Warning:** This is tedious and not recommended for large repositories.

1. **Download repository code:** Use GitHub's "Download ZIP" button
2. **Manually download model files:** 
   - Navigate to each model file in the GitHub web interface
   - Click on the file, then "View raw" 
   - Right-click "View raw" and "Save link as..."
   - Repeat for all model files in these directories:
     - `models/diverse_bact_12_768_6_20000/`
     - `emb_class_bact/models/`
     - `emb_class_frame/models/`
     - `emb_class_coding/models/`

#### Option 2: Use Git GUI Clients
Some GUI clients handle Git LFS automatically:
- **GitHub Desktop:** https://desktop.github.com/
- **Sourcetree:** https://www.sourcetreeapp.com/
- **GitKraken:** https://www.gitkraken.com/

### Common Installation Issues

#### Issue: "git: command not found"
**Solution:** Install Git using the instructions above.

#### Issue: "git-lfs: command not found" 
**Solution:** Install Git LFS using the platform-specific instructions in Section 1.1.

#### Issue: "tokenizers version conflict" (transformers ImportError)
**Solution:** Install the correct tokenizers version:
```bash
conda activate BBERT_mac  # or your environment name
pip install tokenizers==0.13.3
```

#### Issue: "CUDA not available" on systems with GPU
**Solutions:**
- Verify CUDA installation: `nvidia-smi`
- Reinstall PyTorch with CUDA support
- For Mac: The model will automatically use MPS (Metal Performance Shaders)

#### Issue: "Out of memory" errors
**Solutions:**
- Reduce batch size: `--batch_size 32` or lower
- Close other applications to free memory
- For CPU-only systems, use smaller batch sizes (8-16)

#### Issue: Repository download as ZIP doesn't include model files
**Explanation:** GitHub's ZIP download doesn't include Git LFS files by default.
**Solution:** Use `git clone` with Git LFS as described in Section 1.1, or try the alternative methods above.

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues:** https://github.com/AmirErez/BBERT/issues
2. **Create a new issue:** Include your:
   - Operating system and version
   - Python version
   - Complete error message
   - Command you were trying to run
3. **Provide system information:**
   ```bash
   python --version
   conda --version  # or mamba --version
   git --version
   git lfs version
   ```
