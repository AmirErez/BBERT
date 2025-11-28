#!/usr/bin/env python3
"""
BBERT - Cross-platform Python executable for BERT-based DNA sequence analysis
Usage: python bbert.py file1.fasta file2.fastq.gz --output_dir results [options]

This is a pure Python implementation that works on all platforms.
"""

import os
import sys
import subprocess
import shutil
import importlib
from pathlib import Path
import argparse
import platform

# Color codes for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color
    
    # Disable colors on Windows unless we have colorama
    @classmethod
    def init(cls):
        if platform.system() == "Windows":
            try:
                import colorama
                colorama.init()
            except ImportError:
                # Disable colors on Windows without colorama
                cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ''

Colors.init()

def print_error(msg):
    """Print error message in red"""
    print(f"{Colors.RED}[ERROR] {msg}{Colors.NC}", file=sys.stderr)

def print_success(msg):
    """Print success message in green"""
    print(f"{Colors.GREEN}[OK] {msg}{Colors.NC}")

def print_warning(msg):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}[WARNING] {msg}{Colors.NC}")

def print_info(msg):
    """Print info message in blue"""
    print(f"{Colors.BLUE}[INFO] {msg}{Colors.NC}")

def command_exists(command):
    """Check if command exists in PATH"""
    return shutil.which(command) is not None

def check_python_version():
    """Check if Python version is adequate"""
    print_info("Checking Python environment...")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 10):
        print_error(f"Python {version_info.major}.{version_info.minor} detected. BBERT requires Python 3.10+")
        return False
    
    print_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} detected")
    return True

def check_conda_env():
    """Check conda environment status"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env and conda_env != 'base':
        print_success(f"Conda environment: {conda_env}")
    elif command_exists('conda'):
        env_name = "BBERT_mac" if platform.system() == "Darwin" else "BBERT"
        print_warning("You're in the base conda environment")
        print_info(f"Consider activating a BBERT environment: conda activate {env_name}")

def check_python_packages():
    """Check if required Python packages are installed"""
    print_info("Checking required Python packages...")
    
    # List of (import_name, package_name) tuples
    packages_to_check = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pyarrow', 'pyarrow'),
        ('sklearn', 'scikit-learn'),
        ('Bio', 'biopython'),
        ('tqdm', 'tqdm'),
        ('yaml', 'pyyaml'),
        ('huggingface_hub', 'huggingface_hub')
    ]
    
    missing_packages = []
    
    for import_name, package_name in packages_to_check:
        try:
            if import_name == 'sklearn':
                # Special import for scikit-learn
                from sklearn import __version__
            elif import_name == 'Bio':
                # Special import for biopython
                from Bio import SeqIO
            elif import_name == 'yaml':
                # Standard yaml import for pyyaml
                import yaml
            else:
                importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print_error(f"Missing required packages: {', '.join(missing_packages)}")
        print_info("Please install missing packages or activate the BBERT environment")
        print_info(f"To install: pip install {' '.join(missing_packages)}")
        return False
    
    print_success("All required packages found")
    return True

def is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer file instead of actual content."""
    try:
        # LFS pointer files are small text files (< 200 bytes)
        if file_path.stat().st_size > 200:
            return False

        # Check if it starts with "version https://git-lfs"
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if first_line.startswith('version https://git-lfs'):
                return True
    except Exception:
        # If we can't read it, assume it's a real binary file
        pass

    return False


def check_model_files():
    """Check if BBERT model files exist, download from HF if missing"""
    print_info("Checking BBERT model files...")

    # Check for key model files
    model_files = [
        "models/diverse_bact_12_768_6_20000/checkpoint-32500/pytorch_model.bin",
        "emb_class_bact/models/emb_class_model_768H_3906K_80e/epoch_80.pt",
        "emb_class_frame/models/classifier_model_2000K_37e.pth",
        "emb_class_coding/models/emb_coding_model_768_3906K_50e/epoch_46.pt"
    ]

    missing_models = []

    for model_file in model_files:
        file_path = Path(model_file)
        # Check if file doesn't exist OR is a Git LFS pointer
        if not file_path.exists() or is_lfs_pointer(file_path):
            missing_models.append(model_file)

    if missing_models:
        print_warning("Some model files are missing")
        print_info("Attempting to download models from Hugging Face Hub...")
        print()

        try:
            # Import here to avoid dependency at module level
            sys.path.insert(0, str(Path(__file__).parent / "source"))
            from download_models import download_all_models

            success = download_all_models(force=False)
            if not success:
                print_error("Failed to download models from Hugging Face")
                print_info("Alternative: Download models using Git LFS:")
                print_info("  git lfs install")
                print_info("  git lfs pull")
                return False

            print()
            print_success("Models downloaded successfully from Hugging Face")
            return True

        except ImportError as e:
            print_error(f"Cannot import model downloader: {e}")
            print_info("Please install: pip install huggingface_hub")
            return False
        except Exception as e:
            print_error(f"Failed to download models: {e}")
            print_info("Alternative: Download models using Git LFS:")
            print_info("  git lfs install")
            print_info("  git lfs pull")
            return False

    print_success("Model files found")
    return True

def check_gpu():
    """Check GPU availability"""
    print_info("Checking GPU availability...")
    
    try:
        import torch
        
        # Check for NVIDIA GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"NVIDIA GPU detected: {gpu_name}")
            return True
        
        # Check for Apple Silicon (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("Apple MPS acceleration available")
            return True
            
    except ImportError:
        pass
    
    # Check for NVIDIA GPU using nvidia-smi
    if command_exists('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            if result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                print_success(f"NVIDIA GPU detected: {gpu_name}")
                return True
        except subprocess.CalledProcessError:
            pass
    
    print_warning("No GPU acceleration detected - will use CPU")
    print_info("This will be slower but still functional")
    return True

def validate_input_files(files):
    """Validate that input files exist"""
    if not files:
        print_error("No input files specified")
        return False
    
    invalid_files = []
    for file in files:
        if not Path(file).exists():
            invalid_files.append(file)
    
    if invalid_files:
        print_error("Input files not found:")
        for file in invalid_files:
            print(f"  - {file}")
        return False
    
    print_success(f"{len(files)} input file(s) found")
    return True

def show_usage():
    """Show usage information"""
    print("""
BBERT - BERT for Bacterial DNA Classification

USAGE:
    python bbert.py <input_files...> --output_dir <directory> [OPTIONS]

EXAMPLES:
    # Single file
    python bbert.py example/sample.fasta --output_dir results
    
    # Multiple files
    python bbert.py file1.fasta file2.fastq.gz --output_dir results --batch_size 512

    # With embeddings (large output files, requires --max_reads)
    python bbert.py example/*.fasta.gz --output_dir results --emb_out --max_reads 1000

    # All example files
    python bbert.py example/Pseudomonas_*.fasta.gz example/Saccharomyces_*.fasta.gz --output_dir results

OPTIONS:
    --output_dir DIR    Directory to save output files (required)
    --batch_size N      Batch size for processing (default: 1024)
    --emb_out          Include sequence embeddings in output (warning: large files, requires --max_reads)
    --max_reads N      Maximum number of reads to process (default: all reads, required with --emb_out)
    --help             Show this help message
    --check            Run system checks only (don't process files)

SYSTEM REQUIREMENTS:
    - Python 3.10+
    - PyTorch, Transformers, BioPython, pandas, numpy, pyarrow
    - Git LFS for model files
    - GPU recommended but not required

For more information, see: https://github.com/AmirErez/BBERT
""")

def run_system_checks():
    """Run all system checks"""
    print_info("Running system checks...")
    print()
    
    checks = [
        check_python_version(),
        check_python_packages(),
        check_model_files(),
        check_gpu()
    ]
    
    check_conda_env()
    
    if all(checks):
        print()
        print_success("All system checks passed! BBERT is ready to use.")
        return True
    else:
        print()
        print_error("Some system checks failed. Please address the issues above.")
        return False

def main():
    """Main function"""
    # Handle special flags first
    if '--help' in sys.argv or '-h' in sys.argv:
        # Show inference.py help with wrapper context
        print("BBERT Wrapper - System checks + inference")
        print("==========================================")
        print()
        subprocess.run([sys.executable, 'source/inference.py', '--help'])
        return 0
    
    if '--check' in sys.argv:
        return 0 if run_system_checks() else 1
    
    # Check if we're in the right directory
    if not Path('source/inference.py').exists():
        print_error("BBERT inference script not found")
        print_info("Please run this script from the BBERT root directory")
        return 1
    
    # Show header
    print()
    print("BBERT - BERT for Bacterial DNA Classification")
    print("==============================================")
    print()
    
    # Run system checks
    checks = [
        check_python_version(),
        check_python_packages(), 
        check_model_files(),
        check_gpu()
    ]
    
    check_conda_env()
    print()
    
    if not all(checks):
        print_error("System checks failed. Use --check for detailed diagnostics.")
        return 1
    
    # Basic validation - need some arguments
    if len(sys.argv) <= 1:
        print_error("No arguments provided")
        print()
        print("USAGE: python bbert.py <arguments...> (same as: python source/inference.py <arguments...>)")
        print("For full help: python bbert.py --help")
        return 1
    
    # Count input files (arguments not starting with --)
    file_count = 0
    for arg in sys.argv[1:]:  # Skip script name
        if not arg.startswith('--') and not arg.startswith('-'):
            file_count += 1
    
    if file_count == 0:
        print_error("No input files specified")
        print()
        print("For help: python bbert.py --help")
        return 1
    
    print_success(f"{file_count} input file(s) specified")
    print()
    
    # Run BBERT inference - pass all arguments directly to inference.py
    print_info("Starting BBERT inference...")
    print()
    
    try:
        result = subprocess.run([sys.executable, 'source/inference.py'] + sys.argv[1:], check=True)
        print()
        print_success("BBERT analysis completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print_error("BBERT analysis failed")
        return e.returncode
    except KeyboardInterrupt:
        print()
        print_warning("Analysis interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())