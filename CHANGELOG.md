# Changelog

All notable changes to BBERT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-11-29

### Added - Major Refactoring for Pip Installation

#### Package Structure
- **Pip-installable package**: BBERT is now a proper Python package that can be installed with `pip install -e .`
- **New package structure**: Code organized under `bbert/` namespace
  - `bbert.core` - Core functionality (config, dataset, collator, utils)
  - `bbert.models` - Model architectures (BertClassifier)
  - `bbert.utils` - Utility functions (inference, common utilities)
  - `bbert.cli` - Command-line interface
- **Package metadata**: `pyproject.toml`, `setup.py`, version management
- **Type hints support**: PEP 561 compliant with `py.typed` marker

#### Command-Line Interface (CLI)
- **New `bbert` command**: Unified CLI for all operations
  - `bbert infer` - Run inference on DNA sequences (replaces `python source/inference.py`)
  - `bbert download` - Download models from HuggingFace (replaces `python source/download_models.py`)
  - `bbert --version` - Show version information
  - `bbert --help` - Show help and available commands
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux
- **Automatic entry point**: `bbert` command available after `pip install`
- **Alternative syntax**: `python -m bbert` also works

#### Code Organization
- **Centralized configuration**: All constants in `bbert/core/config.py`
- **Shared utilities**: Common functions in `bbert/utils/common.py` (reduces code duplication)
- **Clean imports**: Package-based imports (`from bbert import ...`)
- **Better error handling**: Informative error messages with recovery suggestions

#### Documentation
- **Updated README.md**:
  - Quick start guide for v0.2.0
  - Pip installation instructions
  - CLI usage examples
  - Backward compatibility notes
- **Implementation guides**: Detailed migration documentation
- **Type annotations**: Improved function signatures with type hints

### Changed

#### Breaking Changes
- **Import paths**: Old imports from `BERT_model.*` and `emb_model.*` replaced with `bbert.*`
  - Old: `from BERT_model.config import SEQUENCE_LENGTH`
  - New: `from bbert.core.config import SEQUENCE_LENGTH`
  - Old: `from emb_model.architecture import BertClassifier`
  - New: `from bbert.models.classifier import BertClassifier`

#### Removed Legacy Code
- **Removed old package directories**: `BERT_model/` and `emb_model/` have been removed (replaced by `bbert/`)
- **Removed duplicate files**:
  - `source/inference_utils.py` (now `bbert.utils.inference`)
  - `source/download_models.py` (now `bbert.cli.commands.download`)
  - `source/inference.py` (replaced by `bbert infer` CLI command)
  - `bbert.py` (replaced by `bbert` CLI command)
  - `test_package.py` (temporary test file)
- **Removed `source/` directory entirely** (replaced by organized structure below)

#### Reorganized Project Structure
Scripts and examples moved to clearer organization:

**scripts/** - Development and testing scripts
- `scripts/training/` - Model training scripts (train.py, emb_class_train.py, emb_class_inference.py)
- `scripts/testing/` - Validation scripts (test_genomic_accuracy.py, test_inference_accuracy.py)

**examples/** - Example workflows, utilities, and data
- `examples/data/` - Example data files (~15MB compressed reference genomes and test reads)
- `examples/utilities/` - Utility scripts (score.py, extract_coding_AA.py, merge_*.py, convert_scores_to_tsv.py)
- `examples/visualization/` - Visualization tools (visualize_embeddings.py)

#### Updated Scripts
All scripts now use `bbert` package imports:
- Training scripts in `scripts/training/` - Use `bbert.core.*` and `bbert.models.*`
- Testing scripts in `scripts/testing/` - Use `bbert.utils.common`
- Example utilities in `examples/` - Use `bbert.*` imports throughout

#### Testing
- **Test suite**: 93/93 tests passing (100% pass rate) ✅
- **Removed obsolete tests**: Removed 9 tests for old `source/inference.py` (functionality now in CLI)
- **Fixed test expectations**: Tests updated for new directory structure
- **Path validation mocks**: Added proper mocking for file existence checks
- **Import updates**: All tests updated to use new `bbert.*` imports and new paths

### Fixed

#### Paired-End Read Merging
- **Fixed `merge_paired_scores.py`**: Correctly handles paired-end read IDs with `/1` and `/2` suffixes
  - **Problem**: Script was merging R1 and R2 directly on `id` column, causing zero matches (R1 has `/1` suffix, R2 has `/2`)
  - **Solution**: Strip `/1` and `/2` suffixes to create `pair_id` for matching, then merge on `pair_id`
  - **Impact**: `merge_paired_scores.py` now successfully pairs 5000/5000 reads instead of 0/5000
  - **Location**: `examples/utilities/merge_paired_scores.py` lines 80-102

#### README Documentation Paths
- **Fixed all outdated `source/` paths**: Updated all README examples to use new directory structure
  - `source/extract_coding_AA.py` → `examples/utilities/extract_coding_AA.py`
  - `source/visualize_embeddings.py` → `examples/visualization/visualize_embeddings.py`
  - `source/test_genomic_accuracy.py` → `scripts/testing/test_genomic_accuracy.py`
- **Fixed missing workflow steps**: Added `bbert infer` steps before post-processing examples
  - `extract_coding_AA.py` examples now include score generation step
  - Updated paths from `examples/data/` to `results/` for consistency
- **Fixed embedding visualization paths**: Corrected parquet file paths to match `--output-dir example`
  - Changed `examples/data/*_emb.parquet` → `example/*_emb.parquet` in visualization examples
- **Fixed compressed file extensions**: Updated genomic accuracy examples to use `.gz` extensions
  - Changed `.fasta` and `.gtf` → `.fasta.gz` and `.gtf.gz` (6 examples updated)
  - All reference genome files are now correctly referenced as compressed files
- **Impact**: All README examples now work correctly with complete workflows

#### GitHub Actions CI/CD Workflow
- **Fixed `.github/workflows/test.yml`**: Updated workflow for v0.2.0 package structure
  - Added `pip install -e .` step to install package in editable mode
  - **Added `bbert download` step**: Downloads models from HuggingFace Hub before tests
  - Replaced `python bbert.py --check` → `bbert --version` for system diagnostics
  - Updated test coverage source from `--source=source` → `--source=bbert`
  - Replaced `python bbert.py` → `bbert infer` in end-to-end example
  - Added graceful handling for missing unit tests (tests/ directory in .gitignore)
  - Added model directory check to diagnostics
- **Impact**: GitHub Actions workflow now runs successfully with v0.2.0 package structure

#### Critical .gitignore Fixes
- **Removed `bbert/` from .gitignore**: The package directory was being ignored by git
  - **Problem 1**: Line 1 of .gitignore was `bbert/`, preventing the entire package from being tracked
  - **Impact**: Package code wasn't in the repository, causing `pip install -e .` to fail in CI/CD
  - **Problem 2**: Pattern `models/` was also matching `bbert/models/`, ignoring model code files
  - **Solution**:
    - Removed `bbert/` from .gitignore
    - Changed `models/` → `/models/` to only ignore root-level models directory
    - Changed `tests/` → `/tests/` and `results/` → `/results/` for consistency
    - Added all 20 package files to git: CLI, core, models, utils, data modules
  - **Files added**:
    - `bbert/__init__.py`, `__main__.py`, `_version.py`, `py.typed`
    - `bbert/cli/`: main.py, commands/download.py, commands/infer.py
    - `bbert/core/`: config.py, dataset.py, collator.py, utils.py
    - `bbert/models/`: classifier.py
    - `bbert/utils/`: inference.py, common.py
    - `bbert/data/`: __init__.py
- **Impact**: Package can now be installed from git repository, CI/CD will work

#### Legacy Command References in Scripts
- **Fixed outdated help text in utility scripts**: Updated all script examples to use new directory structure
  - **Files updated**:
    - `scripts/testing/test_genomic_accuracy.py`: Changed `python source/test_genomic_accuracy2.py` → `python scripts/testing/test_genomic_accuracy.py`
    - `examples/utilities/extract_coding_AA.py`: Changed `python source/extract_coding_AA.py` → `python examples/utilities/extract_coding_AA.py`
    - `scripts/testing/generate_annotated_reads.py`: Changed `python source/generate_annotated_reads.py` → `python scripts/testing/generate_annotated_reads.py`
  - **Impact**: All `--help` outputs now show correct command paths
- **Fixed error messages in `bbert/utils/inference.py`**: Updated 5 locations with old download command references
  - Changed all references from `python source/download_models.py` → `bbert download`
  - **Locations**: Lines 104, 114, 136, 174, 203
  - **Impact**: Error messages now guide users to correct CLI command

### Migration Guide

#### For End Users
```bash
# New installation method
git clone https://github.com/AmirErez/BBERT.git
cd BBERT
pip install -e .

# New CLI usage (recommended)
bbert download                              # Download models
bbert infer data.fasta --output-dir results # Run inference

# Example utilities (optional)
python examples/utilities/score.py ...
python examples/visualization/visualize_embeddings.py ...
```

#### For Developers
```python
# New import style
from bbert.core.config import SEQUENCE_LENGTH, HIDDEN_SIZE_768
from bbert.models.classifier import BertClassifier
from bbert.utils.inference import load_bbert_model, get_device
from bbert.utils.common import setup_logging, FRAME_MAPPING

# Old imports still work in source/ scripts but deprecated
```

### Data Packaging

#### Example Data Included (~15MB)
- **Test reads**: Small FASTA/FASTQ.gz files for quick testing
- **Reference genomes**: Compressed (.gz) bacterial and eukaryotic genomes
- **Visualizations**: Example t-SNE plots

#### Data Optimization
- **Directory reorganization**:
  - Moved `example/` → `examples/data/` for consistency
  - Consolidated model directories: `emb_class_*` → `models/classifiers/`
- Reference genomes compressed with gzip (83% size reduction: 60MB → 10.4MB)
- All data generation scripts updated to handle compressed (.gz) files automatically
- Generated outputs (*.parquet) excluded from package (saved 561MB)
- See `examples/data/README.md` for complete data documentation

### Technical Details

#### Package Configuration
- **Python requirement**: 3.10+
- **License**: MIT
- **Entry points**: Registered in `pyproject.toml`
- **Dependencies**: Managed via `requirements.txt` and `pyproject.toml`
- **Package data**: ~15MB of example data included

#### Code Quality
- **Refactored functions**: Long functions split into smaller helpers
- **Reduced duplication**: ~170 lines of duplicate code consolidated
- **Enhanced error handling**: Better validation and error messages
- **Improved docstrings**: Google-style documentation

### Breaking Changes from v0.1.0

- ❌ **`source/` directory removed**: Use CLI commands (`bbert infer`, `bbert download`) or scripts in `scripts/` and `examples/`
- ❌ **Old imports removed**: Must use `bbert.*` imports instead of `BERT_model.*` or `emb_model.*`
- ✅ **Data format unchanged**: Input/output formats are the same
- ✅ **Model compatibility**: All existing models work with new code
- ✅ **Conda environments**: Existing conda environments still work with `pip install -e .`

### Known Issues

- Old `python bbert.py` wrapper not yet updated (use `bbert infer` instead)

### Future Plans

- [ ] Publish to PyPI (pip install bbert)
- [ ] Additional CLI commands (test, visualize)
- [ ] Enhanced documentation and tutorials

---

## [0.1.0] - 2024-09-07

### Initial Release
- BERT-based transformer for bacterial DNA classification
- Three classification tasks: bacterial/non-bacterial, reading frame, coding/non-coding
- HuggingFace model hosting
- Example data and validation tests
- Conda environment support
- Mac (Apple Silicon), Linux (CUDA), Windows support

[0.2.0]: https://github.com/AmirErez/BBERT/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AmirErez/BBERT/releases/tag/v0.1.0
