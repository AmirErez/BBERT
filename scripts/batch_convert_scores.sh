#!/bin/bash
#SBATCH --job-name=convert_scores
#SBATCH --output=logs/convert_scores_%A_%a.log
#SBATCH --error=logs/convert_scores_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

# SLURM array job script for batch processing single-end score conversion
# Usage: sbatch --array=1-N scripts/batch_convert_scores.sh accessions.csv /path/to/scores

set -euo pipefail

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: sbatch --array=1-N $0 <accessions.csv> <scores_directory>"
    echo "Example: sbatch --array=1-100 scripts/batch_convert_scores.sh accessions.csv /data/sra_scores"
    exit 1
fi

ACCESSIONS_FILE="$1"
SCORES_DIR="$2"

# Validate inputs
if [[ ! -f "$ACCESSIONS_FILE" ]]; then
    echo "Error: Accessions file not found: $ACCESSIONS_FILE"
    exit 1
fi

if [[ ! -d "$SCORES_DIR" ]]; then
    echo "Error: Scores directory not found: $SCORES_DIR"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Get the accession for this array task (1-indexed)
ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACCESSIONS_FILE")

if [[ -z "$ACCESSION" ]]; then
    echo "Error: No accession found at line $SLURM_ARRAY_TASK_ID in $ACCESSIONS_FILE"
    exit 1
fi

echo "=================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing accession: $ACCESSION"
echo "Scores directory: $SCORES_DIR"
echo "Started at: $(date)"
echo "=================================================="

# Activate conda environment (adjust path as needed)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate BBERT

# Set up environment - adjust these paths for your system
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
cd "$SLURM_SUBMIT_DIR"

# Find the parquet file for this accession
ACCESSION_DIR="${SCORES_DIR}/${ACCESSION}"
PARQUET_FILE="${ACCESSION_DIR}/${ACCESSION}_scores_len.parquet"

# Check if parquet file exists
if [[ ! -f "$PARQUET_FILE" ]]; then
    echo "Error: Parquet file not found: $PARQUET_FILE"
    echo "Looking for alternative naming patterns..."
    
    # Try alternative patterns
    ALT_PATTERNS=(
        "${ACCESSION_DIR}/${ACCESSION}-scores_len.parquet"
        "${ACCESSION_DIR}/scores_len.parquet" 
        "${ACCESSION_DIR}/*scores_len.parquet"
    )
    
    FOUND_FILE=""
    for pattern in "${ALT_PATTERNS[@]}"; do
        if compgen -G "$pattern" > /dev/null; then
            FOUND_FILE=$(ls $pattern | head -1)
            echo "Found alternative file: $FOUND_FILE"
            break
        fi
    done
    
    if [[ -z "$FOUND_FILE" ]]; then
        echo "Error: No parquet scores file found for $ACCESSION"
        exit 1
    fi
    
    PARQUET_FILE="$FOUND_FILE"
fi

echo "Input file: $PARQUET_FILE ($(du -h "$PARQUET_FILE" | cut -f1))"

# Run the single-end converter
python source/convert_scores_to_tsv.py \
    --input "$PARQUET_FILE" \
    --output_dir "$ACCESSION_DIR" \
    --output_prefix "$ACCESSION" \
    --verbose

EXIT_CODE=$?

echo "=================================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Successfully processed $ACCESSION"
    # Show output file sizes
    if [[ -f "${ACCESSION_DIR}/${ACCESSION}_good_long_scores.tsv.gz" ]]; then
        echo "Output: ${ACCESSION_DIR}/${ACCESSION}_good_long_scores.tsv.gz ($(du -h "${ACCESSION_DIR}/${ACCESSION}_good_long_scores.tsv.gz" | cut -f1))"
    fi
else
    echo "Failed to process $ACCESSION"
fi

echo "=================================================="

exit $EXIT_CODE