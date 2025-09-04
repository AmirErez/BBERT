#!/bin/bash
#SBATCH --job-name=merge_scores
#SBATCH --output=logs/merge_scores_%A_%a.log
#SBATCH --error=logs/merge_scores_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# SLURM array job script for batch processing paired-end score merging
# Usage: sbatch --array=1-N scripts/batch_merge_scores.sh accessions.csv /path/to/scores

set -euo pipefail

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: sbatch --array=1-N $0 <accessions.csv> <scores_directory>"
    echo "Example: sbatch --array=1-100 scripts/batch_merge_scores.sh accessions.csv /data/sra_scores"
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

# Run the single-accession merger
python source/merge_paired_scores.py \
    --accession "$ACCESSION" \
    --input_dir "$SCORES_DIR" \
    --verbose

EXIT_CODE=$?

echo "=================================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Successfully processed $ACCESSION"
else
    echo "Failed to process $ACCESSION"
fi

echo "=================================================="

exit $EXIT_CODE