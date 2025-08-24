#!/bin/bash

FASTQ_BASE="D:/data/test_datasets_3/datasets_1K_R1_R2"
TEST_FILE="ds_1_R1.fasta"
OUT_BASE="D:/data/test_datasets_3"
# PY_SCRIPT="/cs/usr/stavperez/sp/InSilicoSeq/dataset_unique_genus/embeddings/BBERTooD/source/inference.py"
PY_SCRIPT="source/inference.py"

python "$PY_SCRIPT" \
    --input_dir "$FASTQ_BASE" \
    --input_files "$TEST_FILE" \
    --output_dir "$OUT_BASE" \
    --batch_size 1024 \
    --emb_out