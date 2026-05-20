#!/usr/bin/env python3
"""
Merge paired-end BBERT scores from R1 and R2 files.

This script combines R1 and R2 BBERT inference results from the same DNA fragments,
averaging bact_prob to produce consolidated results.
"""

import pandas as pd
import os
import logging
import gzip
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def read_parquet_efficient(path):
    try:
        table = pq.read_table(path, columns=['id', 'bact_prob'])
        table = table.cast(pa.schema([
            ('id', pa.string()),
            ('bact_prob', pa.float32())
        ]))
        return table.to_pandas()
    except Exception as e:
        raise FileNotFoundError(f"Error reading {path}: {e}")

def merge_paired_scores(r1_file, r2_file):
    logger = logging.getLogger(__name__)

    logger.info(f"Processing paired files:")
    logger.info(f"  R1: {r1_file} ({os.path.getsize(r1_file) / 1024 / 1024:.1f} MB)")
    logger.info(f"  R2: {r2_file} ({os.path.getsize(r2_file) / 1024 / 1024:.1f} MB)")

    try:
        r1_scores = read_parquet_efficient(r1_file)
        r2_scores = read_parquet_efficient(r2_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    if r1_scores.empty or r2_scores.empty:
        logger.error("One or both score files are empty")
        return None

    if len(r1_scores) != len(r2_scores):
        logger.warning(f"Read count mismatch: R1={len(r1_scores)}, R2={len(r2_scores)}")

    logger.info(f"Read counts: R1={len(r1_scores)}, R2={len(r2_scores)}")

    # Strip /1 and /2 suffixes from read IDs for proper pairing
    r1_scores['pair_id'] = r1_scores['id'].str.replace(r'/[12]$', '', regex=True)
    r2_scores['pair_id'] = r2_scores['id'].str.replace(r'/[12]$', '', regex=True)

    r1_scores = r1_scores.rename(columns={'bact_prob': 'R1_bact_prob'})
    r2_scores = r2_scores.rename(columns={'bact_prob': 'R2_bact_prob'})

    merged = r1_scores.merge(r2_scores, on='pair_id', how='inner', suffixes=('_r1', '_r2'))
    merged['id'] = merged['id_r1']
    merged = merged.drop(columns=['id_r1', 'id_r2', 'pair_id'])

    merged['bact_prob'] = ((merged['R1_bact_prob'] + merged['R2_bact_prob']) / 2).round(3)

    result = merged[['id', 'bact_prob']].reset_index(drop=True)
    logger.info(f"Merged dataset: {len(result)} read pairs")
    return result

def save_results(scores, output_dir, output_prefix):
    logger = logging.getLogger(__name__)
    output = os.path.join(output_dir, f"{output_prefix}_merged_scores.tsv.gz")
    try:
        with gzip.open(output, 'wt', encoding='utf-8') as f:
            scores.to_csv(f, sep='\t', index=False)
        logger.info(f"Saved {len(scores)} merged scores to {output}")
        return True
    except Exception as e:
        logger.error(f"Error saving scores: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge paired-end BBERT scores from R1 and R2 files")
    parser.add_argument("--r1", required=True, help="Path to R1 scores parquet file")
    parser.add_argument("--r2", required=True, help="Path to R2 scores parquet file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_prefix", required=True, help="Output filename prefix (e.g., 'SRR8100008')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(args.r1):
        logger.error(f"R1 file not found: {args.r1}")
        return 1

    if not os.path.exists(args.r2):
        logger.error(f"R2 file not found: {args.r2}")
        return 1

    output = os.path.join(args.output_dir, f"{args.output_prefix}_merged_scores.tsv.gz")
    if os.path.exists(output):
        logger.info(f"Output file already exists: {output}")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)

    scores = merge_paired_scores(args.r1, args.r2)
    if scores is None:
        logger.error("Failed to process files")
        return 1

    if save_results(scores, args.output_dir, args.output_prefix):
        logger.info("Successfully completed")
        return 0
    else:
        logger.error("Failed to save results")
        return 1

if __name__ == "__main__":
    sys.exit(main())
