#!/usr/bin/env python3
"""
Merge paired-end BBERT scores from R1 and R2 files.

This script combines R1 and R2 BBERT inference results from the same DNA fragments,
applying length filtering and score averaging to produce consolidated results.
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
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def read_parquet_efficient(path):
    """Read parquet file with optimized types for memory efficiency."""
    try:
        # Read only needed columns
        table = pq.read_table(path, columns=['id', 'len', 'loss', 'bact_prob'])
        
        # Cast to smaller types for memory efficiency
        table = table.cast(pa.schema([
            ('id', pa.string()),
            ('len', pa.uint16()),
            ('loss', pa.float32()),
            ('bact_prob', pa.float32())
        ]))
        
        return table.to_pandas()
    except Exception as e:
        raise FileNotFoundError(f"Error reading {path}: {e}")

def merge_paired_scores(r1_file, r2_file, min_length=100):
    """
    Merge paired-end BBERT scores from R1 and R2 files.
    
    Args:
        r1_file: Path to R1 scores parquet file
        r2_file: Path to R2 scores parquet file  
        min_length: Minimum read length to include (default: 100)
    
    Returns:
        tuple: (long_scores_df, short_scores_df) or (None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing paired files:")
    logger.info(f"  R1: {r1_file} ({os.path.getsize(r1_file) / 1024 / 1024:.1f} MB)")
    logger.info(f"  R2: {r2_file} ({os.path.getsize(r2_file) / 1024 / 1024:.1f} MB)")
    
    # Read score files
    try:
        r1_scores = read_parquet_efficient(r1_file)
        r2_scores = read_parquet_efficient(r2_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None, None
    
    if r1_scores.empty or r2_scores.empty:
        logger.error("One or both score files are empty")
        return None, None
    
    if len(r1_scores) != len(r2_scores):
        logger.warning(f"Read count mismatch: R1={len(r1_scores)}, R2={len(r2_scores)}")
    
    logger.info(f"Read counts: R1={len(r1_scores)}, R2={len(r2_scores)}")
    
    # Rename columns to distinguish R1/R2
    r1_scores = r1_scores.rename(columns={
        'loss': 'R1_loss',
        'len': 'R1_len', 
        'bact_prob': 'R1_bact_prob'
    })
    
    r2_scores = r2_scores.rename(columns={
        'loss': 'R2_loss',
        'len': 'R2_len',
        'bact_prob': 'R2_bact_prob'  
    })
    
    # Merge on read ID
    merged = r1_scores.merge(r2_scores, on='id', how='inner')
    logger.info(f"Merged dataset: {len(merged)} read pairs")
    
    # Apply length filtering and score combination logic
    conditions = [
        (merged['R1_len'] >= min_length) & (merged['R2_len'] >= min_length),  # Both good
        (merged['R1_len'] >= min_length) & (merged['R2_len'] < min_length),   # R1 only
        (merged['R1_len'] < min_length) & (merged['R2_len'] >= min_length)    # R2 only
    ]
    
    # Calculate combined scores
    merged.loc[conditions[0], 'loss'] = (merged['R1_loss'] + merged['R2_loss']) / 2
    merged.loc[conditions[0], 'bact_prob'] = (merged['R1_bact_prob'] + merged['R2_bact_prob']) / 2
    
    merged.loc[conditions[1], 'loss'] = merged['R1_loss'] 
    merged.loc[conditions[1], 'bact_prob'] = merged['R1_bact_prob']
    
    merged.loc[conditions[2], 'loss'] = merged['R2_loss']
    merged.loc[conditions[2], 'bact_prob'] = merged['R2_bact_prob']
    
    # Split into long (good) and short (filtered out) reads
    long_scores = merged.dropna(subset=['loss']).copy()
    short_scores = merged[merged['loss'].isna()].copy()
    
    # Clean up dataframes
    long_scores = long_scores[['id', 'loss', 'bact_prob']].reset_index(drop=True)
    short_scores = short_scores[['id', 'R1_len', 'R2_len', 'R1_bact_prob', 'R2_bact_prob']].reset_index(drop=True)
    
    logger.info(f"Results: {len(long_scores)} long pairs, {len(short_scores)} short pairs")
    
    return long_scores, short_scores

def save_results(long_scores, short_scores, output_dir, output_prefix):
    """Save merged results to compressed files in the same directory."""
    logger = logging.getLogger(__name__)
    
    long_output = os.path.join(output_dir, f"{output_prefix}_good_long_scores.tsv.gz")
    short_output = os.path.join(output_dir, f"{output_prefix}_good_short_scores.tsv.gz")
    
    # Save long scores
    if not long_scores.empty:
        try:
            with gzip.open(long_output, 'wt', encoding='utf-8') as f:
                long_scores.to_csv(f, sep='\t', index=False)
            logger.info(f"Saved {len(long_scores)} long scores to {long_output}")
        except Exception as e:
            logger.error(f"Error saving long scores: {e}")
            return False
    
    # Save short scores  
    if not short_scores.empty:
        try:
            with gzip.open(short_output, 'wt', encoding='utf-8') as f:
                short_scores.to_csv(f, sep='\t', index=False)
            logger.info(f"Saved {len(short_scores)} short scores to {short_output}")
        except Exception as e:
            logger.error(f"Error saving short scores: {e}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Merge paired-end BBERT scores from R1 and R2 files")
    parser.add_argument("--r1", required=True, help="Path to R1 scores parquet file")
    parser.add_argument("--r2", required=True, help="Path to R2 scores parquet file")
    parser.add_argument("--output_dir", required=True, help="Output directory for both long and short scores")
    parser.add_argument("--output_prefix", required=True, help="Output filename prefix (e.g., 'SRR8100008')")
    parser.add_argument("--min_length", type=int, default=100, help="Minimum read length (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.r1):
        logger.error(f"R1 file not found: {args.r1}")
        return 1
        
    if not os.path.exists(args.r2):
        logger.error(f"R2 file not found: {args.r2}")
        return 1
    
    # Create output paths
    long_output = os.path.join(args.output_dir, f"{args.output_prefix}_good_long_scores.tsv.gz")
    
    # Check for existing output
    if os.path.exists(long_output):
        logger.info(f"Output file already exists: {long_output}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the files
    logger.info("Merging paired-end scores")
    long_scores, short_scores = merge_paired_scores(args.r1, args.r2, args.min_length)
    
    if long_scores is None:
        logger.error("Failed to process files")
        return 1
    
    # Save results
    if save_results(long_scores, short_scores, args.output_dir, args.output_prefix):
        logger.info("Successfully completed")
        return 0
    else:
        logger.error("Failed to save results")
        return 1

if __name__ == "__main__":
    sys.exit(main())