#!/usr/bin/env python3
"""
Convert BBERT parquet scores to compressed TSV format for consistency.

This script converts single-end BBERT inference results from Parquet format
to the same TSV.GZ format used by the paired-end merger, for consistency.
"""

import pandas as pd
import os
import logging
import gzip
import argparse
import pyarrow.parquet as pq
import sys

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def convert_parquet_to_tsv(parquet_file, output_dir, output_prefix, min_length=100):
    """
    Convert parquet scores to TSV format, splitting by length.
    
    Args:
        parquet_file: Path to input parquet file
        output_dir: Output directory  
        output_prefix: Output filename prefix
        min_length: Minimum read length for "long" classification (default: 100)
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Converting {parquet_file} ({os.path.getsize(parquet_file) / 1024 / 1024:.1f} MB)")
    
    # Read parquet file
    try:
        df = pd.read_parquet(parquet_file, columns=['id', 'len', 'loss', 'bact_prob'])
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")
        return False
    
    if df.empty:
        logger.error("Parquet file is empty")
        return False
    
    logger.info(f"Total reads: {len(df)}")
    
    # Split by length
    long_reads = df[df['len'] >= min_length].copy()
    short_reads = df[df['len'] < min_length].copy()
    
    # Keep only needed columns for consistency with paired-end output
    long_reads = long_reads[['id', 'loss', 'bact_prob']].reset_index(drop=True)
    short_reads = short_reads[['id', 'len', 'bact_prob']].reset_index(drop=True)
    
    logger.info(f"Long reads (≥{min_length}bp): {len(long_reads)}")
    logger.info(f"Short reads (<{min_length}bp): {len(short_reads)}")
    
    # Create output paths
    long_output = os.path.join(output_dir, f"{output_prefix}_good_long_scores.tsv.gz")
    short_output = os.path.join(output_dir, f"{output_prefix}_good_short_scores.tsv.gz")
    
    # Save long reads
    try:
        with gzip.open(long_output, 'wt', encoding='utf-8') as f:
            long_reads.to_csv(f, sep='\t', index=False)
        logger.info(f"Saved {len(long_reads)} long scores to {long_output}")
    except Exception as e:
        logger.error(f"Error saving long scores: {e}")
        return False
    
    # Save short reads (if any)
    if not short_reads.empty:
        try:
            with gzip.open(short_output, 'wt', encoding='utf-8') as f:
                short_reads.to_csv(f, sep='\t', index=False)
            logger.info(f"Saved {len(short_reads)} short scores to {short_output}")
        except Exception as e:
            logger.error(f"Error saving short scores: {e}")
            return False
    else:
        logger.info("No short reads to save")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert BBERT parquet scores to TSV format")
    parser.add_argument("--input", required=True, help="Path to input parquet file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_prefix", required=True, help="Output filename prefix (e.g., 'SRR8100008')")
    parser.add_argument("--min_length", type=int, default=100, help="Minimum read length for long classification (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Check for existing output
    long_output = os.path.join(args.output_dir, f"{args.output_prefix}_good_long_scores.tsv.gz")
    if os.path.exists(long_output):
        logger.info(f"Output file already exists: {long_output}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert the file
    logger.info("Converting parquet to TSV format")
    if convert_parquet_to_tsv(args.input, args.output_dir, args.output_prefix, args.min_length):
        logger.info("Successfully completed")
        return 0
    else:
        logger.error("Conversion failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())