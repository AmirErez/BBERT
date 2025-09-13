#!/usr/bin/env python3
"""
Extract amino acid sequences from coding sequences based on BBERT predictions.

This script reads BBERT parquet output and the original input file, filters for
coding sequences (both bacterial and non-bacterial), determines the correct 
reading frame, and outputs amino acid sequences in FASTA format.
"""

import pandas as pd
import numpy as np
import os
import logging
import argparse
import pyarrow.parquet as pq
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def detect_file_format(file_path):
    """
    Detect file format and compression from file extension.
    Returns (open_function, file_format_string)
    """
    file_path_lower = file_path.lower()
    
    # Compressed FASTA formats
    if file_path_lower.endswith(('.fasta.gz', '.fna.gz', '.fa.gz')):
        return lambda x: gzip.open(x, 'rt'), "fasta"
    
    # Compressed FASTQ formats  
    elif file_path_lower.endswith(('.fastq.gz', '.fq.gz')):
        return lambda x: gzip.open(x, 'rt'), "fastq"
    
    # Generic compressed (assume FASTQ for backward compatibility)
    elif file_path_lower.endswith('.gz'):
        return lambda x: gzip.open(x, 'rt'), "fastq"
        
    # Uncompressed FASTA formats
    elif file_path_lower.endswith(('.fasta', '.fna', '.fa')):
        return lambda x: open(x, 'rt'), "fasta"
        
    # Uncompressed FASTQ formats
    elif file_path_lower.endswith(('.fastq', '.fq')):
        return lambda x: open(x, 'rt'), "fastq"
        
    else:
        supported_formats = ['.fasta', '.fna', '.fa', '.fastq', '.fq', 
                           '.fasta.gz', '.fna.gz', '.fa.gz', '.fastq.gz', '.fq.gz']
        raise ValueError(f"Unsupported file format. Supported extensions: {', '.join(supported_formats)}")

def get_best_reading_frame(frame_probs):
    """
    Get the reading frame with highest probability.
    
    Args:
        frame_probs: Array of 6 frame probabilities corresponding to model labels 0-5
    
    Returns:
        int: Frame number (+1, +2, +3, -1, -2, -3)
    """
    frame_idx = np.argmax(frame_probs)
    # Correct BBERT mapping: positions 0-5 correspond to frames [-1, -3, -2, +1, +3, +2]
    frame_mapping = [-1, -3, -2, +1, +3, +2]
    return frame_mapping[frame_idx]

def translate_sequence(seq_str, frame):
    """
    Translate DNA sequence to amino acids based on BBERT's frame prediction.
    
    BBERT frame predictions [-1, -3, -2, +1, +3, +2] correspond to biological reading frames.
    The sequence needs to be translated in the correct reading frame.
    
    Args:
        seq_str: DNA sequence string from the original data
        frame: Reading frame predicted by BBERT (-1, -3, -2, +1, +3, +2)
    
    Returns:
        str: Amino acid sequence
    """
    seq = Seq(seq_str.upper())
    
    if frame > 0:
        # Positive frames: translate forward strand
        if frame == 1:
            # Frame +1: start at position 0
            coding_seq = seq
        elif frame == 2:
            # Frame +2: start at position 1  
            coding_seq = seq[1:]
        elif frame == 3:
            # Frame +3: start at position 2
            coding_seq = seq[2:]
    else:
        # Negative frames: translate reverse complement strand
        rev_comp = seq.reverse_complement()
        if frame == -1:
            # Frame -1: start at position 0 of reverse complement
            coding_seq = rev_comp
        elif frame == -2:
            # Frame -2: start at position 1 of reverse complement
            coding_seq = rev_comp[1:]
        elif frame == -3:
            # Frame -3: start at position 2 of reverse complement
            coding_seq = rev_comp[2:]
    
    # Translate to amino acids
    aa_seq = coding_seq.translate()
    return str(aa_seq)

def extract_coding_aa_sequences(input_file, parquet_file, output_bacterial, output_nonbacterial, 
                               bacterial_threshold=0.5, coding_threshold=0.5):
    """
    Extract amino acid sequences from coding sequences (bacterial and non-bacterial).
    
    Args:
        input_file: Path to original sequence file (FASTA/FASTQ)
        parquet_file: Path to BBERT parquet results
        output_bacterial: Path to output FASTA file for bacterial coding sequences
        output_nonbacterial: Path to output FASTA file for non-bacterial coding sequences
        bacterial_threshold: Minimum probability for bacterial classification
        coding_threshold: Minimum probability for coding classification
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Read BBERT results
    logger.info(f"Reading BBERT results from {parquet_file}")
    try:
        df = pd.read_parquet(parquet_file, columns=['id', 'bact_prob', 'frame_prob', 'coding_prob'])
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")
        return False
    
    if df.empty:
        logger.error("Parquet file is empty")
        return False
    
    logger.info(f"Total reads in results: {len(df)}")
    
    # Filter for coding sequences first
    coding_mask = df['coding_prob'] >= coding_threshold
    coding_df = df[coding_mask].copy()
    logger.info(f"Coding reads (prob >= {coding_threshold}): {len(coding_df)}")
    
    if coding_df.empty:
        logger.warning("No coding reads found with current thresholds")
        # Create empty output files
        with open(output_bacterial, 'w') as f:
            pass
        with open(output_nonbacterial, 'w') as f:
            pass
        return True
    
    # Split coding sequences into bacterial and non-bacterial
    bacterial_mask = coding_df['bact_prob'] >= bacterial_threshold
    coding_bacterial_df = coding_df[bacterial_mask].copy()
    coding_nonbacterial_df = coding_df[~bacterial_mask].copy()
    
    logger.info(f"Bacterial coding reads (bact_prob >= {bacterial_threshold}): {len(coding_bacterial_df)}")
    logger.info(f"Non-bacterial coding reads (bact_prob < {bacterial_threshold}): {len(coding_nonbacterial_df)}")
    
    # Create sets of target sequence IDs for fast lookup
    bacterial_ids = set(coding_bacterial_df['id'].values) if not coding_bacterial_df.empty else set()
    nonbacterial_ids = set(coding_nonbacterial_df['id'].values) if not coding_nonbacterial_df.empty else set()
    all_target_ids = bacterial_ids | nonbacterial_ids
    
    # Read original sequences and extract target sequences
    logger.info(f"Reading original sequences from {input_file}")
    open_func, file_format = detect_file_format(input_file)
    
    # Create ID to predictions mapping for all coding sequences
    id_to_predictions = {}
    for _, row in coding_df.iterrows():
        id_to_predictions[row['id']] = {
            'bact_prob': row['bact_prob'],
            'coding_prob': row['coding_prob'],
            'frame_probs': row['frame_prob']
        }
    
    bacterial_aa_sequences = []
    nonbacterial_aa_sequences = []
    sequences_found = 0
    
    with open_func(input_file) as handle:
        for record in SeqIO.parse(handle, file_format):
            if record.id in all_target_ids:
                sequences_found += 1
                predictions = id_to_predictions[record.id]
                
                # Get best reading frame
                best_frame = get_best_reading_frame(predictions['frame_probs'])
                
                # Translate to amino acids
                try:
                    aa_seq = translate_sequence(str(record.seq), best_frame)
                    
                    # Create FASTA header with prediction info (no frame)
                    header = (f"{record.id} | bact_prob={predictions['bact_prob']:.3f} | "
                             f"coding_prob={predictions['coding_prob']:.3f}")
                    
                    aa_record = SeqRecord(Seq(aa_seq), id=record.id, description=header)
                    
                    # Add to appropriate list based on bacterial classification
                    if record.id in bacterial_ids:
                        bacterial_aa_sequences.append(aa_record)
                    else:
                        nonbacterial_aa_sequences.append(aa_record)
                    
                except Exception as e:
                    logger.warning(f"Error translating sequence {record.id}: {e}")
                    continue
                
                if sequences_found % 1000 == 0:
                    logger.info(f"Processed {sequences_found} sequences")
    
    logger.info(f"Found {sequences_found} target sequences in input file")
    logger.info(f"Successfully translated {len(bacterial_aa_sequences)} bacterial and {len(nonbacterial_aa_sequences)} non-bacterial sequences")
    
    # Write bacterial amino acid sequences to FASTA
    logger.info(f"Writing bacterial amino acid sequences to {output_bacterial}")
    try:
        with open(output_bacterial, 'w') as handle:
            SeqIO.write(bacterial_aa_sequences, handle, "fasta")
        logger.info(f"Successfully wrote {len(bacterial_aa_sequences)} bacterial amino acid sequences")
    except Exception as e:
        logger.error(f"Error writing bacterial output file: {e}")
        return False
    
    # Write non-bacterial amino acid sequences to FASTA
    logger.info(f"Writing non-bacterial amino acid sequences to {output_nonbacterial}")
    try:
        with open(output_nonbacterial, 'w') as handle:
            SeqIO.write(nonbacterial_aa_sequences, handle, "fasta")
        logger.info(f"Successfully wrote {len(nonbacterial_aa_sequences)} non-bacterial amino acid sequences")
    except Exception as e:
        logger.error(f"Error writing non-bacterial output file: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Extract amino acid sequences from coding sequences (bacterial and non-bacterial)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic usage
  python source/extract_coding_AA.py --input example/sample.fasta --parquet results/sample_scores_len.parquet --out_bact bacterial_proteins.fasta --out_nonbact nonbacterial_proteins.fasta
  
  # Custom thresholds
  python source/extract_coding_AA.py --input data.fastq.gz --parquet results/data_scores_len.parquet --out_bact bact_proteins.fasta --out_nonbact nonbact_proteins.fasta --bacterial_threshold 0.8 --coding_threshold 0.7

OUTPUT FORMAT:
  Two FASTA files with amino acid sequences. Headers include:
  - Original sequence ID
  - Bacterial probability
  - Coding probability
        """
    )
    
    parser.add_argument("--input", required=True, 
                       help="Path to original sequence file (FASTA/FASTQ, compressed or uncompressed)")
    parser.add_argument("--parquet", required=True, 
                       help="Path to BBERT parquet results file")
    parser.add_argument("--out_bact", required=True, 
                       help="Path to output amino acid FASTA file for bacterial coding sequences")
    parser.add_argument("--out_nonbact", required=True, 
                       help="Path to output amino acid FASTA file for non-bacterial coding sequences")
    parser.add_argument("--bacterial_threshold", type=float, default=0.5,
                       help="Minimum bacterial probability threshold (default: 0.5)")
    parser.add_argument("--coding_threshold", type=float, default=0.5,
                       help="Minimum coding probability threshold (default: 0.5)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    if not os.path.exists(args.parquet):
        logger.error(f"Parquet file not found: {args.parquet}")
        return 1
    
    # Validate thresholds
    if not (0.0 <= args.bacterial_threshold <= 1.0):
        logger.error(f"Bacterial threshold must be between 0.0 and 1.0, got: {args.bacterial_threshold}")
        return 1
    
    if not (0.0 <= args.coding_threshold <= 1.0):
        logger.error(f"Coding threshold must be between 0.0 and 1.0, got: {args.coding_threshold}")
        return 1
    
    # Create output directories if needed
    for output_file in [args.out_bact, args.out_nonbact]:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Extract coding AA sequences
    logger.info("Starting coding amino acid extraction")
    logger.info(f"Bacterial threshold: {args.bacterial_threshold}")
    logger.info(f"Coding threshold: {args.coding_threshold}")
    
    if extract_coding_aa_sequences(args.input, args.parquet, args.out_bact, args.out_nonbact,
                                 args.bacterial_threshold, args.coding_threshold):
        logger.info("Successfully completed")
        return 0
    else:
        logger.error("Extraction failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())