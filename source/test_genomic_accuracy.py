#!/usr/bin/env python3
"""
Comprehensive genomic test for BBERT predictions.

Uses the generate_annotated_reads.py script for cleaner data generation.

The test generates reads from both coding (CDS from GFF/GTF) and non-coding 
(intergenic) regions of a genome, runs BBERT inference, and validates predictions.
"""

import os
import sys
import subprocess
import argparse
import logging
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def run_command(cmd, description, logger, check=True):
    """Run a shell command with logging."""
    logger.info(f"{description}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout and logger.level <= logging.DEBUG:
            logger.debug(f"STDOUT: {result.stdout}")
        if result.stderr and result.stderr.strip():
            if check:
                logger.error(f"STDERR: {result.stderr}")
            else:
                logger.warning(f"STDERR: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise

def generate_test_reads(fasta_file, annotation_file, is_bacterial, taxon, 
                       reads_per_cds=20, noncoding_reads=-1, read_length=100, 
                       output_dir=".", logger=None):
    """
    Generate test reads using the generate_annotated_reads.py script.
    
    Args:
        fasta_file: Path to genome FASTA file
        annotation_file: Path to annotation file (GFF/GTF)
        is_bacterial: Boolean, whether this is a bacterial genome
        taxon: Taxon name for output files
        reads_per_cds: Number of reads per CDS
        noncoding_reads: Number of noncoding reads (-1 for proportional)
        read_length: Length of each read
        output_dir: Directory for output files
        logger: Logger instance
        
    Returns:
        Tuple of (fasta_file, metadata_file) paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Determine annotation file type
    annotation_arg = "--gff" if annotation_file.lower().endswith('.gff') else "--gtf"
    
    # Prepare command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_script = os.path.join(script_dir, "generate_annotated_reads.py")
    
    output_prefix = f"genomic_test_reads_{taxon}"
    
    cmd = [
        sys.executable, generate_script,
        "--fasta", fasta_file,
        annotation_arg, annotation_file,
        "--is_bact", "true" if is_bacterial else "false",
        "--output_prefix", output_prefix,
        "--reads_per_cds", str(reads_per_cds),
        "--noncoding_reads", str(noncoding_reads),
        "--read_length", str(read_length),
        "--output_dir", output_dir
    ]
    
    # Run read generation
    result = run_command(cmd, "Generating test reads", logger)
    
    # Return paths to generated files
    fasta_output = os.path.join(output_dir, f"{output_prefix}_reads.fasta")
    metadata_output = os.path.join(output_dir, f"{output_prefix}_metadata.csv")
    
    # Verify files were created
    if not os.path.exists(fasta_output):
        raise FileNotFoundError(f"Expected FASTA output not found: {fasta_output}")
    if not os.path.exists(metadata_output):
        raise FileNotFoundError(f"Expected metadata output not found: {metadata_output}")
    
    logger.info(f"Successfully generated test reads:")
    logger.info(f"  FASTA: {fasta_output}")
    logger.info(f"  Metadata: {metadata_output}")
    
    return fasta_output, metadata_output

def run_bbert_inference(fasta_file, output_dir, logger, batch_size=128):
    """
    Run BBERT inference on the test reads.
    
    Args:
        fasta_file: Path to FASTA file with test reads
        output_dir: Directory for BBERT output
        logger: Logger instance
        batch_size: Batch size for inference
        
    Returns:
        Path to BBERT results parquet file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(script_dir, "inference.py")
    
    cmd = [
        sys.executable, inference_script,
        fasta_file,
        "--output_dir", output_dir,
        "--batch_size", str(batch_size)
    ]
    
    # Run BBERT inference
    result = run_command(cmd, "Running BBERT inference", logger)
    
    # Find the output parquet file
    fasta_basename = os.path.basename(fasta_file)
    expected_output = os.path.join(output_dir, fasta_basename.replace('.fasta', '_scores_len.parquet'))
    
    if not os.path.exists(expected_output):
        # Try alternative naming
        expected_output = os.path.join(output_dir, f"{os.path.splitext(fasta_basename)[0]}_scores_len.parquet")
        
    if not os.path.exists(expected_output):
        raise FileNotFoundError(f"BBERT results not found. Expected: {expected_output}")
    
    logger.info(f"BBERT inference completed: {expected_output}")
    return expected_output

def analyze_predictions(metadata_file, bbert_results_file, is_bacterial, logger):
    """
    Analyze BBERT predictions against ground truth from metadata.
    
    Args:
        metadata_file: Path to CSV metadata file with ground truth
        bbert_results_file: Path to BBERT results parquet file
        is_bacterial: Boolean, whether this is a bacterial genome
        logger: Logger instance
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing BBERT predictions")
    
    # Load ground truth metadata
    metadata_df = pd.read_csv(metadata_file)
    logger.info(f"Loaded metadata for {len(metadata_df)} reads")
    
    # Load BBERT results
    bbert_df = pd.read_parquet(bbert_results_file)
    logger.info(f"Loaded BBERT results for {len(bbert_df)} reads")
    
    # Merge on read ID
    # Extract read ID from BBERT results (remove file path prefix if present)
    bbert_df['read_id'] = bbert_df['id'].str.split('/').str[-1].str.replace('.fasta:', '')
    metadata_df['read_id'] = metadata_df['read_id']
    
    # Merge dataframes
    df = pd.merge(metadata_df, bbert_df, on='read_id', how='inner')
    logger.info(f"Merged {len(df)} reads for analysis")
    
    if df.empty:
        raise ValueError("No matching reads found between metadata and BBERT results")
    
    # Separate coding and noncoding reads
    coding_df = df[df['is_coding'] == True].copy()
    noncoding_df = df[df['is_coding'] == False].copy()
    
    total_reads = len(df)
    coding_reads = len(coding_df)
    noncoding_reads = len(noncoding_df)
    
    logger.info(f"Analysis dataset: {total_reads} total ({coding_reads} coding, {noncoding_reads} noncoding)")
    
    # 1. Coding/Noncoding classification analysis
    # Ground truth: True for coding, False for noncoding
    true_coding = df['is_coding'].values
    pred_coding = (df['coding_prob'] >= 0.5).values
    
    coding_correct = (true_coding == pred_coding).sum()
    coding_accuracy = coding_correct / total_reads if total_reads > 0 else 0
    
    # Per-type accuracies
    if not coding_df.empty:
        coding_predicted_correctly = (coding_df['coding_prob'] >= 0.5).sum()
        coding_type_accuracy = coding_predicted_correctly / len(coding_df)
    else:
        coding_predicted_correctly = 0
        coding_type_accuracy = 0
        
    if not noncoding_df.empty:
        noncoding_predicted_correctly = (noncoding_df['coding_prob'] < 0.5).sum()
        noncoding_type_accuracy = noncoding_predicted_correctly / len(noncoding_df)
    else:
        noncoding_predicted_correctly = 0
        noncoding_type_accuracy = 0
    
    # 2. Reading frame prediction analysis (coding sequences only)
    frame_correct = 0
    frame_accuracy = 0
    
    if not coding_df.empty:
        # BBERT frame mapping: positions 0-5 correspond to frames [-1, -3, -2, +1, +3, +2]
        frame_mapping = [-1, -3, -2, +1, +3, +2]
        
        def get_predicted_frame(frame_prob_array):
            return frame_mapping[np.argmax(frame_prob_array)]
        
        coding_df['predicted_frame'] = coding_df['frame_prob'].apply(get_predicted_frame)
        
        # Compare predicted vs true frames directly
        frame_matches = (coding_df['predicted_frame'] == coding_df['true_frame']).sum()
        frame_correct = frame_matches
        frame_accuracy = frame_matches / len(coding_df)
    
    # 3. Bacterial/Nonbacterial classification analysis
    true_bacterial = df['is_bacterial'].values
    pred_bacterial = (df['bact_prob'] >= 0.5).values
    
    bacterial_correct = (true_bacterial == pred_bacterial).sum()
    bacterial_accuracy = bacterial_correct / total_reads if total_reads > 0 else 0
    
    # Per-type bacterial classification
    if not coding_df.empty:
        if is_bacterial:
            bacterial_coding_correct = (coding_df['bact_prob'] >= 0.5).sum()
        else:
            bacterial_coding_correct = (coding_df['bact_prob'] < 0.5).sum()
        bacterial_coding_accuracy = bacterial_coding_correct / len(coding_df)
    else:
        bacterial_coding_correct = 0
        bacterial_coding_accuracy = 0
        
    if not noncoding_df.empty:
        if is_bacterial:
            bacterial_noncoding_correct = (noncoding_df['bact_prob'] >= 0.5).sum()
        else:
            bacterial_noncoding_correct = (noncoding_df['bact_prob'] < 0.5).sum()
        bacterial_noncoding_accuracy = bacterial_noncoding_correct / len(noncoding_df)
    else:
        bacterial_noncoding_correct = 0
        bacterial_noncoding_accuracy = 0
    
    # Overall statistics
    overall_coding_correct = coding_predicted_correctly + noncoding_predicted_correctly
    overall_coding_accuracy = overall_coding_correct / total_reads if total_reads > 0 else 0
    
    # Compile results
    results = {
        'total_reads': total_reads,
        'coding_reads': coding_reads,
        'noncoding_reads': noncoding_reads,
        'coding_accuracy': coding_type_accuracy,
        'coding_correct': coding_predicted_correctly,
        'noncoding_accuracy': noncoding_type_accuracy,
        'noncoding_correct': noncoding_predicted_correctly,
        'overall_coding_accuracy': overall_coding_accuracy,
        'overall_coding_correct': overall_coding_correct,
        'frame_accuracy': frame_accuracy,
        'frame_correct': frame_correct,
        'bacterial_accuracy': bacterial_accuracy,
        'bacterial_correct': bacterial_correct,
        'bacterial_coding_accuracy': bacterial_coding_accuracy,
        'bacterial_coding_correct': bacterial_coding_correct,
        'bacterial_noncoding_accuracy': bacterial_noncoding_accuracy,
        'bacterial_noncoding_correct': bacterial_noncoding_correct,
        'is_bacterial': is_bacterial,
        'mean_bacterial_prob': df['bact_prob'].mean(),
        'coding_mean_bacterial_prob': coding_df['bact_prob'].mean() if not coding_df.empty else 0,
        'noncoding_mean_bacterial_prob': noncoding_df['bact_prob'].mean() if not noncoding_df.empty else 0,
        'mean_coding_prob': df['coding_prob'].mean(),
        'coding_mean_coding_prob': coding_df['coding_prob'].mean() if not coding_df.empty else 0,
        'noncoding_mean_coding_prob': noncoding_df['coding_prob'].mean() if not noncoding_df.empty else 0,
    }
    
    return results

def print_detailed_results(results, taxon_name):
    """Print detailed test results."""
    
    print("\n" + "="*80)
    print(f"BBERT GENOMIC TEST RESULTS - {taxon_name}")
    print("="*80)
    
    total = results['total_reads']
    coding_total = results['coding_reads']
    noncoding_total = results['noncoding_reads']
    
    print(f"Total test reads: {total}")
    print(f"  Coding reads: {coding_total}")
    print(f"  Non-coding reads: {noncoding_total}")
    print()
    
    print("SEQUENCE TYPE CLASSIFICATION:")
    print(f"  Coding prediction:     {results['coding_correct']}/{coding_total} ({results['coding_accuracy']*100:.1f}%)")
    print(f"  Non-coding prediction: {results['noncoding_correct']}/{noncoding_total} ({results['noncoding_accuracy']*100:.1f}%)")
    print(f"  Overall coding/non-coding: {results['overall_coding_correct']}/{total} ({results['overall_coding_accuracy']*100:.1f}%)")
    print()
    
    print("READING FRAME PREDICTION (coding sequences only):")
    print(f"  Frame accuracy: {results['frame_correct']}/{coding_total} ({results['frame_accuracy']*100:.1f}%)")
    print()
    
    organism_type = "Bacterial" if results['is_bacterial'] else "Non-bacterial"
    print(f"{organism_type.upper()} CLASSIFICATION:")
    print(f"  {organism_type} prediction (overall): {results['bacterial_correct']}/{total} ({results['bacterial_accuracy']*100:.1f}%)")
    print(f"    Coding sequences:     {results['bacterial_coding_correct']}/{coding_total} ({results['bacterial_coding_accuracy']*100:.1f}%)")
    print(f"    Non-coding sequences: {results['bacterial_noncoding_correct']}/{noncoding_total} ({results['bacterial_noncoding_accuracy']*100:.1f}%)")
    print()
    
    print("PROBABILITY DISTRIBUTIONS:")
    print(f"  Mean bacterial probability (all): {results['mean_bacterial_prob']:.3f}")
    print(f"  Mean bacterial probability (coding seqs): {results['coding_mean_bacterial_prob']:.3f}")
    print(f"  Mean bacterial probability (non-coding seqs): {results['noncoding_mean_bacterial_prob']:.3f}")
    print(f"  Mean coding probability (all): {results['mean_coding_prob']:.3f}")
    print(f"  Mean coding probability (coding seqs): {results['coding_mean_coding_prob']:.3f}")
    print(f"  Mean coding probability (non-coding seqs): {results['noncoding_mean_coding_prob']:.3f}")
    print()

def print_summary_stats(results, taxon_name):
    """Print tab-separated summary statistics for easy parsing."""
    print("SUMMARY_STATS\t" + "\t".join([
        taxon_name,
        str(results['total_reads']),
        str(results['coding_reads']),
        str(results['noncoding_reads']),
        f"{results['coding_accuracy']:.4f}",
        f"{results['noncoding_accuracy']:.4f}",
        f"{results['overall_coding_accuracy']:.4f}",
        f"{results['frame_accuracy']:.4f}",
        f"{results['bacterial_accuracy']:.4f}",
        f"{results['bacterial_coding_accuracy']:.4f}",
        f"{results['bacterial_noncoding_accuracy']:.4f}",
        f"{results['mean_bacterial_prob']:.4f}",
        f"{results['mean_coding_prob']:.4f}",
        str(results['is_bacterial']).lower()
    ]))

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive genomic test for BBERT predictions (Version 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
This test generates reads from both coding (CDS from GFF/GTF) and non-coding 
(intergenic) regions of a genome, runs BBERT inference, and validates predictions.

The test evaluates:
1. Coding vs non-coding classification accuracy
2. Reading frame prediction accuracy for coding sequences  
3. Bacterial vs non-bacterial classification accuracy

This version uses generate_annotated_reads.py for cleaner data generation.

EXAMPLES:
  # Bacterial genome test
  python source/test_genomic_accuracy2.py --fasta genome.fasta --gff annotations.gff --is_bact true --taxon "E.coli"
  
  # Eukaryotic genome test  
  python source/test_genomic_accuracy2.py --fasta genome.fasta --gtf annotations.gtf --is_bact false --taxon "S.cerevisiae"
  
  # Using provided test files
  python source/test_genomic_accuracy2.py --fasta example/GCF_000146045.fasta --gff example/GCF_000146045.gff --is_bact false --taxon "S.cerevisiae"

OUTPUT:
  Detailed results followed by tab-separated summary line starting with "SUMMARY_STATS"
  
FILES GENERATED:
  All files are saved to --output_dir (default: current directory):
  1. {taxon}_reads.fasta - Test sequences with annotated headers
  2. {taxon}_metadata.csv - Ground truth labels and coordinates  
  3. {taxon}_reads_scores_len.parquet - BBERT prediction results
  
  Use --cleanup to delete files after completion (default: preserve files)
        """
    )
    
    parser.add_argument("--fasta", required=True,
                       help="Path to genome FASTA file")
    parser.add_argument("--gff", 
                       help="Path to GFF annotation file")
    parser.add_argument("--gtf",
                       help="Path to GTF annotation file")
    parser.add_argument("--is_bact", required=True, choices=['true', 'false'],
                       help="Whether this is a bacterial genome (true/false)")
    parser.add_argument("--taxon", required=True,
                       help="Taxon name for output")
    parser.add_argument("--reads_per_cds", type=int, default=20,
                       help="Number of reads per CDS (coding sequences only) (default: 20)")
    parser.add_argument("--noncoding_reads", type=int, default=-1,
                       help="Number of non-coding reads (default: -1 = calculate proportionally based on genome composition)")
    parser.add_argument("--read_length", type=int, default=100,
                       help="Length of each test read (default: 100)")
    parser.add_argument("--output_dir", default=".",
                       help="Directory for output files (default: current directory)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Delete generated test files after completion (default: keep files)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.fasta):
        logger.error(f"FASTA file not found: {args.fasta}")
        return 1
    
    # Check for annotation file
    annotation_file = None
    if args.gff:
        if not os.path.exists(args.gff):
            logger.error(f"GFF file not found: {args.gff}")
            return 1
        annotation_file = args.gff
    elif args.gtf:
        if not os.path.exists(args.gtf):
            logger.error(f"GTF file not found: {args.gtf}")
            return 1
        annotation_file = args.gtf
    else:
        logger.error("Either --gff or --gtf must be provided")
        return 1
    
    # Parse bacterial flag
    is_bacterial = args.is_bact.lower() == 'true'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting BBERT genomic accuracy test (Version 2)")
    logger.info(f"Genome: {args.fasta}")
    logger.info(f"Annotations: {annotation_file}")
    logger.info(f"Is bacterial: {is_bacterial}")
    logger.info(f"Taxon: {args.taxon}")
    logger.info(f"Reads per CDS: {args.reads_per_cds}")
    logger.info(f"Noncoding reads: {args.noncoding_reads}")
    logger.info(f"Read length: {args.read_length}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Generate test reads
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Generating test reads")
        logger.info("="*50)
        
        fasta_file, metadata_file = generate_test_reads(
            args.fasta, annotation_file, is_bacterial, args.taxon,
            reads_per_cds=args.reads_per_cds,
            noncoding_reads=args.noncoding_reads,
            read_length=args.read_length,
            output_dir=str(output_dir),
            logger=logger
        )
        
        # Step 2: Run BBERT inference
        logger.info("\n" + "="*50)
        logger.info("STEP 2: Running BBERT inference")
        logger.info("="*50)
        
        bbert_results_file = run_bbert_inference(fasta_file, str(output_dir), logger)
        
        # Step 3: Analyze predictions
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Analyzing predictions")
        logger.info("="*50)
        
        results = analyze_predictions(metadata_file, bbert_results_file, is_bacterial, logger)
        
        # Step 4: Print results
        print_detailed_results(results, args.taxon)
        print_summary_stats(results, args.taxon)
        
        # Report generated files
        logger.info(f"\nGenerated files in {output_dir}:")
        logger.info(f"  Test reads (FASTA): {fasta_file}")
        logger.info(f"    - Contains {results['total_reads']} DNA sequences with annotated headers")
        logger.info(f"    - {results['coding_reads']} coding sequences from CDS regions")  
        logger.info(f"    - {results['noncoding_reads']} noncoding sequences from intergenic regions")
        logger.info(f"  Ground truth metadata (CSV): {metadata_file}")
        logger.info(f"    - Contains true labels: coding/noncoding, bacterial/nonbacterial, reading frames")
        logger.info(f"    - Includes sequence coordinates and annotation details")
        logger.info(f"  BBERT predictions (Parquet): {bbert_results_file}")
        logger.info(f"    - Contains BBERT probability outputs: bact_prob, coding_prob, frame_prob")
        logger.info(f"    - Ready for further analysis or comparison with ground truth")
        
        # Cleanup only if explicitly requested
        if args.cleanup:
            logger.info("\nCleaning up generated files...")
            try:
                os.unlink(fasta_file)
                os.unlink(metadata_file)
                os.unlink(bbert_results_file)
                logger.info("Files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Could not clean up some files: {e}")
        else:
            logger.info(f"\nFiles preserved for further analysis (use --cleanup to delete)")
        
        logger.info("\nTest completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())