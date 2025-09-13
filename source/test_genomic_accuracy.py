#!/usr/bin/env python3
"""
Comprehensive genomic test for BBERT's predictions on coding and non-coding sequences.

This script takes a genome FASTA file and corresponding GFF/GTF annotation file,
generates test reads from both coding (CDS) and non-coding (intergenic) regions,
runs BBERT inference, and validates the predictions against ground truth.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import subprocess
import gzip
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import re

# Add BBERT source to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GenomicAnnotation:
    """Class to handle GFF/GTF parsing and genomic coordinate management."""
    
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.cds_features = []
        self.parse_annotations()
    
    def parse_annotations(self):
        """Parse GFF/GTF file to extract CDS features."""
        print(f"Parsing annotations from {self.annotation_file}")
        
        open_func = gzip.open if self.annotation_file.endswith('.gz') else open
        mode = 'rt' if self.annotation_file.endswith('.gz') else 'r'
        
        with open_func(self.annotation_file, mode) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) < 9:
                        continue
                    
                    seqid, source, feature_type, start, end, score, strand, phase, attributes = parts
                    
                    # Only process CDS features
                    if feature_type.upper() != 'CDS':
                        continue
                    
                    # Convert coordinates to 0-based (GFF is 1-based)
                    start_pos = int(start) - 1
                    end_pos = int(end)
                    
                    self.cds_features.append({
                        'seqid': seqid,
                        'start': start_pos,
                        'end': end_pos,
                        'strand': strand,
                        'phase': int(phase) if phase != '.' else 0,
                        'attributes': attributes
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
                    continue
        
        print(f"Found {len(self.cds_features)} CDS features")
        
        # Sort CDS features by sequence ID and position
        self.cds_features.sort(key=lambda x: (x['seqid'], x['start']))
    
    def get_coding_regions(self, seqid):
        """Get all CDS regions for a given sequence ID."""
        return [cds for cds in self.cds_features if cds['seqid'] == seqid]
    
    def get_intergenic_regions(self, seqid, seq_length, min_length=200):
        """Get intergenic regions (non-coding) for a given sequence."""
        coding_regions = self.get_coding_regions(seqid)
        if not coding_regions:
            # If no CDS, treat entire sequence as intergenic
            return [{'start': 0, 'end': seq_length}]
        
        intergenic = []
        
        # Region before first CDS
        first_start = coding_regions[0]['start']
        if first_start >= min_length:
            intergenic.append({'start': 0, 'end': first_start})
        
        # Regions between CDS features
        for i in range(len(coding_regions) - 1):
            current_end = coding_regions[i]['end']
            next_start = coding_regions[i + 1]['start']
            
            if next_start - current_end >= min_length:
                intergenic.append({'start': current_end, 'end': next_start})
        
        # Region after last CDS
        last_end = coding_regions[-1]['end']
        if seq_length - last_end >= min_length:
            intergenic.append({'start': last_end, 'end': seq_length})
        
        return intergenic

def setup_test_environment():
    """Check if we're in the right directory and have required files."""
    if not Path('bbert.py').exists():
        raise FileNotFoundError("bbert.py not found. Please ensure you're running from BBERT root directory.")
    return True

def generate_coding_reads(genome_seqs, annotations, read_length=100, reads_per_cds=20):
    """
    Generate test reads from coding sequences (CDS) in all 6 reading frames (like real sequencing).
    Ground truth frame is determined from GFF annotations for evaluation only.
    
    Args:
        genome_seqs: Dictionary of {seqid: sequence_string}
        annotations: GenomicAnnotation object
        read_length: Length of each read
        reads_per_cds: Number of reads per CDS (distributed across all 6 frames)
    
    Returns:
        tuple: (list of SeqRecord objects with ground truth labels, dict with statistics)
    """
    all_reads = []
    total_cds = 0
    short_cds_count = 0
    no_positions_count = 0
    used_cds_count = 0
    
    for seqid, seq_str in genome_seqs.items():
        coding_regions = annotations.get_coding_regions(seqid)
        
        for cds in coding_regions:
            total_cds += 1
            cds_seq = seq_str[cds['start']:cds['end']]
            cds_strand = cds['strand']
            cds_phase = cds['phase']
            
            # Determine the correct reading frame for ground truth (based on strand and phase)
            if cds_strand == '+':
                correct_frame = cds_phase + 1  # phase 0 → frame +1, phase 1 → frame +2, phase 2 → frame +3
            else:
                correct_frame = -(cds_phase + 1)  # phase 0 → frame -1, phase 1 → frame -2, phase 2 → frame -3
            
            # Check if CDS is long enough
            if len(cds_seq) < read_length:
                short_cds_count += 1
                continue
            
            # Generate reads that are specifically in different frames
            cds_generated_reads = 0
            
            # Randomly select frames for this CDS based on reads_per_cds
            all_frames = [1, 2, 3, -1, -2, -3]
            if reads_per_cds >= 6:
                # If reads_per_cds >= 6, distribute evenly across all frames
                reads_per_frame = reads_per_cds // 6
                remaining_reads = reads_per_cds - (reads_per_frame * 6)
                selected_frames = []
                for frame in all_frames:
                    num_reads_this_frame = reads_per_frame
                    if remaining_reads > 0:
                        num_reads_this_frame += 1
                        remaining_reads -= 1
                    for _ in range(num_reads_this_frame):
                        selected_frames.append(frame)
            else:
                # If reads_per_cds < 6, randomly select frames
                selected_frames = random.choices(all_frames, k=reads_per_cds)
            
            # Generate reads for the selected frames
            for read_idx, target_frame in enumerate(selected_frames):
                # Generate read that is specifically in the target frame
                # Use the correct logic from generate_coding_reads.py
                # Get sense sequence (coding orientation)
                sense = cds_seq if cds_strand == '+' else str(Seq(cds_seq).reverse_complement())
                
                # Pick position in sense so (phase - pos) % 3 == |frame| - 1
                f = abs(target_frame)
                desired_offset = (cds_phase - (f - 1)) % 3
                
                # Find valid positions in sense sequence
                valid_positions = []
                for pos in range(len(sense) - read_length + 1):
                    if pos % 3 == desired_offset:
                        valid_positions.append(pos)
                
                if not valid_positions:
                    continue
                    
                read_start = random.choice(valid_positions)
                frag = sense[read_start:read_start + read_length]
                
                # Emit fragment as-is for positive frames; reverse complement for negative frames
                read_seq = frag if target_frame > 0 else str(Seq(frag).reverse_complement())
                
                if len(read_seq) == read_length:  # Only add if we got a full-length read
                    read_id = f"coding_{seqid}_{cds['start']}_{cds['end']}_frame{target_frame:+d}"
                    read_record = SeqRecord(
                        Seq(read_seq),
                        id=read_id,
                        description=f""
                    )
                    
                    # Store ground truth metadata
                    read_record.ground_truth = {
                        'is_coding': True,
                        'true_frame': target_frame,  # Frame this read is actually in (ground truth)
                        'actual_frame': target_frame,  # Same as true_frame
                        'is_correct_frame': True,  # This read should be correctly identified by BBERT
                        'seqid': seqid,
                        'region_type': 'coding'
                    }
                    
                    all_reads.append(read_record)
                    cds_generated_reads += 1
            
            if cds_generated_reads > 0:
                used_cds_count += 1
            else:
                no_positions_count += 1
    
    # Count frame distribution for diagnostics
    frame_counts = defaultdict(int)
    for read in all_reads:
        frame_counts[read.ground_truth['actual_frame']] += 1
    
    stats = {
        'total_cds': total_cds,
        'used_cds': used_cds_count,
        'short_cds': short_cds_count,
        'no_positions_cds': no_positions_count,
        'frame_distribution': dict(frame_counts)
    }
    
    return all_reads, stats

def calculate_noncoding_reads_needed(genome_seqs, annotations, coding_reads_generated):
    """
    Calculate the number of noncoding reads needed based on genome composition.
    
    Args:
        genome_seqs: Dictionary of {seqid: sequence_string}
        annotations: GenomicAnnotation object
        coding_reads_generated: Number of coding reads that were generated
    
    Returns:
        int: Number of noncoding reads to generate
    """
    total_coding_length = 0
    total_noncoding_length = 0
    
    for seqid, seq_str in genome_seqs.items():
        seq_length = len(seq_str)
        
        # Get coding regions
        coding_regions = annotations.get_coding_regions(seqid)
        coding_length = sum(cds['end'] - cds['start'] + 1 for cds in coding_regions)
        
        # Get noncoding regions  
        intergenic_regions = annotations.get_intergenic_regions(seqid, seq_length)
        noncoding_length = sum(region['end'] - region['start'] for region in intergenic_regions)
        
        total_coding_length += coding_length
        total_noncoding_length += noncoding_length
    
    total_genome_length = total_coding_length + total_noncoding_length
    
    if total_coding_length == 0:
        return 0
    
    # Calculate proportional noncoding reads
    noncoding_ratio = total_noncoding_length / total_genome_length
    coding_ratio = total_coding_length / total_genome_length
    
    # Generate noncoding reads proportional to noncoding DNA
    noncoding_reads_needed = int(coding_reads_generated * (noncoding_ratio / coding_ratio))
    
    print(f"Genome composition:")
    print(f"  Total length: {total_genome_length:,} bp")
    print(f"  Coding DNA: {total_coding_length:,} bp ({coding_ratio*100:.1f}%)")
    print(f"  Non-coding DNA: {total_noncoding_length:,} bp ({noncoding_ratio*100:.1f}%)")
    print(f"  Proportional non-coding reads needed: {noncoding_reads_needed:,}")
    
    return noncoding_reads_needed

def generate_noncoding_reads(genome_seqs, annotations, read_length=100, reads_total=200):
    """
    Generate test reads from non-coding (intergenic) sequences.
    
    Args:
        genome_seqs: Dictionary of {seqid: sequence_string}
        annotations: GenomicAnnotation object
        read_length: Length of each read
        reads_total: Total number of non-coding reads to generate
    
    Returns:
        list: List of SeqRecord objects with ground truth labels
    """
    all_reads = []
    
    # Collect all intergenic regions
    all_intergenic = []
    for seqid, seq_str in genome_seqs.items():
        intergenic_regions = annotations.get_intergenic_regions(seqid, len(seq_str))
        for region in intergenic_regions:
            all_intergenic.append({
                'seqid': seqid,
                'start': region['start'],
                'end': region['end'],
                'length': region['end'] - region['start']
            })
    
    if not all_intergenic:
        print("Warning: No intergenic regions found")
        return all_reads
    
    # Weight regions by length for sampling
    total_intergenic_length = sum(r['length'] for r in all_intergenic)
    
    reads_generated = 0
    attempts = 0
    max_attempts = reads_total * 10
    
    while reads_generated < reads_total and attempts < max_attempts:
        attempts += 1
        
        # Select region proportional to its length
        rand_pos = random.randint(0, total_intergenic_length - 1)
        cumulative = 0
        selected_region = None
        
        for region in all_intergenic:
            cumulative += region['length']
            if rand_pos < cumulative:
                selected_region = region
                break
        
        if not selected_region:
            continue
        
        # Select random position within the region
        region_seq = genome_seqs[selected_region['seqid']][selected_region['start']:selected_region['end']]
        
        if len(region_seq) < read_length:
            continue
        
        max_start = len(region_seq) - read_length
        read_start = random.randint(0, max_start)
        read_seq = region_seq[read_start:read_start + read_length]
        
        read_id = f"noncoding_{selected_region['seqid']}_{selected_region['start']}_{selected_region['end']}"
        read_record = SeqRecord(
            Seq(read_seq),
            id=read_id,
            description=f""
        )
        
        # Store ground truth metadata
        read_record.ground_truth = {
            'is_coding': False,
            'true_frame': None,  # No meaningful frame for non-coding
            'is_correct_frame': False,
            'seqid': selected_region['seqid'],
            'region_type': 'noncoding'
        }
        
        all_reads.append(read_record)
        reads_generated += 1
    
    print(f"Generated {reads_generated} non-coding reads from {len(all_intergenic)} intergenic regions")
    return all_reads

def run_bbert_inference(input_file, output_dir, verbose=False):
    """
    Run BBERT inference on the test reads.
    
    Args:
        input_file: Path to test reads FASTA file
        output_dir: Directory for output files
        verbose: Whether to show verbose output from BBERT
    
    Returns:
        str: Path to output parquet file
    """
    print("Running BBERT inference...")
    
    # Count reads for progress tracking
    read_count = 0
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                read_count += 1
    
    if verbose:
        print(f"  Input file: {input_file}")
        print(f"  Output directory: {output_dir}")
        print(f"  Processing {read_count:,} reads")
        print(f"  Batch size: 128")
    
    # Construct command
    cmd = [sys.executable, 'bbert.py', input_file, '--output_dir', output_dir, '--batch_size', '128']
    
    if verbose:
        print(f"  Command: {' '.join(cmd)}")
        print("  Running BBERT...")
    
    try:
        if verbose:
            # Show BBERT output in real-time for verbose mode
            result = subprocess.run(cmd, text=True, check=True)
        else:
            # Capture output for non-verbose mode
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("BBERT inference completed successfully")
        
        # Determine output file name
        base_name = Path(input_file).stem
        output_file = Path(output_dir) / f"{base_name}_scores_len.parquet"
        
        if output_file.exists():
            if verbose:
                file_size = output_file.stat().st_size
                print(f"  Output file: {output_file}")
                print(f"  Output size: {file_size:,} bytes")
            return str(output_file)
        else:
            raise FileNotFoundError(f"Expected output file {output_file} not found")
            
    except subprocess.CalledProcessError as e:
        print(f"BBERT inference failed: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"stdout: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        raise

def extract_ground_truth_from_id(read_id):
    """Extract ground truth information from read ID."""
    if read_id.startswith('coding_'):
        parts = read_id.split('_')
        # Find the frame part (this is the actual frame the read was generated in)
        actual_frame_part = None
        for part in parts:
            if part.startswith('frame'):
                actual_frame_part = part
                break
        
        if actual_frame_part:
            actual_frame_str = actual_frame_part.replace('frame', '')
            return {
                'is_coding': True,
                'actual_frame': int(actual_frame_str),
                'region_type': 'coding'
            }
    elif read_id.startswith('noncoding_'):
        return {
            'is_coding': False,
            'actual_frame': None,
            'region_type': 'noncoding'
        }
    
    return None

def get_predicted_frame(frame_probs):
    """Convert BBERT frame probabilities to frame number."""
    best_idx = np.argmax(frame_probs)
    # Correct BBERT mapping: positions 0-5 correspond to frames [-1, -3, -2, +1, +3, +2]
    frame_mapping = [-1, -3, -2, +1, +3, +2]
    return frame_mapping[best_idx]

def analyze_genomic_predictions(parquet_file, test_reads_metadata, is_bacterial):
    """
    Analyze BBERT predictions against ground truth for genomic test.
    
    Args:
        parquet_file: Path to BBERT parquet results
        test_reads_metadata: Dictionary mapping read IDs to ground truth
        is_bacterial: Boolean indicating if this is a bacterial genome
    
    Returns:
        dict: Analysis results
    """
    print(f"Analyzing genomic predictions from {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    print(f"Available columns in parquet file: {list(df.columns)}")
    
    # Add ground truth information from the test_reads_metadata
    ground_truth_data = []
    for _, row in df.iterrows():
        read_id = row['id']
        if read_id is None or read_id not in test_reads_metadata:
            # Handle case where read_id is None or not found
            ground_truth_data.append({
                'is_coding': None,
                'true_frame': None,
                'actual_frame': None,
                'region_type': 'unknown'
            })
            continue
            
        # Get ground truth from metadata (which has the correct frame from GFF)
        gt = test_reads_metadata[read_id]
        ground_truth_data.append({
            'is_coding': gt['is_coding'],
            'true_frame': gt['true_frame'],  # Correct frame from GFF
            'actual_frame': gt.get('actual_frame', gt.get('true_frame')),  # Frame read was generated in
            'region_type': gt['region_type']
        })
    
    gt_df = pd.DataFrame(ground_truth_data)
    df = pd.concat([df, gt_df], axis=1)
    
    # Get predicted frames
    df['predicted_frame'] = df['frame_prob'].apply(get_predicted_frame)
    
    # Separate coding and non-coding results
    coding_df = df[df['region_type'] == 'coding'].copy()
    noncoding_df = df[df['region_type'] == 'noncoding'].copy()
    
    # Overall statistics
    total_reads = len(df)
    coding_reads = len(coding_df)
    noncoding_reads = len(noncoding_df)
    
    # Coding sequence analysis
    if not coding_df.empty:
        coding_predicted_correctly = (coding_df['coding_prob'] >= 0.5).sum()
        coding_accuracy = coding_predicted_correctly / len(coding_df)
        
        # Frame accuracy for coding sequences (compare predicted vs actual frame)
        coding_frame_correct = (coding_df['predicted_frame'] == coding_df['true_frame']).sum()
        coding_frame_accuracy = coding_frame_correct / len(coding_df) if len(coding_df) > 0 else 0
    else:
        coding_predicted_correctly = 0
        coding_accuracy = 0
        coding_frame_correct = 0
        coding_frame_accuracy = 0
    
    # Non-coding sequence analysis
    if not noncoding_df.empty:
        noncoding_predicted_correctly = (noncoding_df['coding_prob'] < 0.5).sum()
        noncoding_accuracy = noncoding_predicted_correctly / len(noncoding_df)
    else:
        noncoding_predicted_correctly = 0
        noncoding_accuracy = 0
    
    # Bacterial/non-bacterial classification - calculate both original and custom threshold results
    # Custom threshold: 1.3654 (values below = bacterial, above = non-bacterial)
    custom_threshold = 1.3654
    
    if is_bacterial:
        # Standard classification for overall accuracy
        bacterial_correct = (df['bact_prob'] >= 0.5).sum()
        bacterial_accuracy = bacterial_correct / total_reads
        
        # ORIGINAL: Standard threshold for all sequences
        if not coding_df.empty:
            bacterial_coding_correct_orig = (coding_df['bact_prob'] >= 0.5).sum()
            bacterial_coding_accuracy_orig = bacterial_coding_correct_orig / len(coding_df)
        else:
            bacterial_coding_correct_orig = 0
            bacterial_coding_accuracy_orig = 0
            
        if not noncoding_df.empty:
            bacterial_noncoding_correct_orig = (noncoding_df['bact_prob'] >= 0.5).sum()
            bacterial_noncoding_accuracy_orig = bacterial_noncoding_correct_orig / len(noncoding_df)
        else:
            bacterial_noncoding_correct_orig = 0
            bacterial_noncoding_accuracy_orig = 0
        
        # CUSTOM: Mixed threshold approach for problematic cases
        if not coding_df.empty:
            # For coding sequences in bacterial genome: use standard threshold
            bacterial_coding_correct = (coding_df['bact_prob'] >= 0.5).sum()
            bacterial_coding_accuracy = bacterial_coding_correct / len(coding_df)
        else:
            bacterial_coding_correct = 0
            bacterial_coding_accuracy = 0
            
        if not noncoding_df.empty:
            # For noncoding sequences in bacterial genome: use custom threshold (problematic case)
            # Use raw loss scores where < custom_threshold = bacterial
            bacterial_noncoding_correct = (noncoding_df['loss'] < custom_threshold).sum()
            bacterial_noncoding_accuracy = bacterial_noncoding_correct / len(noncoding_df)
        else:
            bacterial_noncoding_correct = 0
            bacterial_noncoding_accuracy = 0
    else:
        # Standard classification for overall accuracy  
        bacterial_correct = (df['bact_prob'] < 0.5).sum()
        bacterial_accuracy = bacterial_correct / total_reads
        
        # ORIGINAL: Standard threshold for all sequences
        if not coding_df.empty:
            bacterial_coding_correct_orig = (coding_df['bact_prob'] < 0.5).sum()
            bacterial_coding_accuracy_orig = bacterial_coding_correct_orig / len(coding_df)
        else:
            bacterial_coding_correct_orig = 0
            bacterial_coding_accuracy_orig = 0
            
        if not noncoding_df.empty:
            bacterial_noncoding_correct_orig = (noncoding_df['bact_prob'] < 0.5).sum()
            bacterial_noncoding_accuracy_orig = bacterial_noncoding_correct_orig / len(noncoding_df)
        else:
            bacterial_noncoding_correct_orig = 0
            bacterial_noncoding_accuracy_orig = 0
        
        # CUSTOM: Mixed threshold approach for problematic cases
        if not coding_df.empty:
            # For coding sequences in non-bacterial genome: use custom threshold (problematic case)
            # Use raw loss scores where >= custom_threshold = non-bacterial
            bacterial_coding_correct = (coding_df['loss'] >= custom_threshold).sum()
            bacterial_coding_accuracy = bacterial_coding_correct / len(coding_df)
        else:
            bacterial_coding_correct = 0
            bacterial_coding_accuracy = 0
            
        if not noncoding_df.empty:
            # For noncoding sequences in non-bacterial genome: use standard threshold
            bacterial_noncoding_correct = (noncoding_df['bact_prob'] < 0.5).sum()
            bacterial_noncoding_accuracy = bacterial_noncoding_correct / len(noncoding_df)
        else:
            bacterial_noncoding_correct = 0
            bacterial_noncoding_accuracy = 0
    
    # Overall coding prediction (both coding and non-coding)
    overall_coding_correct = coding_predicted_correctly + noncoding_predicted_correctly
    overall_coding_accuracy = overall_coding_correct / total_reads if total_reads > 0 else 0
    
    results = {
        'total_reads': total_reads,
        'coding_reads': coding_reads,
        'noncoding_reads': noncoding_reads,
        'coding_accuracy': coding_accuracy,
        'coding_correct': coding_predicted_correctly,
        'noncoding_accuracy': noncoding_accuracy,
        'noncoding_correct': noncoding_predicted_correctly,
        'overall_coding_accuracy': overall_coding_accuracy,
        'overall_coding_correct': overall_coding_correct,
        'frame_accuracy': coding_frame_accuracy,
        'frame_correct': coding_frame_correct,
        'bacterial_accuracy': bacterial_accuracy,
        'bacterial_correct': bacterial_correct,
        'bacterial_coding_accuracy': bacterial_coding_accuracy,
        'bacterial_coding_correct': bacterial_coding_correct,
        'bacterial_noncoding_accuracy': bacterial_noncoding_accuracy,
        'bacterial_noncoding_correct': bacterial_noncoding_correct,
        'bacterial_coding_accuracy_orig': bacterial_coding_accuracy_orig,
        'bacterial_coding_correct_orig': bacterial_coding_correct_orig,
        'bacterial_noncoding_accuracy_orig': bacterial_noncoding_accuracy_orig,
        'bacterial_noncoding_correct_orig': bacterial_noncoding_correct_orig,
        'is_bacterial': is_bacterial,
        'mean_bacterial_prob': df['bact_prob'].mean(),
        'coding_mean_bacterial_prob': coding_df['bact_prob'].mean() if not coding_df.empty else 0,
        'noncoding_mean_bacterial_prob': noncoding_df['bact_prob'].mean() if not noncoding_df.empty else 0,
        'mean_loss': df['loss'].mean(),
        'coding_mean_loss': coding_df['loss'].mean() if not coding_df.empty else 0,
        'noncoding_mean_loss': noncoding_df['loss'].mean() if not noncoding_df.empty else 0,
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
    print(f"  {organism_type} prediction (overall): {results['bacterial_correct']}/{total} ({results['bacterial_accuracy']*100:.1f}%) [standard 0.5 threshold]")
    print()
    
    print("  ORIGINAL APPROACH (standard bact_prob >= 0.5 threshold for all):")
    print(f"    Coding sequences:     {results['bacterial_coding_correct_orig']}/{coding_total} ({results['bacterial_coding_accuracy_orig']*100:.1f}%)")
    print(f"    Non-coding sequences: {results['bacterial_noncoding_correct_orig']}/{noncoding_total} ({results['bacterial_noncoding_accuracy_orig']*100:.1f}%)")
    print()
    
    print("  CUSTOM APPROACH (loss threshold 1.3654 for problematic cases):")
    print(f"    Coding sequences:     {results['bacterial_coding_correct']}/{coding_total} ({results['bacterial_coding_accuracy']*100:.1f}%) [{'standard 0.5' if results['is_bacterial'] else 'loss 1.3654'} threshold]")
    print(f"    Non-coding sequences: {results['bacterial_noncoding_correct']}/{noncoding_total} ({results['bacterial_noncoding_accuracy']*100:.1f}%) [{'loss 1.3654' if results['is_bacterial'] else 'standard 0.5'} threshold]")
    print()
    
    # Calculate improvement
    orig_combined = results['bacterial_coding_correct_orig'] + results['bacterial_noncoding_correct_orig']
    custom_combined = results['bacterial_coding_correct'] + results['bacterial_noncoding_correct']
    improvement = custom_combined - orig_combined
    print(f"  IMPROVEMENT: {improvement:+d} more correct predictions with custom approach")
    print()
    
    print("PROBABILITY DISTRIBUTIONS:")
    print(f"  Mean bacterial probability (all): {results['mean_bacterial_prob']:.3f}")
    print(f"  Mean bacterial probability (coding seqs): {results['coding_mean_bacterial_prob']:.3f}")
    print(f"  Mean bacterial probability (non-coding seqs): {results['noncoding_mean_bacterial_prob']:.3f}")
    print(f"  Mean coding probability (all): {results['mean_coding_prob']:.3f}")
    print(f"  Mean coding probability (coding seqs): {results['coding_mean_coding_prob']:.3f}")
    print(f"  Mean coding probability (non-coding seqs): {results['noncoding_mean_coding_prob']:.3f}")
    print()
    print("LOSS SCORE DISTRIBUTIONS (for custom threshold 1.3654):")
    print(f"  Mean loss (all): {results['mean_loss']:.3f}")
    print(f"  Mean loss (coding seqs): {results['coding_mean_loss']:.3f}")
    print(f"  Mean loss (non-coding seqs): {results['noncoding_mean_loss']:.3f}")
    print()

def print_summary_stats(results, taxon_name):
    """Print tab-separated summary statistics."""
    
    # Tab-separated final line with key statistics
    summary = [
        taxon_name,
        "bacterial" if results['is_bacterial'] else "non-bacterial",
        str(results['total_reads']),
        str(results['coding_reads']),
        str(results['noncoding_reads']),
        f"{results['overall_coding_accuracy']:.3f}",
        f"{results['coding_accuracy']:.3f}",
        f"{results['noncoding_accuracy']:.3f}",
        f"{results['frame_accuracy']:.3f}",
        f"{results['bacterial_accuracy']:.3f}",
        f"{results['mean_bacterial_prob']:.3f}",
        f"{results['mean_coding_prob']:.3f}"
    ]
    
    print("SUMMARY_STATS\t" + "\t".join(summary))

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive genomic test for BBERT predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
This test generates reads from both coding (CDS from GFF/GTF) and non-coding 
(intergenic) regions of a genome, runs BBERT inference, and validates predictions.

The test evaluates:
1. Coding vs non-coding classification accuracy
2. Reading frame prediction accuracy for coding sequences  
3. Bacterial vs non-bacterial classification accuracy

EXAMPLES:
  # Bacterial genome test
  python source/test_genomic_accuracy.py --fasta genome.fasta --gff annotations.gff --is_bact true --taxon "E.coli"
  
  # Eukaryotic genome test  
  python source/test_genomic_accuracy.py --fasta genome.fasta --gtf annotations.gtf --is_bact false --taxon "S.cerevisiae"
  
  # Using provided test files
  python source/test_genomic_accuracy.py --fasta tests/GCF_000146045.2_R64_genomic.fasta --gff tests/GCF_000146045.2_R64_genomic.gff --is_bact false --taxon "S.cerevisiae"

OUTPUT:
  Detailed results followed by tab-separated summary line starting with "SUMMARY_STATS"
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
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Directory for output files (default: current directory)")
    parser.add_argument("--keep_files", action="store_true",
                       help="Keep generated test files after completion")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if not os.path.exists(args.fasta):
            raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
        
        annotation_file = args.gff or args.gtf
        if not annotation_file:
            raise ValueError("Either --gff or --gtf must be provided")
        
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        is_bacterial = args.is_bact.lower() == 'true'
        
        # Setup test environment
        setup_test_environment()
        
        print(f"BBERT Genomic Accuracy Test - {args.taxon}")
        print("=" * 50)
        print(f"Genome: {args.fasta}")
        print(f"Annotations: {annotation_file}")
        print(f"Organism type: {'Bacterial' if is_bacterial else 'Non-bacterial'}")
        print(f"Reads per CDS: {args.reads_per_cds}")
        print(f"Non-coding reads: {'manual override' if args.noncoding_reads >= 0 else 'calculated proportionally'} ({args.noncoding_reads if args.noncoding_reads >= 0 else 'auto'})")
        print(f"Read length: {args.read_length}")
        print()
        
        # Load genome sequences
        print("Loading genome sequences...")
        genome_seqs = {}
        with open(args.fasta, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                genome_seqs[record.id] = str(record.seq).upper()
        
        total_genome_length = sum(len(seq) for seq in genome_seqs.values())
        print(f"Loaded {len(genome_seqs)} sequences, total length: {total_genome_length:,} bp")
        
        # Parse annotations
        annotations = GenomicAnnotation(annotation_file)
        
        # Generate test reads
        print("\nGenerating coding sequence reads...")
        coding_reads, cds_stats = generate_coding_reads(genome_seqs, annotations, 
                                                       args.read_length, args.reads_per_cds)
        
        print(f"CDS statistics: {cds_stats['used_cds']}/{cds_stats['total_cds']} CDS used " +
              f"({cds_stats['short_cds']} too short, {cds_stats['no_positions_cds']} no valid positions)")
        
        # Show frame distribution
        frame_dist = cds_stats['frame_distribution']
        total_coding_reads = sum(frame_dist.values())
        expected_reads = cds_stats['used_cds'] * args.reads_per_cds
        
        print(f"Read generation summary:")
        print(f"  Expected reads: {expected_reads:,} ({cds_stats['used_cds']:,} CDS × {args.reads_per_cds} reads/CDS)")
        print(f"  Actual reads: {total_coding_reads:,}")
        print(f"  Generation efficiency: {(total_coding_reads/expected_reads*100) if expected_reads > 0 else 0:.1f}%")
        
        print(f"Frame distribution (total: {total_coding_reads}):")
        for frame in sorted(frame_dist.keys()):
            count = frame_dist[frame]
            pct = (count / total_coding_reads * 100) if total_coding_reads > 0 else 0
            expected_per_frame = expected_reads // 6
            print(f"  Frame {frame:+d}: {count:5d} ({pct:5.1f}%) [expected: ~{expected_per_frame}]")
        print()
        
        print("Generating non-coding sequence reads...")
        # Use manual override if specified (>= 0), otherwise calculate proportionally (-1)
        if args.noncoding_reads >= 0:
            noncoding_reads_needed = args.noncoding_reads
            print(f"Using manual override: {noncoding_reads_needed} non-coding reads")
        else:
            noncoding_reads_needed = calculate_noncoding_reads_needed(genome_seqs, annotations, len(coding_reads))
        
        noncoding_reads = generate_noncoding_reads(genome_seqs, annotations,
                                                 args.read_length, noncoding_reads_needed)
        
        all_reads = coding_reads + noncoding_reads
        print(f"Generated {len(all_reads)} total reads ({len(coding_reads)} coding, {len(noncoding_reads)} non-coding)")
        
        if not all_reads:
            raise ValueError("No test reads generated")
        
        # Create metadata for analysis
        test_reads_metadata = {}
        for read in all_reads:
            test_reads_metadata[read.id] = read.ground_truth
        
        # Write test reads to file
        test_reads_file = os.path.abspath(os.path.join(args.output_dir, f"genomic_test_reads_{args.taxon}.fasta"))
        with open(test_reads_file, 'w') as f:
            SeqIO.write(all_reads, f, 'fasta')
        print(f"Wrote test reads to {test_reads_file}")
        print()
        
        # Run BBERT inference
        parquet_file = run_bbert_inference(test_reads_file, args.output_dir, args.verbose)
        print()
        
        # Analyze results
        results = analyze_genomic_predictions(parquet_file, test_reads_metadata, is_bacterial)
        
        # Print results
        print_detailed_results(results, args.taxon)
        print_summary_stats(results, args.taxon)
        
        # Cleanup if requested
        if not args.keep_files:
            if os.path.exists(test_reads_file):
                # os.remove(test_reads_file)
                # print(f"\nCleaned up {test_reads_file}")
                print(f"\nKeeping {test_reads_file}")
            if os.path.exists(parquet_file):
                os.remove(parquet_file)
                print(f"Cleaned up {parquet_file}")
        else:
            print(f"\nTest files kept:")
            print(f"  Reads: {test_reads_file}")
            print(f"  Results: {parquet_file}")
        
        return 0
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
