#!/usr/bin/env python3
"""
Generate annotated reads with frame and coding/noncoding labels.

This script generates reads from both coding (CDS from GFF/GTF) and non-coding 
(intergenic) regions of a genome, with detailed annotations including:
- True reading frame for coding sequences
- Coding/noncoding labels
- Bacterial/nonbacterial labels
- Sequence coordinates

Output format is suitable for training and evaluation purposes.
"""

import os
import sys
import argparse
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
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

def calculate_true_frame(pos, phase):
    """
    Calculate the correct reading frame for a position within a CDS.
    
    Args:
        pos: Position within the CDS (0-based)
        phase: Phase from GFF (0, 1, or 2)
        
    Returns:
        Frame number: +1, +2, or +3
    """
    # Frame calculation: (phase - pos) % 3 gives the frame offset
    # Frame = offset + 1 to convert to 1-based frames
    frame_offset = (phase - pos) % 3
    return frame_offset + 1

def generate_coding_reads_with_frames(fasta_file, gff_file, reads_per_cds=2, read_length=100, is_bacterial=True):
    """
    Generate coding reads with true frame annotations.
    
    Args:
        fasta_file: Path to genome FASTA file
        gff_file: Path to GFF annotation file
        reads_per_cds: Number of reads to generate per CDS
        read_length: Length of each read
        is_bacterial: Whether this is a bacterial genome
        
    Returns:
        List of dictionaries with read information
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating coding reads from {gff_file}")
    
    # Read genome sequence
    genome = {}
    logger.info(f"Loading genome from {fasta_file}")
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            genome[record.id] = str(record.seq).upper()
    
    reads_data = []
    cds_count = 0
    
    # Parse GFF and extract CDS
    logger.info(f"Parsing annotations from {gff_file}")
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
                
            # Handle both GFF and GTF formats
            feature_type = fields[2]
            if feature_type not in ['CDS', 'exon']:
                continue
                
            chrom = fields[0]
            start = int(fields[3]) - 1  # GFF is 1-based, convert to 0-based
            end = int(fields[4])
            strand = fields[6]
            phase = int(fields[7]) if fields[7] != '.' else 0
            
            if chrom not in genome:
                continue
                
            cds_seq = genome[chrom][start:end]
            if not cds_seq or len(cds_seq) < read_length:
                continue
                
            cds_count += 1
            
            # Get sense sequence (forward or reverse complement based on strand)
            if strand == '+':
                sense_seq = cds_seq
            else:
                sense_seq = str(Seq(cds_seq).reverse_complement())
            
            if len(sense_seq) < read_length:
                continue
            
            # Generate reads from all 6 frames: +1, +2, +3, -1, -2, -3
            # Note: reads_per_cds is the TOTAL number of reads from this CDS, not per frame
            
            # If reads_per_cds < 6, randomly select which frames to use for this CDS
            # to ensure overall balance across all CDS regions
            if reads_per_cds < 6:
                frames_to_use = random.sample([1, 2, 3, -1, -2, -3], reads_per_cds)
                frame_counts = {frame: 1 for frame in frames_to_use}
            else:
                # For larger numbers, distribute across all frames
                per_frame = reads_per_cds // 6
                remainder = reads_per_cds - per_frame * 6
                frame_counts = {}
                for i, frame in enumerate([1, 2, 3, -1, -2, -3]):
                    frame_counts[frame] = per_frame + (1 if remainder > 0 else 0)
                    if remainder > 0:
                        remainder -= 1
            
            read_idx = 0
            for frame in [1, 2, 3, -1, -2, -3]:
                nthis = frame_counts.get(frame, 0)
                if nthis == 0:
                    continue
                
                # Pick positions so (phase - pos) % 3 == |frame| - 1
                f = abs(frame)
                desired_offset = (phase - (f - 1)) % 3
                max_pos = len(sense_seq) - read_length
                if max_pos < 0:
                    continue
                    
                valid_positions = [pos for pos in range(0, max_pos + 1) if (pos % 3) == desired_offset]
                if not valid_positions:
                    continue
                
                for _ in range(nthis):
                    pos = random.choice(valid_positions)
                    frag = sense_seq[pos:pos + read_length]
                    
                    # For negative frames, take reverse complement of fragment
                    if frame > 0:
                        read_seq = frag
                    else:
                        read_seq = str(Seq(frag).reverse_complement())
                    
                    read_idx += 1
                    
                    # Create read data
                    read_data = {
                        'read_id': f"coding_{chrom}_{start}_{end}_{strand}_{read_idx}_frame{frame:+d}",
                        'sequence': read_seq,
                        'is_coding': True,
                        'is_bacterial': is_bacterial,
                        'true_frame': frame,
                        'chromosome': chrom,
                        'cds_start': start,
                        'cds_end': end,
                        'strand': strand,
                        'phase': phase,
                        'read_start_in_cds': pos,
                        'read_length': len(read_seq)
                    }
                    reads_data.append(read_data)
                
            if cds_count % 1000 == 0:
                logger.info(f"Processed {cds_count} CDS regions, generated {len(reads_data)} reads")
    
    logger.info(f"Generated {len(reads_data)} coding reads from {cds_count} CDS regions")
    return reads_data

def generate_noncoding_reads(fasta_file, gff_file, num_reads, read_length=100, is_bacterial=True):
    """
    Generate noncoding reads from intergenic regions.
    
    Args:
        fasta_file: Path to genome FASTA file
        gff_file: Path to GFF annotation file  
        num_reads: Number of noncoding reads to generate
        read_length: Length of each read
        is_bacterial: Whether this is a bacterial genome
        
    Returns:
        List of dictionaries with read information
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_reads} noncoding reads")
    
    # Read genome sequence
    genome = {}
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            genome[record.id] = str(record.seq).upper()
    
    # Parse GFF to find coding regions
    coding_regions = {}
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
                
            feature_type = fields[2]
            # Include all gene-related features to avoid generating reads from coding regions
            if feature_type not in ['CDS', 'exon', 'gene', 'mRNA', 'transcript']:
                continue
                
            chrom = fields[0]
            start = int(fields[3]) - 1
            end = int(fields[4])
            
            if chrom not in coding_regions:
                coding_regions[chrom] = []
            coding_regions[chrom].append((start, end))
    
    # Sort coding regions by start position
    for chrom in coding_regions:
        coding_regions[chrom].sort()
    
    reads_data = []
    attempts = 0
    max_attempts = num_reads * 10
    
    # Generate noncoding reads
    for chrom, seq in genome.items():
        if len(reads_data) >= num_reads:
            break
            
        chrom_coding = coding_regions.get(chrom, [])
        
        while len(reads_data) < num_reads and attempts < max_attempts:
            attempts += 1
            
            # Random position in genome
            start_pos = random.randint(0, max(0, len(seq) - read_length))
            end_pos = start_pos + read_length
            
            # Check if this overlaps with any coding region
            is_coding = False
            for cds_start, cds_end in chrom_coding:
                if not (end_pos <= cds_start or start_pos >= cds_end):
                    is_coding = True
                    break
            
            if not is_coding:
                read_seq = seq[start_pos:end_pos]
                if len(read_seq) == read_length and 'N' not in read_seq:
                    # Create read data
                    read_data = {
                        'read_id': f"noncoding_{chrom}_{start_pos}_{end_pos}_{len(reads_data)}",
                        'sequence': read_seq,
                        'is_coding': False,
                        'is_bacterial': is_bacterial,
                        'true_frame': None,  # No frame for noncoding
                        'chromosome': chrom,
                        'cds_start': None,
                        'cds_end': None,
                        'strand': None,
                        'phase': None,
                        'read_start_in_cds': None,
                        'read_length': len(read_seq)
                    }
                    reads_data.append(read_data)
    
    logger.info(f"Generated {len(reads_data)} noncoding reads in {attempts} attempts")
    return reads_data

def save_reads_to_fasta(reads_data, output_file):
    """Save reads to FASTA format with detailed headers."""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving {len(reads_data)} reads to {output_file}")
    
    records = []
    for read in reads_data:
        # Create concise FASTA header with essential information only
        if read['is_coding']:
            header = (f"{read['read_id']} | coding=True | "
                     f"bacterial={read['is_bacterial']} | frame={read['true_frame']}")
        else:
            header = (f"{read['read_id']} | coding=False | "
                     f"bacterial={read['is_bacterial']}")
        
        record = SeqRecord(Seq(read['sequence']), id=read['read_id'], description=header)
        records.append(record)
    
    with open(output_file, 'w') as f:
        SeqIO.write(records, f, "fasta")
    
    logger.info(f"Successfully saved {len(records)} reads")

def save_reads_to_csv(reads_data, output_file):
    """Save reads metadata to CSV format."""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving metadata for {len(reads_data)} reads to {output_file}")
    
    df = pd.DataFrame(reads_data)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Successfully saved metadata")

def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated reads with frame and coding/noncoding labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate reads from bacterial genome
  python source/generate_annotated_reads.py --fasta genome.fasta --gff annotations.gff --is_bact true --output_prefix bacterial_reads
  
  # Generate reads from eukaryotic genome  
  python source/generate_annotated_reads.py --fasta genome.fasta --gtf annotations.gtf --is_bact false --output_prefix eukaryotic_reads
  
  # Custom read counts
  python source/generate_annotated_reads.py --fasta genome.fasta --gff annotations.gff --is_bact true --reads_per_cds 10 --noncoding_reads 5000 --output_prefix custom_reads

OUTPUT FILES:
  - {prefix}_reads.fasta: FASTA file with annotated headers
  - {prefix}_metadata.csv: CSV file with detailed metadata
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
    parser.add_argument("--output_prefix", required=True,
                       help="Prefix for output files")
    parser.add_argument("--reads_per_cds", type=int, default=20,
                       help="Number of reads per CDS (default: 20)")
    parser.add_argument("--noncoding_reads", type=int, default=-1,
                       help="Number of noncoding reads (default: -1 = proportional to coding reads)")
    parser.add_argument("--read_length", type=int, default=100,
                       help="Length of each read (default: 100)")
    parser.add_argument("--output_dir", default=".",
                       help="Output directory (default: current directory)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
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
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    logger.info("Starting read generation")
    logger.info(f"Genome: {args.fasta}")
    logger.info(f"Annotations: {annotation_file}")
    logger.info(f"Is bacterial: {is_bacterial}")
    logger.info(f"Reads per CDS: {args.reads_per_cds}")
    logger.info(f"Read length: {args.read_length}")
    
    # Generate coding reads
    coding_reads = generate_coding_reads_with_frames(
        args.fasta, annotation_file, 
        reads_per_cds=args.reads_per_cds, 
        read_length=args.read_length,
        is_bacterial=is_bacterial
    )
    
    # Determine number of noncoding reads
    if args.noncoding_reads == -1:
        # Proportional: roughly 20% of coding reads (similar to typical genome composition)
        num_noncoding = max(100, len(coding_reads) // 5)
    else:
        num_noncoding = args.noncoding_reads
    
    logger.info(f"Will generate {num_noncoding} noncoding reads")
    
    # Generate noncoding reads
    noncoding_reads = generate_noncoding_reads(
        args.fasta, annotation_file,
        num_reads=num_noncoding,
        read_length=args.read_length,
        is_bacterial=is_bacterial
    )
    
    # Combine all reads
    all_reads = coding_reads + noncoding_reads
    logger.info(f"Total reads generated: {len(all_reads)} ({len(coding_reads)} coding + {len(noncoding_reads)} noncoding)")
    
    # Shuffle reads for output
    random.shuffle(all_reads)
    
    # Save outputs
    fasta_file = output_dir / f"{args.output_prefix}_reads.fasta"
    csv_file = output_dir / f"{args.output_prefix}_metadata.csv"
    
    save_reads_to_fasta(all_reads, fasta_file)
    save_reads_to_csv(all_reads, csv_file)
    
    # Print summary statistics
    logger.info("\nSUMMARY STATISTICS:")
    logger.info(f"  Total reads: {len(all_reads)}")
    logger.info(f"  Coding reads: {len(coding_reads)} ({len(coding_reads)/len(all_reads)*100:.1f}%)")
    logger.info(f"  Noncoding reads: {len(noncoding_reads)} ({len(noncoding_reads)/len(all_reads)*100:.1f}%)")
    
    if coding_reads:
        # Frame distribution for coding reads
        frame_counts = {}
        for read in coding_reads:
            frame = read['true_frame']
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        
        logger.info(f"  Frame distribution (coding reads):")
        for frame in sorted(frame_counts.keys()):
            count = frame_counts[frame]
            logger.info(f"    Frame {frame:+d}: {count} reads ({count/len(coding_reads)*100:.1f}%)")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  FASTA: {fasta_file}")
    logger.info(f"  Metadata: {csv_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
