#!/usr/bin/env python3
"""
Unit tests for generate_annotated_reads.py module.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import logging
import tempfile
import gzip

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "testing"))


class TestSetupLogging(unittest.TestCase):
    """Test setup_logging function."""

    def test_setup_logging(self):
        """Test that setup_logging returns a logger."""
        from generate_annotated_reads import setup_logging

        logger = setup_logging()

        self.assertIsInstance(logger, logging.Logger)
        # Logger name is now from bbert.utils.common since we import from there
        self.assertEqual(logger.name, 'bbert.utils.common')


class TestDetectFileFormat(unittest.TestCase):
    """Test detect_file_format function."""

    def test_fasta_format(self):
        """Test FASTA format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fasta')
        self.assertEqual(fmt, 'fasta')

    def test_fna_format(self):
        """Test FNA format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fna')
        self.assertEqual(fmt, 'fasta')

    def test_fa_format(self):
        """Test FA format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fa')
        self.assertEqual(fmt, 'fasta')

    def test_fastq_format(self):
        """Test FASTQ format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fastq')
        self.assertEqual(fmt, 'fastq')

    def test_fq_format(self):
        """Test FQ format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fq')
        self.assertEqual(fmt, 'fastq')

    def test_fasta_gz_format(self):
        """Test compressed FASTA format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fasta.gz')
        self.assertEqual(fmt, 'fasta')

    def test_fna_gz_format(self):
        """Test compressed FNA format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fna.gz')
        self.assertEqual(fmt, 'fasta')

    def test_fa_gz_format(self):
        """Test compressed FA format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fa.gz')
        self.assertEqual(fmt, 'fasta')

    def test_fastq_gz_format(self):
        """Test compressed FASTQ format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fastq.gz')
        self.assertEqual(fmt, 'fastq')

    def test_fq_gz_format(self):
        """Test compressed FQ format detection."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.fq.gz')
        self.assertEqual(fmt, 'fastq')

    def test_generic_gz_format(self):
        """Test generic .gz format detection (defaults to FASTQ)."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('file.gz')
        self.assertEqual(fmt, 'fastq')

    def test_unsupported_format(self):
        """Test unsupported format raises ValueError."""
        from generate_annotated_reads import detect_file_format

        with self.assertRaises(ValueError) as context:
            detect_file_format('file.txt')

        self.assertIn('Unsupported file format', str(context.exception))

    def test_case_insensitive(self):
        """Test format detection is case insensitive."""
        from generate_annotated_reads import detect_file_format

        open_fn, fmt = detect_file_format('FILE.FASTA')
        self.assertEqual(fmt, 'fasta')

        open_fn, fmt = detect_file_format('FILE.FASTQ.GZ')
        self.assertEqual(fmt, 'fastq')


class TestCalculateTrueFrame(unittest.TestCase):
    """Test calculate_true_frame function."""

    def test_frame_phase0_pos0(self):
        """Test frame calculation for phase=0, pos=0."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (0 - 0) % 3 = 0, frame = 0 + 1 = 1
        frame = calculate_true_frame(pos=0, phase=0)
        self.assertEqual(frame, 1)

    def test_frame_phase0_pos1(self):
        """Test frame calculation for phase=0, pos=1."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (0 - 1) % 3 = 2, frame = 2 + 1 = 3
        frame = calculate_true_frame(pos=1, phase=0)
        self.assertEqual(frame, 3)

    def test_frame_phase0_pos2(self):
        """Test frame calculation for phase=0, pos=2."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (0 - 2) % 3 = 1, frame = 1 + 1 = 2
        frame = calculate_true_frame(pos=2, phase=0)
        self.assertEqual(frame, 2)

    def test_frame_phase1_pos0(self):
        """Test frame calculation for phase=1, pos=0."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (1 - 0) % 3 = 1, frame = 1 + 1 = 2
        frame = calculate_true_frame(pos=0, phase=1)
        self.assertEqual(frame, 2)

    def test_frame_phase1_pos1(self):
        """Test frame calculation for phase=1, pos=1."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (1 - 1) % 3 = 0, frame = 0 + 1 = 1
        frame = calculate_true_frame(pos=1, phase=1)
        self.assertEqual(frame, 1)

    def test_frame_phase2_pos0(self):
        """Test frame calculation for phase=2, pos=0."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (2 - 0) % 3 = 2, frame = 2 + 1 = 3
        frame = calculate_true_frame(pos=0, phase=2)
        self.assertEqual(frame, 3)

    def test_frame_phase2_pos2(self):
        """Test frame calculation for phase=2, pos=2."""
        from generate_annotated_reads import calculate_true_frame

        # (phase - pos) % 3 = (2 - 2) % 3 = 0, frame = 0 + 1 = 1
        frame = calculate_true_frame(pos=2, phase=2)
        self.assertEqual(frame, 1)


class TestSaveReadsToFasta(unittest.TestCase):
    """Test save_reads_to_fasta function."""

    def test_save_coding_reads(self):
        """Test saving coding reads to FASTA."""
        from generate_annotated_reads import save_reads_to_fasta

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test.fasta')

            reads_data = [
                {
                    'read_id': 'read1',
                    'sequence': 'ATGCGATCG',
                    'is_coding': True,
                    'is_bacterial': True,
                    'true_frame': 1
                },
                {
                    'read_id': 'read2',
                    'sequence': 'GCTAGCTAGC',
                    'is_coding': True,
                    'is_bacterial': False,
                    'true_frame': -2
                }
            ]

            save_reads_to_fasta(reads_data, output_file)

            # Verify file exists
            self.assertTrue(os.path.exists(output_file))

            # Read and verify content
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn('read1', content)
                self.assertIn('read2', content)
                self.assertIn('ATGCGATCG', content)
                self.assertIn('GCTAGCTAGC', content)
                self.assertIn('coding=True', content)
                self.assertIn('frame=1', content)
                self.assertIn('frame=-2', content)

    def test_save_noncoding_reads(self):
        """Test saving noncoding reads to FASTA."""
        from generate_annotated_reads import save_reads_to_fasta

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test.fasta')

            reads_data = [
                {
                    'read_id': 'noncoding1',
                    'sequence': 'ATGCGATCG',
                    'is_coding': False,
                    'is_bacterial': True,
                    'true_frame': None
                }
            ]

            save_reads_to_fasta(reads_data, output_file)

            # Verify file exists
            self.assertTrue(os.path.exists(output_file))

            # Read and verify content
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn('noncoding1', content)
                self.assertIn('coding=False', content)
                self.assertNotIn('frame=', content)


class TestSaveReadsToCsv(unittest.TestCase):
    """Test save_reads_to_csv function."""

    def test_save_reads_to_csv(self):
        """Test saving reads metadata to CSV."""
        from generate_annotated_reads import save_reads_to_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test.csv')

            reads_data = [
                {
                    'read_id': 'read1',
                    'sequence': 'ATGCGATCG',
                    'is_coding': True,
                    'is_bacterial': True,
                    'true_frame': 1,
                    'chromosome': 'chr1',
                    'read_length': 9
                },
                {
                    'read_id': 'read2',
                    'sequence': 'GCTAGCTAGC',
                    'is_coding': False,
                    'is_bacterial': False,
                    'true_frame': None,
                    'chromosome': 'chr2',
                    'read_length': 10
                }
            ]

            save_reads_to_csv(reads_data, output_file)

            # Verify file exists
            self.assertTrue(os.path.exists(output_file))

            # Read and verify content
            df = pd.read_csv(output_file)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.loc[0, 'read_id'], 'read1')
            self.assertEqual(df.loc[1, 'read_id'], 'read2')
            self.assertEqual(df.loc[0, 'is_coding'], True)
            self.assertEqual(df.loc[1, 'is_coding'], False)


class TestGenerateCodingReads(unittest.TestCase):
    """Test generate_coding_reads_with_frames function."""

    def test_generate_coding_reads_basic(self):
        """Test basic coding read generation."""
        from generate_annotated_reads import generate_coding_reads_with_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal FASTA file
            fasta_file = os.path.join(tmpdir, 'genome.fasta')
            with open(fasta_file, 'w') as f:
                f.write('>chr1\n')
                # 300bp sequence divisible by 3
                f.write('ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG' * 7 + '\n')

            # Create minimal GFF file
            gff_file = os.path.join(tmpdir, 'annotations.gff')
            with open(gff_file, 'w') as f:
                f.write('##gff-version 3\n')
                # CDS from position 1-300 (1-based), phase 0, plus strand
                f.write('chr1\t.\tCDS\t1\t300\t.\t+\t0\t.\n')

            reads = generate_coding_reads_with_frames(
                fasta_file, gff_file,
                reads_per_cds=2,
                read_length=100,
                is_bacterial=True
            )

            # Should generate 2 reads
            self.assertEqual(len(reads), 2)

            # Verify read structure
            for read in reads:
                self.assertIn('read_id', read)
                self.assertIn('sequence', read)
                self.assertIn('is_coding', read)
                self.assertIn('is_bacterial', read)
                self.assertIn('true_frame', read)
                self.assertTrue(read['is_coding'])
                self.assertTrue(read['is_bacterial'])
                self.assertEqual(len(read['sequence']), 100)

    def test_generate_coding_reads_short_cds(self):
        """Test that short CDS regions are skipped."""
        from generate_annotated_reads import generate_coding_reads_with_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'genome.fasta')
            with open(fasta_file, 'w') as f:
                f.write('>chr1\n')
                f.write('ATGCGATCGATCGATCGATCG\n')  # Only 21bp

            gff_file = os.path.join(tmpdir, 'annotations.gff')
            with open(gff_file, 'w') as f:
                f.write('##gff-version 3\n')
                f.write('chr1\t.\tCDS\t1\t21\t.\t+\t0\t.\n')

            reads = generate_coding_reads_with_frames(
                fasta_file, gff_file,
                reads_per_cds=2,
                read_length=100,
                is_bacterial=True
            )

            # Should not generate any reads (CDS too short)
            self.assertEqual(len(reads), 0)

    def test_generate_coding_reads_minus_strand(self):
        """Test coding read generation from minus strand."""
        from generate_annotated_reads import generate_coding_reads_with_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'genome.fasta')
            with open(fasta_file, 'w') as f:
                f.write('>chr1\n')
                f.write('ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG' * 7 + '\n')

            gff_file = os.path.join(tmpdir, 'annotations.gff')
            with open(gff_file, 'w') as f:
                f.write('##gff-version 3\n')
                f.write('chr1\t.\tCDS\t1\t300\t.\t-\t0\t.\n')  # Minus strand

            reads = generate_coding_reads_with_frames(
                fasta_file, gff_file,
                reads_per_cds=2,
                read_length=100,
                is_bacterial=True
            )

            # Should generate reads
            self.assertGreater(len(reads), 0)

            # Verify strand info
            for read in reads:
                self.assertEqual(read['strand'], '-')


class TestGenerateNoncodingReads(unittest.TestCase):
    """Test generate_noncoding_reads function."""

    def test_generate_noncoding_reads_basic(self):
        """Test basic noncoding read generation."""
        from generate_annotated_reads import generate_noncoding_reads

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create FASTA file with long sequence
            fasta_file = os.path.join(tmpdir, 'genome.fasta')
            with open(fasta_file, 'w') as f:
                f.write('>chr1\n')
                # Create a long sequence (10kb)
                f.write('ATGCGATCGATCGATCGATCG' * 500 + '\n')

            # Create GFF with a CDS region
            gff_file = os.path.join(tmpdir, 'annotations.gff')
            with open(gff_file, 'w') as f:
                f.write('##gff-version 3\n')
                # Small CDS region, leaving most of sequence as intergenic
                f.write('chr1\t.\tCDS\t1\t100\t.\t+\t0\t.\n')

            reads = generate_noncoding_reads(
                fasta_file, gff_file,
                num_reads=10,
                read_length=100,
                is_bacterial=True
            )

            # Should generate reads
            self.assertGreater(len(reads), 0)

            # Verify read structure
            for read in reads:
                self.assertIn('read_id', read)
                self.assertIn('sequence', read)
                self.assertIn('is_coding', read)
                self.assertIn('is_bacterial', read)
                self.assertFalse(read['is_coding'])
                self.assertTrue(read['is_bacterial'])
                self.assertEqual(len(read['sequence']), 100)
                self.assertIsNone(read['true_frame'])

    def test_generate_noncoding_reads_no_N(self):
        """Test that reads with 'N' are excluded."""
        from generate_annotated_reads import generate_noncoding_reads

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_file = os.path.join(tmpdir, 'genome.fasta')
            with open(fasta_file, 'w') as f:
                f.write('>chr1\n')
                # Sequence with N characters
                f.write('ATGCGATCGNNNATCGATCGATCG' * 100 + '\n')

            gff_file = os.path.join(tmpdir, 'annotations.gff')
            with open(gff_file, 'w') as f:
                f.write('##gff-version 3\n')

            reads = generate_noncoding_reads(
                fasta_file, gff_file,
                num_reads=10,
                read_length=100,
                is_bacterial=True
            )

            # Verify no reads contain 'N'
            for read in reads:
                self.assertNotIn('N', read['sequence'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
