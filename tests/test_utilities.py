#!/usr/bin/env python3
"""
Unit tests for utility functions in various source modules.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import logging

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))


class TestConvertScoresToTsv(unittest.TestCase):
    """Test functions from convert_scores_to_tsv.py."""

    def test_setup_logging(self):
        """Test setup_logging function."""
        from convert_scores_to_tsv import setup_logging

        logger = setup_logging()

        # Verify logger is returned
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'convert_scores_to_tsv')

    @patch('convert_scores_to_tsv.pd.read_parquet')
    @patch('convert_scores_to_tsv.gzip.open')
    @patch('convert_scores_to_tsv.os.path.getsize')
    @patch('convert_scores_to_tsv.os.path.join')
    def test_convert_parquet_to_tsv_success(self, mock_join, mock_getsize, mock_gzip_open, mock_read_parquet):
        """Test successful parquet to TSV conversion."""
        from convert_scores_to_tsv import convert_parquet_to_tsv

        # Mock data
        mock_df = pd.DataFrame({
            'id': ['read1', 'read2', 'read3', 'read4'],
            'len': [150, 80, 120, 50],
            'loss': [1.2, 1.5, 1.1, 1.8],
            'bact_prob': [0.9, 0.3, 0.8, 0.2]
        })
        mock_read_parquet.return_value = mock_df
        mock_getsize.return_value = 1024 * 1024  # 1MB
        mock_join.side_effect = lambda *args: '/'.join(args)

        # Mock gzip file
        mock_file = mock_open()
        mock_gzip_open.return_value = mock_file.return_value

        result = convert_parquet_to_tsv(
            parquet_file='/fake/input.parquet',
            output_dir='/fake/output',
            output_prefix='test',
            min_length=100
        )

        # Verify success
        self.assertTrue(result)

        # Verify parquet was read
        mock_read_parquet.assert_called_once()

        # Verify gzip.open was called twice (long and short)
        self.assertEqual(mock_gzip_open.call_count, 2)

    @patch('convert_scores_to_tsv.pd.read_parquet')
    def test_convert_parquet_to_tsv_empty_file(self, mock_read_parquet):
        """Test handling of empty parquet file."""
        from convert_scores_to_tsv import convert_parquet_to_tsv

        # Mock empty dataframe
        mock_read_parquet.return_value = pd.DataFrame()

        with patch('convert_scores_to_tsv.os.path.getsize', return_value=0):
            result = convert_parquet_to_tsv(
                parquet_file='/fake/empty.parquet',
                output_dir='/fake/output',
                output_prefix='test'
            )

        # Should return False for empty file
        self.assertFalse(result)

    @patch('convert_scores_to_tsv.pd.read_parquet')
    def test_convert_parquet_to_tsv_read_error(self, mock_read_parquet):
        """Test handling of read errors."""
        from convert_scores_to_tsv import convert_parquet_to_tsv

        # Mock read error
        mock_read_parquet.side_effect = Exception("Read error")

        with patch('convert_scores_to_tsv.os.path.getsize', return_value=1024):
            result = convert_parquet_to_tsv(
                parquet_file='/fake/bad.parquet',
                output_dir='/fake/output',
                output_prefix='test'
            )

        # Should return False on error
        self.assertFalse(result)


class TestMergeScores(unittest.TestCase):
    """Test functions from merge_scores.py."""

    @patch('merge_scores.pq.read_table')
    def test_read_parquet_smalltypes(self, mock_read_table):
        """Test read_parquet_smalltypes function."""
        from merge_scores import read_parquet_smalltypes

        # Mock PyArrow table
        mock_table = Mock()
        mock_cast_table = Mock()
        mock_table.cast.return_value = mock_cast_table
        mock_cast_table.to_pandas.return_value = pd.DataFrame({
            'id': ['read1', 'read2'],
            'len': [100, 150],
            'loss': [1.2, 1.5],
            'bact_prob': [0.9, 0.8]
        })
        mock_read_table.return_value = mock_table

        result = read_parquet_smalltypes('/fake/file.parquet')

        # Verify table was read with correct columns
        mock_read_table.assert_called_once_with('/fake/file.parquet', columns=['id', 'len', 'loss', 'bact_prob'])

        # Verify cast was called
        mock_table.cast.assert_called_once()

        # Verify result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)


class TestDownloadModels(unittest.TestCase):
    """Test functions from download_models.py."""

    def test_is_lfs_pointer_true(self):
        """Test LFS pointer detection for actual LFS pointer."""
        from download_models import is_lfs_pointer
        from pathlib import Path
        import tempfile

        # Create a temporary LFS pointer file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as f:
            f.write('version https://git-lfs.github.com/spec/v1\n')
            f.write('oid sha256:abc123\n')
            f.write('size 1234\n')
            temp_path = f.name

        try:
            result = is_lfs_pointer(Path(temp_path))
            self.assertTrue(result)
        finally:
            os.unlink(temp_path)

    def test_is_lfs_pointer_false_large_file(self):
        """Test LFS pointer detection for large file."""
        from download_models import is_lfs_pointer
        from pathlib import Path
        import tempfile

        # Create a large file (> 200 bytes)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('x' * 300)
            temp_path = f.name

        try:
            result = is_lfs_pointer(Path(temp_path))
            self.assertFalse(result)
        finally:
            os.unlink(temp_path)

    def test_is_lfs_pointer_false_regular_file(self):
        """Test LFS pointer detection for regular small file."""
        from download_models import is_lfs_pointer
        from pathlib import Path
        import tempfile

        # Create a small regular file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('Just regular content\n')
            temp_path = f.name

        try:
            result = is_lfs_pointer(Path(temp_path))
            self.assertFalse(result)
        finally:
            os.unlink(temp_path)

    @patch('download_models.Path.is_file')
    @patch('download_models.Path.is_dir')
    @patch('download_models.is_lfs_pointer')
    def test_check_model_exists_real_file(self, mock_is_lfs, mock_is_dir, mock_is_file):
        """Test check_model_exists for real file."""
        from download_models import check_model_exists
        from pathlib import Path

        mock_is_file.return_value = True
        mock_is_dir.return_value = False
        mock_is_lfs.return_value = False

        result = check_model_exists(Path('/fake/model.bin'))
        self.assertTrue(result)

    @patch('download_models.Path.is_file')
    @patch('download_models.is_lfs_pointer')
    def test_check_model_exists_lfs_pointer(self, mock_is_lfs, mock_is_file):
        """Test check_model_exists rejects LFS pointers."""
        from download_models import check_model_exists
        from pathlib import Path

        mock_is_file.return_value = True
        mock_is_lfs.return_value = True

        result = check_model_exists(Path('/fake/lfs_pointer.bin'))
        self.assertFalse(result)


class TestExtractCodingAA(unittest.TestCase):
    """Test functions from extract_coding_AA.py."""

    def test_setup_logging(self):
        """Test setup_logging function."""
        from extract_coding_AA import setup_logging

        logger = setup_logging()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'extract_coding_AA')

    def test_detect_file_format_fasta(self):
        """Test file format detection for FASTA."""
        from extract_coding_AA import detect_file_format

        open_fn, fmt = detect_file_format('/path/to/file.fasta')
        self.assertEqual(fmt, 'fasta')

    def test_detect_file_format_fastq(self):
        """Test file format detection for FASTQ."""
        from extract_coding_AA import detect_file_format

        open_fn, fmt = detect_file_format('/path/to/file.fastq')
        self.assertEqual(fmt, 'fastq')

    def test_detect_file_format_fasta_gz(self):
        """Test file format detection for gzipped FASTA."""
        from extract_coding_AA import detect_file_format

        open_fn, fmt = detect_file_format('/path/to/file.fasta.gz')
        self.assertEqual(fmt, 'fasta')

    def test_detect_file_format_fastq_gz(self):
        """Test file format detection for gzipped FASTQ."""
        from extract_coding_AA import detect_file_format

        open_fn, fmt = detect_file_format('/path/to/file.fastq.gz')
        self.assertEqual(fmt, 'fastq')

    def test_detect_file_format_unsupported(self):
        """Test file format detection raises error for unsupported format."""
        from extract_coding_AA import detect_file_format

        with self.assertRaises(ValueError) as context:
            detect_file_format('/path/to/file.txt')

        self.assertIn('Unsupported file format', str(context.exception))

    def test_get_best_reading_frame(self):
        """Test reading frame selection."""
        from extract_coding_AA import get_best_reading_frame
        import numpy as np

        # Test each frame position
        # Frame mapping: [-1, -3, -2, +1, +3, +2]

        # Position 0 -> Frame -1
        probs = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(get_best_reading_frame(probs), -1)

        # Position 1 -> Frame -3
        probs = np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(get_best_reading_frame(probs), -3)

        # Position 2 -> Frame -2
        probs = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
        self.assertEqual(get_best_reading_frame(probs), -2)

        # Position 3 -> Frame +1
        probs = np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0])
        self.assertEqual(get_best_reading_frame(probs), +1)

        # Position 4 -> Frame +3
        probs = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0])
        self.assertEqual(get_best_reading_frame(probs), +3)

        # Position 5 -> Frame +2
        probs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
        self.assertEqual(get_best_reading_frame(probs), +2)

    def test_translate_sequence_frame_plus_1(self):
        """Test DNA translation in frame +1."""
        from extract_coding_AA import translate_sequence

        # Simple test: ATG -> M (Methionine)
        dna = "ATGCGT"  # ATG = M, CGT = R
        aa = translate_sequence(dna, +1)
        self.assertEqual(aa, "MR")

    def test_translate_sequence_frame_plus_2(self):
        """Test DNA translation in frame +2."""
        from extract_coding_AA import translate_sequence

        # Skip first base
        dna = "CATGCGT"  # Skip C, ATG = M, CGT = R
        aa = translate_sequence(dna, +2)
        self.assertEqual(aa, "MR")

    def test_translate_sequence_frame_plus_3(self):
        """Test DNA translation in frame +3."""
        from extract_coding_AA import translate_sequence

        # Skip first two bases
        dna = "CCATGCGT"  # Skip CC, ATG = M, CGT = R
        aa = translate_sequence(dna, +3)
        self.assertEqual(aa, "MR")

    def test_translate_sequence_frame_minus_1(self):
        """Test DNA translation in frame -1 (reverse complement)."""
        from extract_coding_AA import translate_sequence

        # Reverse complement of ATGCGT is ACGCAT
        # ACGCAT: ACG = T, CAT = H
        dna = "ATGCGT"
        aa = translate_sequence(dna, -1)
        self.assertEqual(aa, "TH")

    def test_translate_sequence_lowercase(self):
        """Test translation handles lowercase input."""
        from extract_coding_AA import translate_sequence

        dna = "atgcgt"  # Should work same as "ATGCGT"
        aa = translate_sequence(dna, +1)
        self.assertEqual(aa, "MR")


# Note: score.py tests skipped - module runs argparse.parse_args() at import time,
# making it untestable without refactoring. Would need to move argparse into
# if __name__ == "__main__" block.


if __name__ == '__main__':
    unittest.main(verbosity=2)
