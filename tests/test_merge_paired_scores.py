#!/usr/bin/env python3
"""
Unit tests for merge_paired_scores.py module.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import logging
import tempfile

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "utilities"))


class TestSetupLogging(unittest.TestCase):
    """Test setup_logging function."""

    def test_setup_logging(self):
        """Test that setup_logging returns a logger."""
        from merge_paired_scores import setup_logging

        logger = setup_logging()

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'merge_paired_scores')


class TestReadParquetEfficient(unittest.TestCase):
    """Test read_parquet_efficient function."""

    @patch('merge_paired_scores.pq.read_table')
    def test_read_parquet_efficient_success(self, mock_read_table):
        """Test successful parquet reading."""
        from merge_paired_scores import read_parquet_efficient

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

        result = read_parquet_efficient('/fake/file.parquet')

        # Verify correct columns were read
        mock_read_table.assert_called_once_with('/fake/file.parquet', columns=['id', 'len', 'loss', 'bact_prob'])

        # Verify cast was called
        mock_table.cast.assert_called_once()

        # Verify result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch('merge_paired_scores.pq.read_table')
    def test_read_parquet_efficient_file_not_found(self, mock_read_table):
        """Test error handling for missing file."""
        from merge_paired_scores import read_parquet_efficient

        mock_read_table.side_effect = FileNotFoundError("File not found")

        with self.assertRaises(FileNotFoundError) as context:
            read_parquet_efficient('/fake/missing.parquet')

        self.assertIn('Error reading', str(context.exception))

    @patch('merge_paired_scores.pq.read_table')
    def test_read_parquet_efficient_read_error(self, mock_read_table):
        """Test error handling for read errors."""
        from merge_paired_scores import read_parquet_efficient

        mock_read_table.side_effect = Exception("Corrupted file")

        with self.assertRaises(FileNotFoundError) as context:
            read_parquet_efficient('/fake/corrupted.parquet')

        self.assertIn('Error reading', str(context.exception))


class TestMergePairedScores(unittest.TestCase):
    """Test merge_paired_scores function."""

    def test_merge_paired_scores_both_long(self):
        """Test merging when both R1 and R2 are long reads."""
        from merge_paired_scores import merge_paired_scores

        # Create test data
        with tempfile.TemporaryDirectory() as tmpdir:
            r1_file = os.path.join(tmpdir, 'r1.parquet')
            r2_file = os.path.join(tmpdir, 'r2.parquet')

            # Create mock data - both reads >= 100bp
            r1_data = pd.DataFrame({
                'id': ['read1', 'read2'],
                'len': [120, 150],
                'loss': [1.2, 1.5],
                'bact_prob': [0.9, 0.8]
            })
            r2_data = pd.DataFrame({
                'id': ['read1', 'read2'],
                'len': [110, 140],
                'loss': [1.1, 1.4],
                'bact_prob': [0.85, 0.75]
            })

            r1_data.to_parquet(r1_file)
            r2_data.to_parquet(r2_file)

            long_scores, short_scores = merge_paired_scores(r1_file, r2_file, min_length=100)

            # Both reads should be in long_scores (averaged)
            self.assertEqual(len(long_scores), 2)
            self.assertEqual(len(short_scores), 0)

            # Check averaged values
            self.assertAlmostEqual(long_scores.loc[0, 'loss'], (1.2 + 1.1) / 2, places=5)
            self.assertAlmostEqual(long_scores.loc[0, 'bact_prob'], (0.9 + 0.85) / 2, places=5)

    def test_merge_paired_scores_mixed_lengths(self):
        """Test merging with mixed read lengths."""
        from merge_paired_scores import merge_paired_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            r1_file = os.path.join(tmpdir, 'r1.parquet')
            r2_file = os.path.join(tmpdir, 'r2.parquet')

            # Mixed: some long, some short
            r1_data = pd.DataFrame({
                'id': ['read1', 'read2', 'read3', 'read4'],
                'len': [120, 80, 150, 50],  # long, short, long, short
                'loss': [1.2, 1.5, 1.3, 1.6],
                'bact_prob': [0.9, 0.8, 0.85, 0.7]
            })
            r2_data = pd.DataFrame({
                'id': ['read1', 'read2', 'read3', 'read4'],
                'len': [50, 110, 60, 40],  # short, long, short, short
                'loss': [1.1, 1.4, 1.2, 1.5],
                'bact_prob': [0.85, 0.75, 0.8, 0.65]
            })

            r1_data.to_parquet(r1_file)
            r2_data.to_parquet(r2_file)

            long_scores, short_scores = merge_paired_scores(r1_file, r2_file, min_length=100)

            # read1: R1 long, R2 short -> use R1 values
            # read2: R1 short, R2 long -> use R2 values
            # read3: R1 long, R2 short -> use R1 values
            # read4: both short -> no score
            self.assertEqual(len(long_scores), 3)
            self.assertEqual(len(short_scores), 1)

            # Check that read1 uses R1 values
            read1_row = long_scores[long_scores['id'] == 'read1'].iloc[0]
            self.assertAlmostEqual(read1_row['loss'], 1.2, places=5)
            self.assertAlmostEqual(read1_row['bact_prob'], 0.9, places=5)

            # Check that read2 uses R2 values
            read2_row = long_scores[long_scores['id'] == 'read2'].iloc[0]
            self.assertAlmostEqual(read2_row['loss'], 1.4, places=5)
            self.assertAlmostEqual(read2_row['bact_prob'], 0.75, places=5)

            # Check that read3 uses R1 values
            read3_row = long_scores[long_scores['id'] == 'read3'].iloc[0]
            self.assertAlmostEqual(read3_row['loss'], 1.3, places=5)
            self.assertAlmostEqual(read3_row['bact_prob'], 0.85, places=5)

    def test_merge_paired_scores_both_short(self):
        """Test merging when both reads are too short."""
        from merge_paired_scores import merge_paired_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            r1_file = os.path.join(tmpdir, 'r1.parquet')
            r2_file = os.path.join(tmpdir, 'r2.parquet')

            # Both reads < 100bp
            r1_data = pd.DataFrame({
                'id': ['read1'],
                'len': [50],
                'loss': [1.2],
                'bact_prob': [0.9]
            })
            r2_data = pd.DataFrame({
                'id': ['read1'],
                'len': [60],
                'loss': [1.1],
                'bact_prob': [0.85]
            })

            r1_data.to_parquet(r1_file)
            r2_data.to_parquet(r2_file)

            long_scores, short_scores = merge_paired_scores(r1_file, r2_file, min_length=100)

            # Both short, should be in short_scores
            self.assertEqual(len(long_scores), 0)
            self.assertEqual(len(short_scores), 1)

            # Check short_scores has correct columns
            self.assertIn('id', short_scores.columns)
            self.assertIn('R1_len', short_scores.columns)
            self.assertIn('R2_len', short_scores.columns)

    @patch('merge_paired_scores.read_parquet_efficient')
    @patch('merge_paired_scores.os.path.getsize')
    def test_merge_paired_scores_empty_file(self, mock_getsize, mock_read):
        """Test handling of empty parquet file."""
        from merge_paired_scores import merge_paired_scores

        mock_getsize.return_value = 0
        mock_read.return_value = pd.DataFrame()

        long_scores, short_scores = merge_paired_scores('/fake/r1.parquet', '/fake/r2.parquet')

        # Should return None for both
        self.assertIsNone(long_scores)
        self.assertIsNone(short_scores)

    @patch('merge_paired_scores.read_parquet_efficient')
    @patch('merge_paired_scores.os.path.getsize')
    def test_merge_paired_scores_file_not_found(self, mock_getsize, mock_read):
        """Test handling when file doesn't exist."""
        from merge_paired_scores import merge_paired_scores

        mock_getsize.return_value = 1024
        mock_read.side_effect = FileNotFoundError("File not found")

        long_scores, short_scores = merge_paired_scores('/fake/r1.parquet', '/fake/r2.parquet')

        # Should return None for both
        self.assertIsNone(long_scores)
        self.assertIsNone(short_scores)

    def test_merge_paired_scores_custom_min_length(self):
        """Test merging with custom minimum length threshold."""
        from merge_paired_scores import merge_paired_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            r1_file = os.path.join(tmpdir, 'r1.parquet')
            r2_file = os.path.join(tmpdir, 'r2.parquet')

            # Test with min_length=150
            r1_data = pd.DataFrame({
                'id': ['read1', 'read2'],
                'len': [160, 120],  # long, short (for min_length=150)
                'loss': [1.2, 1.5],
                'bact_prob': [0.9, 0.8]
            })
            r2_data = pd.DataFrame({
                'id': ['read1', 'read2'],
                'len': [170, 140],  # long, short (for min_length=150)
                'loss': [1.1, 1.4],
                'bact_prob': [0.85, 0.75]
            })

            r1_data.to_parquet(r1_file)
            r2_data.to_parquet(r2_file)

            long_scores, short_scores = merge_paired_scores(r1_file, r2_file, min_length=150)

            # Only read1 should be in long_scores (both >= 150)
            self.assertEqual(len(long_scores), 1)
            self.assertEqual(len(short_scores), 1)


class TestSaveResults(unittest.TestCase):
    """Test save_results function."""

    def test_save_results_both_dataframes(self):
        """Test saving both long and short scores."""
        from merge_paired_scores import save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            long_scores = pd.DataFrame({
                'id': ['read1'],
                'loss': [1.2],
                'bact_prob': [0.9]
            })
            short_scores = pd.DataFrame({
                'id': ['read2'],
                'R1_len': [50],
                'R2_len': [60],
                'R1_bact_prob': [0.8],
                'R2_bact_prob': [0.75]
            })

            result = save_results(long_scores, short_scores, tmpdir, 'test_sample')

            # Should succeed
            self.assertTrue(result)

            # Check files were created
            long_file = os.path.join(tmpdir, 'test_sample_good_long_scores.tsv.gz')
            short_file = os.path.join(tmpdir, 'test_sample_good_short_scores.tsv.gz')

            self.assertTrue(os.path.exists(long_file))
            self.assertTrue(os.path.exists(short_file))

            # Verify file contents
            import gzip
            with gzip.open(long_file, 'rt') as f:
                long_df = pd.read_csv(f, sep='\t')
                self.assertEqual(len(long_df), 1)
                self.assertEqual(long_df.iloc[0]['id'], 'read1')

    def test_save_results_empty_long_scores(self):
        """Test saving with empty long scores."""
        from merge_paired_scores import save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            long_scores = pd.DataFrame(columns=['id', 'loss', 'bact_prob'])
            short_scores = pd.DataFrame({
                'id': ['read1'],
                'R1_len': [50],
                'R2_len': [60],
                'R1_bact_prob': [0.8],
                'R2_bact_prob': [0.75]
            })

            result = save_results(long_scores, short_scores, tmpdir, 'test_sample')

            # Should still succeed
            self.assertTrue(result)

            # Only short file should exist
            short_file = os.path.join(tmpdir, 'test_sample_good_short_scores.tsv.gz')
            self.assertTrue(os.path.exists(short_file))

    def test_save_results_empty_short_scores(self):
        """Test saving with empty short scores."""
        from merge_paired_scores import save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            long_scores = pd.DataFrame({
                'id': ['read1'],
                'loss': [1.2],
                'bact_prob': [0.9]
            })
            short_scores = pd.DataFrame(columns=['id', 'R1_len', 'R2_len', 'R1_bact_prob', 'R2_bact_prob'])

            result = save_results(long_scores, short_scores, tmpdir, 'test_sample')

            # Should succeed
            self.assertTrue(result)

            # Only long file should exist
            long_file = os.path.join(tmpdir, 'test_sample_good_long_scores.tsv.gz')
            self.assertTrue(os.path.exists(long_file))

    @patch('merge_paired_scores.gzip.open')
    def test_save_results_write_error(self, mock_gzip_open):
        """Test error handling during file write."""
        from merge_paired_scores import save_results

        long_scores = pd.DataFrame({
            'id': ['read1'],
            'loss': [1.2],
            'bact_prob': [0.9]
        })
        short_scores = pd.DataFrame()

        # Mock gzip.open to raise an error
        mock_gzip_open.side_effect = IOError("Disk full")

        result = save_results(long_scores, short_scores, '/fake/dir', 'test')

        # Should return False on error
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
