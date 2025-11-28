#!/usr/bin/env python3
"""
Unit tests for score.py module.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import re

# Mock heavy imports before importing score
sys.modules['pynvml'] = Mock()
sys.modules['BERT_model'] = Mock()
sys.modules['BERT_model.dataset'] = Mock()
sys.modules['BERT_model.utils'] = Mock()
sys.modules['BERT_model.collator'] = Mock()

# Mock pynvml functions
mock_pynvml = Mock()
mock_pynvml.nvmlInit = Mock()
mock_pynvml.nvmlDeviceGetHandleByIndex = Mock()
mock_pynvml.nvmlDeviceGetMemoryInfo = Mock()
mock_pynvml.nvmlDeviceGetUtilizationRates = Mock()
sys.modules['pynvml'] = mock_pynvml

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))


class TestSetupLogging(unittest.TestCase):
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default verbosity (INFO)."""
        from score import setup_logging

        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers = []

        logger = setup_logging(verbose=False)

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)

    def test_setup_logging_verbose(self):
        """Test setup_logging with verbose mode (DEBUG)."""
        from score import setup_logging

        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers = []

        logger = setup_logging(verbose=True)

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_setup_logging_no_duplicate_handlers(self):
        """Test that setup_logging doesn't add duplicate handlers."""
        from score import setup_logging

        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers = []

        logger1 = setup_logging(verbose=False)
        initial_handler_count = len(logger1.handlers)

        # Call again - should not add more handlers
        logger2 = setup_logging(verbose=False)
        final_handler_count = len(logger2.handlers)

        self.assertEqual(initial_handler_count, final_handler_count)


class TestCreateCollateFn(unittest.TestCase):
    """Test create_collate_fn function."""

    def test_create_collate_fn_basic(self):
        """Test basic collate function creation."""
        from score import create_collate_fn

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3], [4, 5, 6]]
        }

        collate_fn = create_collate_fn(mock_tokenizer, max_length=102)

        # Verify it returns a callable
        self.assertTrue(callable(collate_fn))

    def test_collate_fn_processes_batch(self):
        """Test that collate function processes batches correctly."""
        from score import create_collate_fn
        import torch

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
        mock_tokenizer.return_value = {'input_ids': mock_input_ids}

        collate_fn = create_collate_fn(mock_tokenizer, max_length=102)

        # Create a batch
        batch = [
            {'id': 'read1', 'seq': 'ATGCGATCG'},
            {'id': 'read2', 'seq': 'GCTAGCTA'}
        ]

        ids, encoded_seq, seq_lens = collate_fn(batch)

        # Verify output structure
        self.assertEqual(ids, ['read1', 'read2'])
        self.assertEqual(seq_lens, [9, 8])
        self.assertTrue(torch.is_tensor(encoded_seq))

        # Verify tokenizer was called with cleaned sequences
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        self.assertEqual(call_args[0][0], ['ATGCGATCG', 'GCTAGCTA'])
        self.assertEqual(call_args[1]['max_length'], 102)
        self.assertTrue(call_args[1]['truncation'])
        self.assertEqual(call_args[1]['padding'], 'max_length')

    def test_collate_fn_cleans_sequences(self):
        """Test that collate function cleans non-ACTGN characters."""
        from score import create_collate_fn
        import torch

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {'input_ids': mock_input_ids}

        collate_fn = create_collate_fn(mock_tokenizer, max_length=102)

        # Create a batch with non-ACTGN characters
        batch = [
            {'id': 'read1', 'seq': 'ATGCgatcgXYZ123'}
        ]

        ids, encoded_seq, seq_lens = collate_fn(batch)

        # Verify tokenizer received cleaned sequence
        call_args = mock_tokenizer.call_args
        cleaned_seq = call_args[0][0][0]

        # Should convert to uppercase and replace non-ACTGN with N
        self.assertNotIn('X', cleaned_seq)
        self.assertNotIn('Y', cleaned_seq)
        self.assertNotIn('Z', cleaned_seq)
        self.assertNotIn('1', cleaned_seq)
        # Should have N in place of invalid characters
        self.assertIn('N', cleaned_seq)
        # Should preserve valid characters
        self.assertIn('A', cleaned_seq)
        self.assertIn('T', cleaned_seq)
        self.assertIn('G', cleaned_seq)
        self.assertIn('C', cleaned_seq)

    def test_collate_fn_uppercase_conversion(self):
        """Test that collate function converts sequences to uppercase."""
        from score import create_collate_fn
        import torch

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {'input_ids': mock_input_ids}

        collate_fn = create_collate_fn(mock_tokenizer, max_length=102)

        # Create a batch with lowercase sequences
        batch = [
            {'id': 'read1', 'seq': 'atgcgatcg'}
        ]

        ids, encoded_seq, seq_lens = collate_fn(batch)

        # Verify tokenizer received uppercase sequence
        call_args = mock_tokenizer.call_args
        cleaned_seq = call_args[0][0][0]
        self.assertEqual(cleaned_seq, 'ATGCGATCG')

    def test_collate_fn_custom_max_length(self):
        """Test collate function with custom max_length."""
        from score import create_collate_fn
        import torch

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {'input_ids': mock_input_ids}

        collate_fn = create_collate_fn(mock_tokenizer, max_length=50)

        # Create a batch
        batch = [
            {'id': 'read1', 'seq': 'ATGC'}
        ]

        ids, encoded_seq, seq_lens = collate_fn(batch)

        # Verify tokenizer was called with custom max_length
        call_args = mock_tokenizer.call_args
        self.assertEqual(call_args[1]['max_length'], 50)

    def test_collate_fn_preserves_n_characters(self):
        """Test that collate function preserves N characters."""
        from score import create_collate_fn
        import torch

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {'input_ids': mock_input_ids}

        collate_fn = create_collate_fn(mock_tokenizer, max_length=102)

        # Create a batch with N characters
        batch = [
            {'id': 'read1', 'seq': 'ATGCNGATCG'}
        ]

        ids, encoded_seq, seq_lens = collate_fn(batch)

        # Verify N was preserved
        call_args = mock_tokenizer.call_args
        cleaned_seq = call_args[0][0][0]
        self.assertEqual(cleaned_seq, 'ATGCNGATCG')


class TestScoreModuleConstants(unittest.TestCase):
    """Test module-level constants and initialization."""

    def test_wandb_disabled(self):
        """Test that WANDB is disabled."""
        import score

        # Check that WANDB_DISABLED is set
        self.assertEqual(os.environ.get('WANDB_DISABLED'), 'true')


if __name__ == '__main__':
    unittest.main(verbosity=2)
