#!/usr/bin/env python3
"""
Unit tests for bbert.utils.inference module.
Tests utility functions used by inference.
"""

import unittest
import sys
import os
from pathlib import Path
import torch
from unittest.mock import Mock, patch, MagicMock

# Import the functions to test from bbert package
from bbert.utils.inference import get_device, get_output_filename


class TestGetOutputFilename(unittest.TestCase):
    """Test get_output_filename function in isolation."""

    def test_fasta_file(self):
        """Test output filename for .fasta file."""
        output = get_output_filename("/path/to/file.fasta", "/output", emb_out=False)
        self.assertEqual(output, "/output/file_scores_len.parquet")

    def test_fastq_file(self):
        """Test output filename for .fastq file."""
        output = get_output_filename("/path/to/file.fastq", "/output", emb_out=False)
        self.assertEqual(output, "/output/file_scores_len.parquet")

    def test_fasta_gz_file(self):
        """Test output filename for .fasta.gz file."""
        output = get_output_filename("/path/to/file.fasta.gz", "/output", emb_out=False)
        self.assertEqual(output, "/output/file_scores_len.parquet")

    def test_fastq_gz_file(self):
        """Test output filename for .fastq.gz file."""
        output = get_output_filename("/path/to/file.fastq.gz", "/output", emb_out=False)
        self.assertEqual(output, "/output/file_scores_len.parquet")

    def test_with_embeddings(self):
        """Test output filename with embeddings enabled."""
        output = get_output_filename("/path/to/file.fasta", "/output", emb_out=True)
        self.assertEqual(output, "/output/file_scores_len_emb.parquet")

    def test_complex_path(self):
        """Test output filename with complex directory structure."""
        output = get_output_filename(
            "/deep/nested/path/to/my_sample.fasta.gz",
            "/my/output/dir",
            emb_out=False
        )
        self.assertEqual(output, "/my/output/dir/my_sample_scores_len.parquet")


class TestGetDevice(unittest.TestCase):
    """Test get_device function in isolation."""

    @patch('bbert.utils.inference.torch.cuda.is_available')
    def test_cuda_device(self, mock_cuda_available):
        """Test CUDA device detection."""
        mock_cuda_available.return_value = True

        with patch('bbert.utils.inference.torch.cuda.get_device_name', return_value='NVIDIA A100'):
            with patch('bbert.utils.inference.torch.cuda.current_device', return_value=0):
                device, use_half_precision = get_device()

        self.assertEqual(device.type, 'cuda')
        self.assertTrue(use_half_precision)

    @patch('bbert.utils.inference.torch.cuda.is_available')
    def test_mps_device(self, mock_cuda_available):
        """Test Apple MPS device detection."""
        mock_cuda_available.return_value = False

        with patch('bbert.utils.inference.torch.backends.mps.is_available', return_value=True):
            device, use_half_precision = get_device()

        self.assertEqual(device.type, 'mps')
        self.assertFalse(use_half_precision)

    @patch('bbert.utils.inference.torch.cuda.is_available')
    def test_cpu_device(self, mock_cuda_available):
        """Test CPU device fallback."""
        mock_cuda_available.return_value = False

        if hasattr(torch.backends, 'mps'):
            with patch('bbert.utils.inference.torch.backends.mps.is_available', return_value=False):
                device, use_half_precision = get_device()
        else:
            device, use_half_precision = get_device()

        self.assertEqual(device.type, 'cpu')
        self.assertFalse(use_half_precision)

    def test_with_logger(self):
        """Test that logger is called when provided."""
        mock_logger = Mock()
        device, use_half_precision = get_device(logger=mock_logger)

        # Logger should have been called at least once
        self.assertTrue(mock_logger.info.called)

    @patch('bbert.utils.inference.torch.cuda.is_available')
    def test_cuda_with_logger(self, mock_cuda_available):
        """Test CUDA device with logger to cover lines 25-26."""
        mock_cuda_available.return_value = True
        mock_logger = Mock()

        with patch('bbert.utils.inference.torch.cuda.get_device_name', return_value='Tesla V100'):
            with patch('bbert.utils.inference.torch.cuda.current_device', return_value=0):
                device, use_half_precision = get_device(logger=mock_logger)

        # Check logger was called with GPU name
        mock_logger.info.assert_called_once()
        call_arg = mock_logger.info.call_args[0][0]
        self.assertIn('Tesla V100', call_arg)
        self.assertIn('CUDA', call_arg)

    @patch('bbert.utils.inference.torch.cuda.is_available')
    def test_cpu_with_logger(self, mock_cuda_available):
        """Test CPU device with logger to cover line 36."""
        mock_cuda_available.return_value = False
        mock_logger = Mock()

        if hasattr(torch.backends, 'mps'):
            with patch('bbert.utils.inference.torch.backends.mps.is_available', return_value=False):
                device, use_half_precision = get_device(logger=mock_logger)
        else:
            device, use_half_precision = get_device(logger=mock_logger)

        # Check logger was called with CPU message
        mock_logger.info.assert_called_once()
        call_arg = mock_logger.info.call_args[0][0]
        self.assertIn('CPU', call_arg)


class TestLoadBbertModel(unittest.TestCase):
    """Test load_bbert_model function."""

    def test_load_bbert_model_basic(self):
        """Test basic BBERT model loading."""
        from bbert.utils.inference import load_bbert_model

        # Mock the classes and os.path.exists for path validation
        with patch('bbert.utils.inference.BertForMaskedLM') as mock_model_class, \
             patch('bbert.utils.inference.AutoTokenizer') as mock_tokenizer_class, \
             patch('bbert.utils.inference.os.path.exists') as mock_exists, \
             patch('bbert.utils.inference.os.path.join') as mock_join, \
             patch('bbert.core.collator.CollateFnWithTokenizer') as mock_collator_class:

            # Mock path validation
            mock_exists.return_value = True
            mock_join.side_effect = lambda *args: '/'.join(args)

            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Mock collator
            mock_collate_fn = Mock()
            mock_collator_class.return_value = mock_collate_fn

            # Mock model
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            device = torch.device('cpu')
            model, tokenizer, collate_fn = load_bbert_model(
                model_path='/fake/model',
                tokenizer_path='/fake/tokenizer',
                device=device,
                use_half_precision=False
            )

            # Verify tokenizer was loaded
            mock_tokenizer_class.from_pretrained.assert_called_once_with('/fake/tokenizer', use_fast=True)

            # Verify model was loaded and configured
            mock_model_class.from_pretrained.assert_called_once_with('/fake/model', local_files_only=True)
            mock_model.eval.assert_called_once()
            mock_model.to.assert_called_once_with(device)

            # Verify half precision not called when False
            mock_model.half.assert_not_called()

    def test_load_bbert_model_half_precision(self):
        """Test BBERT model loading with half precision."""
        from bbert.utils.inference import load_bbert_model

        with patch('bbert.utils.inference.BertForMaskedLM') as mock_model_class, \
             patch('bbert.utils.inference.AutoTokenizer') as mock_tokenizer_class, \
             patch('bbert.utils.inference.os.path.exists') as mock_exists, \
             patch('bbert.utils.inference.os.path.join') as mock_join, \
             patch('bbert.core.collator.CollateFnWithTokenizer'):

            # Mock path validation
            mock_exists.return_value = True
            mock_join.side_effect = lambda *args: '/'.join(args)

            mock_tokenizer_class.from_pretrained.return_value = Mock()
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            device = torch.device('cpu')
            load_bbert_model(
                model_path='/fake/model',
                tokenizer_path='/fake/tokenizer',
                device=device,
                use_half_precision=True
            )

            # Verify half precision was called
            mock_model.half.assert_called_once()

    def test_load_bbert_model_with_logger(self):
        """Test BBERT model loading with logger."""
        from bbert.utils.inference import load_bbert_model

        with patch('bbert.utils.inference.BertForMaskedLM') as mock_model_class, \
             patch('bbert.utils.inference.AutoTokenizer') as mock_tokenizer_class, \
             patch('bbert.utils.inference.os.path.exists') as mock_exists, \
             patch('bbert.utils.inference.os.path.join') as mock_join, \
             patch('bbert.core.collator.CollateFnWithTokenizer'):

            # Mock path validation
            mock_exists.return_value = True
            mock_join.side_effect = lambda *args: '/'.join(args)

            mock_tokenizer_class.from_pretrained.return_value = Mock()
            mock_model_class.from_pretrained.return_value = Mock()

            mock_logger = Mock()
            load_bbert_model(
                model_path='/fake/model',
                tokenizer_path='/fake/tokenizer',
                device=torch.device('cpu'),
                use_half_precision=False,
                logger=mock_logger
            )

            # Verify logger was called
            mock_logger.info.assert_called_once()
            call_arg = mock_logger.info.call_args[0][0]
            self.assertIn('BBERT model loaded', call_arg)


class TestLoadClassifier(unittest.TestCase):
    """Test load_classifier function."""

    def test_load_classifier_basic(self):
        """Test basic classifier loading."""
        from bbert.utils.inference import load_classifier

        with patch('bbert.utils.inference.torch.load') as mock_torch_load, \
             patch('bbert.utils.inference.os.path.exists') as mock_exists, \
             patch('bbert.models.classifier.BertClassifier') as mock_classifier_class:

            # Mock path validation
            mock_exists.return_value = True

            # Mock checkpoint
            mock_checkpoint = {'model_state_dict': {'layer1': 'weights'}}
            mock_torch_load.return_value = mock_checkpoint

            # Mock classifier class
            mock_classifier = Mock()
            mock_classifier_class.return_value = mock_classifier

            device = torch.device('cpu')
            classifier = load_classifier(
                model_path='/fake/classifier.pt',
                hidden_size=768,
                num_classes=2,
                device=device,
                use_half_precision=False
            )

            # Verify classifier was created with correct params
            mock_classifier_class.assert_called_once_with(768, 2)

            # Verify checkpoint was loaded
            mock_torch_load.assert_called_once_with('/fake/classifier.pt', weights_only=True, map_location=device)
            mock_classifier.load_state_dict.assert_called_once_with({'layer1': 'weights'})

            # Verify model configuration
            mock_classifier.eval.assert_called_once()
            mock_classifier.to.assert_called_once_with(device)
            mock_classifier.half.assert_not_called()

    def test_load_classifier_half_precision(self):
        """Test classifier loading with half precision."""
        from bbert.utils.inference import load_classifier

        with patch('bbert.utils.inference.torch.load') as mock_torch_load, \
             patch('bbert.utils.inference.os.path.exists') as mock_exists, \
             patch('bbert.models.classifier.BertClassifier') as mock_classifier_class:

            # Mock path validation
            mock_exists.return_value = True

            mock_torch_load.return_value = {'model_state_dict': {}}
            mock_classifier = Mock()
            mock_classifier_class.return_value = mock_classifier

            load_classifier(
                model_path='/fake/classifier.pt',
                hidden_size=768,
                num_classes=6,
                device=torch.device('cpu'),
                use_half_precision=True
            )

            # Verify half precision was called
            mock_classifier.half.assert_called_once()

    def test_load_classifier_with_logger(self):
        """Test classifier loading with logger and custom name."""
        from bbert.utils.inference import load_classifier

        with patch('bbert.utils.inference.torch.load') as mock_torch_load, \
             patch('bbert.utils.inference.os.path.exists') as mock_exists, \
             patch('bbert.models.classifier.BertClassifier') as mock_classifier_class:

            # Mock path validation
            mock_exists.return_value = True

            mock_torch_load.return_value = {'model_state_dict': {}}
            mock_classifier = Mock()
            mock_classifier_class.return_value = mock_classifier

            mock_logger = Mock()
            load_classifier(
                model_path='/fake/classifier.pt',
                hidden_size=768,
                num_classes=2,
                device=torch.device('cpu'),
                use_half_precision=False,
                logger=mock_logger,
                model_name="Frame Classifier"
            )

            # Verify logger was called with model name
            mock_logger.info.assert_called_once()
            call_arg = mock_logger.info.call_args[0][0]
            self.assertIn('Frame Classifier', call_arg)
            self.assertIn('loaded', call_arg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
