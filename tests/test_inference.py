#!/usr/bin/env python3
"""
Unit tests for inference.py module.

Note: Most of inference.py is in the if __name__ == "__main__" block,
which is excluded from coverage. This file tests the module-level code
that executes on import. The actual inference logic has been extracted
to inference_utils.py which has 100% test coverage.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import platform

# Mock heavy imports before importing inference
sys.modules['BERT_model'] = Mock()
sys.modules['BERT_model.dataset'] = Mock()
sys.modules['BERT_model.utils'] = Mock()
sys.modules['BERT_model.collator'] = Mock()
sys.modules['emb_model'] = Mock()
sys.modules['emb_model.architecture'] = Mock()

# Mock get_slurm_cpus to return a predictable value
mock_utils = Mock()
mock_utils.clear_GPU = Mock()
mock_utils.setup_logger = Mock()
mock_utils.label_to_frame = Mock()
mock_utils.get_resources_msg = Mock()
mock_utils.get_slurm_cpus = Mock(return_value=8)
sys.modules['BERT_model.utils'] = mock_utils

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))


class TestInferenceModuleImport(unittest.TestCase):
    """Test that the inference module imports successfully."""

    def test_module_import(self):
        """Test that inference module can be imported."""
        try:
            import inference
            self.assertIsNotNone(inference)
        except ImportError as e:
            self.fail(f"Failed to import inference module: {e}")


class TestInferenceModuleConstants(unittest.TestCase):
    """Test module-level constants in inference.py."""

    def test_constants_defined(self):
        """Test that module constants are defined."""
        import inference

        # Check essential constants
        self.assertEqual(inference.seq_len, 102)
        self.assertEqual(inference.hidden_size, 768)
        self.assertEqual(inference.bact_classes, 2)
        self.assertEqual(inference.frame_classes, 6)
        self.assertEqual(inference.coding_classes, 2)
        self.assertEqual(inference.log_interval, 100)

    def test_model_paths_defined(self):
        """Test that model paths are defined for hidden_size=768."""
        import inference

        # For hidden_size=768, model paths should be defined
        self.assertIsNotNone(inference.bbert_model_path)
        self.assertIn('models', inference.bbert_model_path)
        self.assertIsNotNone(inference.bact_class_model_path)
        self.assertIsNotNone(inference.frame_class_model_path)
        self.assertIsNotNone(inference.class_model_path)

    def test_script_dir_and_bbert_dir(self):
        """Test that script_dir and bbert_dir are correctly set."""
        import inference

        # Verify script_dir and bbert_dir are defined
        self.assertIsNotNone(inference.script_dir)
        self.assertIsNotNone(inference.bbert_dir)
        self.assertTrue(os.path.isabs(inference.script_dir))
        self.assertTrue(os.path.isabs(inference.bbert_dir))


class TestInferenceWorkerConfiguration(unittest.TestCase):
    """Test num_workers configuration based on platform."""

    @patch('platform.system')
    @patch('multiprocessing.cpu_count')
    def test_num_workers_darwin(self, mock_cpu_count, mock_platform):
        """Test that num_workers=0 on macOS (Darwin)."""
        mock_platform.return_value = 'Darwin'
        mock_cpu_count.return_value = 8

        # Reimport to trigger module-level code
        import importlib
        import inference
        importlib.reload(inference)

        # On macOS, num_workers should be 0
        self.assertEqual(inference.num_workers, 0)

    @patch('platform.system')
    @patch('multiprocessing.cpu_count')
    def test_num_workers_windows(self, mock_cpu_count, mock_platform):
        """Test that num_workers=0 on Windows."""
        mock_platform.return_value = 'Windows'
        mock_cpu_count.return_value = 8

        # Reimport to trigger module-level code
        import importlib
        import inference
        importlib.reload(inference)

        # On Windows, num_workers should be 0
        self.assertEqual(inference.num_workers, 0)

    def test_num_workers_linux(self):
        """Test that num_workers>0 on Linux."""
        import inference

        # On non-Mac/Windows platforms, num_workers should be > 0
        # This test just verifies the current platform behavior
        # since we can't easily reload with different platform
        import platform
        if platform.system() not in ['Darwin', 'Windows']:
            self.assertGreater(inference.num_workers, 0)
        # On Mac/Windows, skip this specific test as we already test those

    def test_prefetch_factor_defined(self):
        """Test that prefetch_factor is defined."""
        import inference

        self.assertEqual(inference.prefetch_factor, 2)


class TestInferenceEnvironment(unittest.TestCase):
    """Test environment configuration."""

    def test_wandb_disabled(self):
        """Test that WANDB is disabled."""
        import inference

        # Check that WANDB_DISABLED is set
        self.assertEqual(os.environ.get('WANDB_DISABLED'), 'true')


if __name__ == '__main__':
    unittest.main(verbosity=2)
