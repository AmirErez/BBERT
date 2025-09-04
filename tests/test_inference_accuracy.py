#!/usr/bin/env python3
"""
Test BBERT inference accuracy using known ground truth sequences.

Uses example/example.fasta which contains:
- First 5 sequences: E. coli (should be bacterial)
- Last 5 sequences: Saccharomyces cerevisiae (should be non-bacterial)
"""

import unittest
import pandas as pd
import os
import subprocess
import tempfile
from pathlib import Path

class TestInferenceAccuracy(unittest.TestCase):
    """Test BBERT inference accuracy against known ground truth."""
    
    @classmethod
    def setUpClass(cls):
        """Run inference once for all tests."""
        cls.bbert_dir = Path(__file__).parent.parent
        cls.example_dir = cls.bbert_dir / "example"
        cls.example_fasta = cls.example_dir / "example.fasta"
        
        # Create temporary output directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.output_file = os.path.join(cls.temp_dir, "example_scores_len.parquet")
        
        # Run BBERT inference
        cmd = [
            "python", str(cls.bbert_dir / "source" / "inference.py"),
            "--input_dir", str(cls.example_dir),
            "--input_files", "example.fasta",
            "--output_dir", cls.temp_dir,
            "--batch_size", "64"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cls.bbert_dir)
        
        if result.returncode != 0:
            raise Exception(f"Inference failed: {result.stderr}")
        
        # Load results
        if not os.path.exists(cls.output_file):
            raise Exception(f"Output file not created: {cls.output_file}")
        
        cls.results = pd.read_parquet(cls.output_file)
        print(f"Loaded {len(cls.results)} results")
    
    def test_all_sequences_processed(self):
        """Test that all 10 sequences were processed."""
        self.assertEqual(len(self.results), 10, "Should process exactly 10 sequences")
    
    def test_output_format(self):
        """Test that output contains expected columns."""
        expected_columns = {'id', 'len', 'loss', 'bact_prob', 'frame_prob', 'coding_prob'}
        actual_columns = set(self.results.columns)
        
        self.assertTrue(expected_columns.issubset(actual_columns), 
                       f"Missing columns: {expected_columns - actual_columns}")
    
    def test_probability_ranges(self):
        """Test that probabilities are in valid ranges [0,1]."""
        # Test bacterial probabilities
        self.assertTrue((self.results['bact_prob'] >= 0).all(), "Bacterial probabilities should be ≥ 0")
        self.assertTrue((self.results['bact_prob'] <= 1).all(), "Bacterial probabilities should be ≤ 1")
        
        # Test coding probabilities  
        self.assertTrue((self.results['coding_prob'] >= 0).all(), "Coding probabilities should be ≥ 0")
        self.assertTrue((self.results['coding_prob'] <= 1).all(), "Coding probabilities should be ≤ 1")
    
    def test_frame_probabilities_sum(self):
        """Test that frame probabilities approximately sum to 1."""
        # Frame probabilities are stored as arrays, need to parse them
        for idx, frame_prob in enumerate(self.results['frame_prob']):
            frame_sum = sum(frame_prob)
            self.assertAlmostEqual(frame_sum, 1.0, places=2, 
                                 msg=f"Frame probabilities should sum to ~1.0 for sequence {idx}")
    
    def test_ecoli_bacterial_classification(self):
        """Test that E. coli sequences are classified as bacterial."""
        ecoli_results = self.results[self.results['id'].str.contains('NC_000913')]
        
        self.assertEqual(len(ecoli_results), 5, "Should have 5 E. coli sequences")
        
        # All E. coli sequences should have high bacterial probability
        ecoli_bacterial_probs = ecoli_results['bact_prob'].values
        
        print(f"E. coli bacterial probabilities: {ecoli_bacterial_probs}")
        
        # All E. coli sequences should be classified as bacterial (> 0.5)
        bacterial_count = sum(ecoli_bacterial_probs > 0.5)
        self.assertEqual(bacterial_count, 5, 
                        f"All 5 E. coli sequences should be bacterial, got {bacterial_count}/5")
        
        # Test that mean probability is high
        mean_prob = ecoli_bacterial_probs.mean()
        self.assertGreater(mean_prob, 0.5, 
                          f"Mean E. coli bacterial probability should be > 0.5, got {mean_prob:.3f}")
    
    def test_saccharomyces_non_bacterial_classification(self):
        """Test that Saccharomyces cerevisiae sequences are classified as non-bacterial."""
        saccharomyces_results = self.results[self.results['id'].str.contains('NC_001133')]
        
        self.assertEqual(len(saccharomyces_results), 5, "Should have 5 Saccharomyces cerevisiae sequences")
        
        # All Saccharomyces cerevisiae sequences should have low bacterial probability
        saccharomyces_bacterial_probs = saccharomyces_results['bact_prob'].values
        
        print(f"Saccharomyces cerevisiae bacterial probabilities: {saccharomyces_bacterial_probs}")
        
        # All Saccharomyces cerevisiae sequences should be classified as non-bacterial (< 0.5)
        non_bacterial_count = sum(saccharomyces_bacterial_probs < 0.5)
        self.assertEqual(non_bacterial_count, 5,
                        f"All 5 Saccharomyces cerevisiae sequences should be non-bacterial, got {non_bacterial_count}/5")
        
        # Test that mean probability is low
        mean_prob = saccharomyces_bacterial_probs.mean()
        self.assertLess(mean_prob, 0.5,
                       f"Mean Saccharomyces cerevisiae bacterial probability should be < 0.5, got {mean_prob:.3f}")
    
    def test_classification_separation(self):
        """Test that E. coli and Saccharomyces cerevisiae are clearly separated."""
        ecoli_results = self.results[self.results['id'].str.contains('NC_000913')]
        saccharomyces_results = self.results[self.results['id'].str.contains('NC_001133')]
        
        ecoli_mean = ecoli_results['bact_prob'].mean()
        saccharomyces_mean = saccharomyces_results['bact_prob'].mean()
        
        # E. coli should have higher bacterial probability than Saccharomyces cerevisiae
        self.assertGreater(ecoli_mean, saccharomyces_mean,
                          f"E. coli mean ({ecoli_mean:.3f}) should be > Saccharomyces cerevisiae mean ({saccharomyces_mean:.3f})")
        
        # E. coli should be above 0.5 threshold, Saccharomyces should be below
        self.assertGreater(ecoli_mean, 0.5, f"E. coli mean should be > 0.5, got {ecoli_mean:.3f}")
        self.assertLess(saccharomyces_mean, 0.5, f"Saccharomyces mean should be < 0.5, got {saccharomyces_mean:.3f}")
    
    def test_sequence_lengths(self):
        """Test that sequence lengths are correct."""
        # All sequences should be ~100bp (102bp including special tokens)
        lengths = self.results['len'].values
        
        self.assertTrue((lengths >= 100).all(), "All sequences should be ≥ 100bp")
        self.assertTrue((lengths <= 105).all(), "All sequences should be ≤ 105bp")
    
    def test_loss_values(self):
        """Test that loss values are reasonable."""
        losses = self.results['loss'].values
        
        # Loss should be positive
        self.assertTrue((losses > 0).all(), "All loss values should be positive")
        
        # Loss should be reasonable (not too high)
        self.assertTrue((losses < 10).all(), "Loss values should be < 10")
        
        print(f"Loss range: {losses.min():.3f} - {losses.max():.3f}")
    
    def test_detailed_results_summary(self):
        """Print detailed results for manual inspection."""
        print("\n" + "="*60)
        print("DETAILED RESULTS SUMMARY")
        print("="*60)
        
        for idx, row in self.results.iterrows():
            organism = "E. coli" if "NC_000913" in row['id'] else "S. cerevisiae"
            print(f"{row['id'][:20]:20} | {organism:12} | Bact: {row['bact_prob']:.3f} | "
                  f"Coding: {row['coding_prob']:.3f} | Loss: {row['loss']:.3f}")
        
        print("\nSUMMARY STATISTICS:")
        ecoli_mask = self.results['id'].str.contains('NC_000913')
        saccharomyces_mask = self.results['id'].str.contains('NC_001133')
        
        print(f"E. coli bacterial prob:           {self.results[ecoli_mask]['bact_prob'].mean():.3f} ± "
              f"{self.results[ecoli_mask]['bact_prob'].std():.3f}")
        print(f"Saccharomyces cerevisiae prob:    {self.results[saccharomyces_mask]['bact_prob'].mean():.3f} ± "
              f"{self.results[saccharomyces_mask]['bact_prob'].std():.3f}")

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)