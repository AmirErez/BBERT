#!/usr/bin/env python3
"""
Download BBERT models from Hugging Face Hub.

This script downloads the required model files from Hugging Face and places them
in the correct directory structure for BBERT to use.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Hugging Face repository ID (will be set after you upload)
# Format: "username/repo-name" or "organization/repo-name"
HF_REPO_ID = "AmirErez/BBERT-models"  # Change this to your HF repo

# Get BBERT root directory (parent of source/)
BBERT_ROOT = Path(__file__).parent.parent

# Model configurations - maps HF paths to local paths
MODELS = {
    "bbert_main": {
        "hf_path": "bbert_checkpoint-32500",  # Folder in HF repo
        "local_path": "models/diverse_bact_12_768_6_20000/checkpoint-32500",
        "description": "BBERT transformer model",
        "is_folder": True
    },
    "bacterial_classifier": {
        "hf_path": "bacterial_classifier/epoch_80.pt",
        "local_path": "emb_class_bact/models/emb_class_model_768H_3906K_80e/epoch_80.pt",
        "description": "Bacterial classification model",
        "is_folder": False
    },
    "frame_classifier": {
        "hf_path": "frame_classifier/classifier_model_2000K_37e.pth",
        "local_path": "emb_class_frame/models/classifier_model_2000K_37e.pth",
        "description": "Reading frame classification model",
        "is_folder": False
    },
    "coding_classifier": {
        "hf_path": "coding_classifier/epoch_46.pt",
        "local_path": "emb_class_coding/models/emb_coding_model_768_3906K_50e/epoch_46.pt",
        "description": "Coding sequence classification model",
        "is_folder": False
    }
}


def check_model_exists(local_path: Path) -> bool:
    """Check if model file or directory exists locally and is not a Git LFS pointer."""
    if local_path.is_file():
        # Check if it's a real file, not a Git LFS pointer
        if is_lfs_pointer(local_path):
            return False
        return True
    elif local_path.is_dir():
        # For directories, check if key files exist and are real
        model_file = local_path / "pytorch_model.bin"
        if model_file.exists() and not is_lfs_pointer(model_file):
            return True
    return False


def is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer file instead of actual content."""
    try:
        # LFS pointer files are small text files (< 200 bytes)
        if file_path.stat().st_size > 200:
            return False

        # Check if it starts with "version https://git-lfs"
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if first_line.startswith('version https://git-lfs'):
                return True
    except Exception:
        # If we can't read it, assume it's a real binary file
        pass

    return False


def download_all_models(force: bool = False, repo_id: str = None):
    """Download all required BBERT models."""
    if repo_id is None:
        repo_id = HF_REPO_ID

    print(f"BBERT Model Downloader")
    print(f"=" * 60)
    print(f"Hugging Face Repository: {repo_id}")
    print(f"Local directory: {BBERT_ROOT}")
    print(f"=" * 60)
    print()

    # Check what's missing
    models_to_download = []
    for model_key, config in MODELS.items():
        local_path = BBERT_ROOT / config["local_path"]
        if force or not check_model_exists(local_path):
            models_to_download.append((model_key, config))
        else:
            print(f"[SKIP] {config['description']} (already exists)")
            print(f"       {local_path}")
            print()

    if not models_to_download:
        print("=" * 60)
        print("All models already downloaded!")
        print("=" * 60)
        return True

    # Download entire repository to temp location
    print(f"Downloading models from Hugging Face Hub...")
    print()

    try:
        # Download to a temp directory within BBERT root
        temp_dir = BBERT_ROOT / ".hf_download_temp"
        temp_dir.mkdir(exist_ok=True)

        # Download the entire repo
        cache_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
        )

        print()
        print("Moving files to correct locations...")
        print()

        # Move files to correct locations
        success_count = 0
        fail_count = 0

        for model_key, config in models_to_download:
            source_path = temp_dir / config["hf_path"]
            dest_path = BBERT_ROOT / config["local_path"]

            try:
                # Create parent directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Move file or directory
                if source_path.is_dir():
                    # For directories, copy contents
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                else:
                    # For files, just copy
                    shutil.copy2(source_path, dest_path)

                print(f"[OK] {config['description']}")
                print(f"     -> {dest_path}")
                print()
                success_count += 1

            except Exception as e:
                print(f"[ERROR] {config['description']}: {e}")
                print()
                fail_count += 1

        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory {temp_dir}: {e}")

        # Summary
        print("=" * 60)
        print(f"Download Summary:")
        print(f"  Downloaded: {success_count}")
        print(f"  Failed: {fail_count}")
        print("=" * 60)

        if fail_count > 0:
            print("\nWARNING: Some models failed to download.")
            print("Please check your internet connection and HF repository access.")
            return False

        print("\nAll models ready!")
        return True

    except Exception as e:
        print(f"\n\nERROR downloading from Hugging Face: {e}")
        print("\nPossible issues:")
        print("  1. Repository doesn't exist or is private")
        print("  2. No internet connection")
        print("  3. Firewall blocking Hugging Face")
        print(f"\nRepository URL: https://huggingface.co/{repo_id}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download BBERT models from Hugging Face Hub"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=HF_REPO_ID,
        help=f"Hugging Face repository ID (default: {HF_REPO_ID})"
    )

    args = parser.parse_args()

    try:
        success = download_all_models(force=args.force, repo_id=args.repo)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
