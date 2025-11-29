"""
BBERT Model Download Command

Download BBERT models from Hugging Face Hub.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Hugging Face repository ID
HF_REPO_ID = "AmirErez/BBERT-models"

# Model configurations - maps HF paths to local paths
MODELS = {
    "bbert_main": {
        "hf_path": "bbert_checkpoint-32500",
        "local_path": "models/diverse_bact_12_768_6_20000/checkpoint-32500",
        "description": "BBERT transformer model",
        "is_folder": True
    },
    "bacterial_classifier": {
        "hf_path": "bacterial_classifier/epoch_80.pt",
        "local_path": "models/classifiers/bacterial/models/emb_class_model_768H_3906K_80e/epoch_80.pt",
        "description": "Bacterial classification model",
        "is_folder": False
    },
    "frame_classifier": {
        "hf_path": "frame_classifier/classifier_model_2000K_37e.pth",
        "local_path": "models/classifiers/frame/models/classifier_model_2000K_37e.pth",
        "description": "Reading frame classification model",
        "is_folder": False
    },
    "coding_classifier": {
        "hf_path": "coding_classifier/epoch_46.pt",
        "local_path": "models/classifiers/coding/models/emb_coding_model_768_3906K_50e/epoch_46.pt",
        "description": "Coding sequence classification model",
        "is_folder": False
    }
}


def add_arguments(parser):
    """Add download command arguments to parser."""
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
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to download models to (default: current directory)"
    )


def check_model_exists(local_path: Path) -> bool:
    """Check if model file or directory exists locally and is not a Git LFS pointer."""
    if local_path.is_file():
        if is_lfs_pointer(local_path):
            return False
        return True
    elif local_path.is_dir():
        model_file = local_path / "pytorch_model.bin"
        if model_file.exists() and not is_lfs_pointer(model_file):
            return True
    return False


def is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer file instead of actual content."""
    try:
        if file_path.stat().st_size > 200:
            return False

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if first_line.startswith('version https://git-lfs'):
                return True
    except Exception:
        pass

    return False


def download_all_models(bbert_root: Path, force: bool = False, repo_id: str = None):
    """Download all required BBERT models."""
    if repo_id is None:
        repo_id = HF_REPO_ID

    print(f"BBERT Model Downloader")
    print(f"=" * 60)
    print(f"Hugging Face Repository: {repo_id}")
    print(f"Local directory: {bbert_root}")
    print(f"=" * 60)
    print()

    # Check what's missing
    models_to_download = []
    for model_key, config in MODELS.items():
        local_path = bbert_root / config["local_path"]
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
        # Download to a temp directory
        temp_dir = bbert_root / ".hf_download_temp"
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
            dest_path = bbert_root / config["local_path"]

            try:
                # Create parent directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Move file or directory
                if source_path.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                else:
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


def main(args):
    """Main download function."""
    # Determine BBERT root directory
    if args.output_dir:
        bbert_root = Path(args.output_dir).absolute()
    else:
        # Use current directory
        bbert_root = Path.cwd()

    # Ensure directory exists
    bbert_root.mkdir(parents=True, exist_ok=True)

    try:
        success = download_all_models(bbert_root, force=args.force, repo_id=args.repo)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
