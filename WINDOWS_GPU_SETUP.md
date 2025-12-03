# Windows GPU Setup Guide for BBERT

This guide addresses the common issue where Windows installations fail to activate the GPU even when CUDA is available.

## The Problem

When installing BBERT on Windows using conda, the environment may install CPU-only PyTorch even though:
- You have an NVIDIA GPU
- CUDA is properly installed
- The environment file specifies CUDA packages

This happens because conda's channel resolution on Windows can prioritize CPU packages or fail to properly link CUDA libraries.

## The Solution

Use a two-step installation process:

### Step 1: Create the conda environment (without PyTorch)

```bash
conda env create -f BBERT_env_windows.yml
conda activate BBERT_windows
```

The updated `BBERT_env_windows.yml` file intentionally excludes PyTorch to avoid conda's unreliable GPU package installation on Windows.

### Step 2: Install PyTorch with explicit CUDA support via pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

This command:
- Uses pip instead of conda for PyTorch installation
- Explicitly specifies the CUDA 12.4 wheel repository
- Guarantees GPU-enabled PyTorch installation

### Step 3: Verify GPU detection

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU name: NVIDIA GeForce RTX 3080  # (or your GPU model)
```

## If You Already Installed Without GPU Support

If you already created the environment and PyTorch is using CPU only:

```bash
# Activate your environment
conda activate BBERT_windows

# Remove existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Reinstall with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Why This Works

1. **Conda channel conflicts**: On Windows, conda may mix packages from `pytorch`, `nvidia`, and `conda-forge` channels in ways that break CUDA support

2. **Package resolution**: Conda's dependency resolver on Windows sometimes selects CPU versions to avoid conflicts

3. **PyTorch wheels**: PyTorch's official pip wheels (`--index-url https://download.pytorch.org/whl/cu124`) are built specifically for CUDA and tested by PyTorch maintainers

4. **Explicit selection**: Using pip with the CUDA index URL explicitly requests GPU versions, bypassing conda's resolution logic

## Alternative: CPU-Only Installation

If you don't have an NVIDIA GPU or want to use CPU only:

```bash
conda env create -f BBERT_env_windows_cpu.yml
conda activate BBERT_windows_cpu
```

## Verifying CUDA Installation

Before installing BBERT, verify CUDA is working:

```bash
# Check NVIDIA driver and CUDA
nvidia-smi
```

This should show your GPU and CUDA version. If this command fails, install or update your NVIDIA drivers from https://www.nvidia.com/drivers

## Troubleshooting

### "nvidia-smi not found"
- Install NVIDIA GPU drivers: https://www.nvidia.com/drivers
- Restart your computer after installation

### "CUDA available: False" after following all steps
- Verify CUDA version compatibility: `nvidia-smi` should show CUDA 12.4 or compatible
- If your CUDA version is different, use the appropriate wheel:
  - CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
  - CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
  - See https://pytorch.org/get-started/locally/ for other versions

### ImportError with torch after installation
- Make sure you're in the correct environment: `conda activate BBERT_windows`
- Reinstall PyTorch: Follow the uninstall/reinstall steps above

## Performance Expectations

With proper GPU setup:
- **GPU**: ~10,000-50,000 sequences/second (depending on GPU and batch size)
- **CPU only**: ~500-2,000 sequences/second

If you're seeing CPU-like speeds on a GPU system, the GPU likely isn't being used.

## Need Help?

If you're still experiencing issues:
1. Check existing issues: https://github.com/AmirErez/BBERT/issues
2. Create a new issue with:
   - Output of `nvidia-smi`
   - Output of the verification command above
   - Your Windows version
   - Complete error messages
