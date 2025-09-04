import psutil
import torch
import os
import logging
import io
import gc

# Optional NVIDIA GPU monitoring (not available on Mac)
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
    nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Global singleton logger
LOGGER = None

def setup_logger(verbose: bool = False, log_file: str = None) -> logging.Logger:
    global LOGGER
    if LOGGER is not None:
        return LOGGER

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Add console handler once
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Add file handler if log_file is provided and not already added
    if log_file and not any(
        isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file)
        for h in logger.handlers
    ):
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_file}")

    LOGGER = logger
    return logger

def get_true_label(id):
    frame = int(id.split('|')[-1])
    if frame < 0:
        label = frame + 3
    else:
        label = frame + 2
    return int(label)

def label_to_frame(label:int) -> int:
    if label >= 3:
        return label - 2
    elif label < 3:
        return label -3
    
def get_frame(id:str) -> int:
    return int(id.split('|')[-1])
    
def clear_GPU():
    """Safely clear GPU and CPU memory caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have equivalent cache clearing functions yet
        pass
    gc.collect()

def get_resources_msg() -> str:
    """Generate a string reporting current RAM (all processes) and GPU usage."""
    process = psutil.Process(os.getpid())
    # Get memory used by this process and all its children
    mem_total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            mem_total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    msg = f"Slurm task RAM (RSS): {mem_total / 1e9:.2f} GB"

    # GPU usage
    if torch.cuda.is_available() and PYNVML_AVAILABLE:
        for i in range(torch.cuda.device_count()):
            handle = nvmlDeviceGetHandleByIndex(i)
            gpu_mem = nvmlDeviceGetMemoryInfo(handle)
            gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu
            msg += f" | GPU {i}: {gpu_mem.used / 1e9:.2f} GB / {gpu_mem.total / 1e9:.2f} GB, Util: {gpu_util}%"
    elif torch.cuda.is_available():
        msg += f" | GPU: CUDA available but monitoring unavailable"

    return msg
    
def tensor_memory_summary(scope_dict, min_MB=1.0):

    summary = []
    total_MB = 0

    for name, var in scope_dict.items():
        if isinstance(var, torch.Tensor):
            numel = var.numel()
            size_bytes = var.element_size() * numel
            size_MB = size_bytes / (1024 ** 2)
            if size_MB >= min_MB:
                summary.append({
                    "name": name,
                    "shape": tuple(var.shape),
                    "dtype": str(var.dtype),
                    "device": str(var.device),
                    "size_MB": round(size_MB, 2)
                })
            total_MB += size_MB

    summary = sorted(summary, key=lambda x: x["size_MB"], reverse=True)

    for item in summary:
        setup_logger(verbose=True).info(f"{item['name']:<30} {str(item['shape']):<20} {item['dtype']:<10} {item['device']:<8} {item['size_MB']:>8} MB")

    setup_logger(verbose=True).info(f"\nTotal tensor memory usage: {round(total_MB, 2)} MB")
    return summary

def model_info(model, verbose: bool = False):
    """Prints the model architecture and number of parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        setup_logger(verbose=True).info(f"Model architecture: {model}")
        setup_logger(verbose=True).info(f"Total parameters: {total_params:,}")
        
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / 1e6
    setup_logger(verbose=True).info(f"Model size: {size_mb:.2f} MB")
