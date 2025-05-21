
import psutil
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
import gc
import logging
import io
nvmlInit()

def setup_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False  # Prevent messages from propagating to root logger

    if not logger.handlers:  # Only check this logger's handlers
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

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
    
def log_resources(msg: str):
    """Log CPU RAM and GPU usage during training."""
    # CPU RAM Usage
    
    cpu_mem = psutil.virtual_memory().used / 1e9
    log_msg = msg + f" \tCPU RAM Used: {cpu_mem:.2f} GB"

    # GPU Usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            handle = nvmlDeviceGetHandleByIndex(i)
            gpu_mem = nvmlDeviceGetMemoryInfo(handle)
            gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu
            log_msg += f" \tGPU {i}: {gpu_mem.used / 1e9:.2f} GB / {gpu_mem.total / 1e9:.2f} GB, Utilization: {gpu_util}%"
    setup_logger(True).info(log_msg)

def clear_GPU():
    for obj in list(globals()):
        if isinstance(globals()[obj], torch.Tensor) and globals()[obj].is_cuda:
            del globals()[obj]
    torch.cuda.empty_cache()
    gc.collect()
    
def tensor_memory_summary(scope_dict, min_MB=1.0):
    import torch

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
