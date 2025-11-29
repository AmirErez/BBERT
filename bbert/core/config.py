"""
BBERT Configuration Constants

Central location for all configuration constants used across BBERT.
This improves maintainability and makes it easier to adjust parameters.
"""

# Model Architecture Constants
SEQUENCE_LENGTH = 102
"""Maximum sequence length for BBERT input"""

HIDDEN_SIZE_384 = 384
"""Hidden size for smaller BBERT models"""

HIDDEN_SIZE_768 = 768
"""Hidden size for standard BBERT models (default)"""

# Classification Constants
NUM_BACTERIAL_CLASSES = 2
"""Number of classes for bacterial classification (bacterial vs non-bacterial)"""

NUM_READING_FRAMES = 6
"""Number of reading frames (+1, +2, +3, -1, -2, -3)"""

NUM_CODING_CLASSES = 2
"""Number of classes for coding sequence classification (coding vs non-coding)"""

# Processing Constants
DEFAULT_BATCH_SIZE = 1024
"""Default batch size for inference"""

DEFAULT_CHUNK_MULTIPLIER = 2
"""Multiplier for chunk size relative to batch size"""

LOG_INTERVAL = 100
"""Number of batches between logging updates"""

# Model Paths (can be overridden via arguments)
DEFAULT_BBERT_MODEL_PATH_768 = "models/diverse_bact_12_768_6_20000/checkpoint-32500"
"""Default path for 768-dimensional BBERT model"""

DEFAULT_BBERT_MODEL_PATH_384 = "models/diverse_bact_3_384_6_50000Ks/checkpoint-7000"
"""Default path for 384-dimensional BBERT model"""

DEFAULT_BACTERIAL_CLASSIFIER_PATH = "models/classifiers/bacterial/models/emb_class_model_768H_3906K_80e/epoch_80.pt"
"""Default path for bacterial classifier"""

DEFAULT_FRAME_CLASSIFIER_PATH = "models/classifiers/frame/models/classifier_model_2000K_37e.pth"
"""Default path for reading frame classifier"""

DEFAULT_CODING_CLASSIFIER_PATH = "models/classifiers/coding/models/emb_coding_model_768_3906K_50e/epoch_46.pt"
"""Default path for coding sequence classifier"""

# Training Constants
DEFAULT_LEARNING_RATE = 1e-4
"""Default learning rate for Adam optimizer"""

DEFAULT_DROPOUT = 0.1
"""Default dropout rate for classifier layers"""

# Hardware Constants
DEFAULT_NUM_WORKERS = 0
"""Default number of DataLoader workers (0 = main process only, safer for multiprocessing)"""

DEFAULT_PREFETCH_FACTOR = 2
"""Default prefetch factor for DataLoader"""
