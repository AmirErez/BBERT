import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertModel
from torch.utils.data import DataLoader, Dataset

from BERT_model.utils import setup_logger

verbose = True  # or use argparse to pass --verbose
logger = setup_logger(verbose)

class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        # Linear layer to classify the entire sequence
        self.classifier = nn.Linear(hidden_size * 102, num_classes)  # Multiply by seq_length (102)

    def forward(self, x):
        # Flatten the sequence embeddings for each sample to [batch_size, hidden_size * seq_length]
        x = x.view(x.size(0), -1)  # Flatten along the sequence length dimension
        
        # Apply the classifier to the flattened embeddings
        logits = self.classifier(x)  # Shape: [batch_size, num_classes]
        
        return logits
    
class emb_CNN_classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(emb_CNN_classifier, self).__init__()
        
        # Convolution with 1 input channel and 32 output filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Channel-wise normalization

        # Convolution with 32 input channels and 64 output filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Channel-wise normalization
        
        # Adaptive layer to compress output to (batch_size, 64, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification into 6 classes
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: (batch_size, 1, 102, 768)
        x = x.unsqueeze(1)  # Add channel dimension → (batch_size, 1, 102, 768)
        
        # Apply first convolution + activation + normalization
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 32, 102, 768)
        
        # Apply second convolution + activation + normalization
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, 102, 768)
        
        # Adaptive compression of feature maps
        x = self.pool(x)  # (batch_size, 64, 1, 1)
        
        # Reshape into 2D for the fully connected layer
        x = x.view(x.size(0), -1)  # (batch_size, 64)
        
        # Apply fully connected layer to get classification over 6 classes
        x = self.fc(x)  # (batch_size, 6)
        
        return x


class emb_CNN_classifier_16f(nn.Module):
    def __init__(self, num_classes=6):
        super(emb_CNN_classifier_16f, self).__init__()

        # Single convolution: input channel = 1, output = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Adaptive pooling to compress to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer from 16 channels to num_classes
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # If input is (batch_size, 102, 768), unsqueeze to add channel dim
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, 102, 768)

        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 16, 102, 768)
        x = self.pool(x)                     # (batch_size, 16, 1, 1)
        x = x.view(x.size(0), -1)            # (batch_size, 16)
        x = self.fc(x)                       # (batch_size, num_classes)

        return x

class BertMeanPoolClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),    # Reduce dimension from 768 → 256
            nn.ReLU(),                      # Add non-linearity
            nn.Dropout(0.1),                # Randomly drop 10% of neurons during training
            nn.Linear(256, num_classes)     # Final classifier to output class logits (e.g., 6 classes)
        )
    def forward(self, embeddings, attention_mask=None):
        if attention_mask is not None:
            # Mean pooling with mask
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            summed = torch.sum(embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            x = summed / counts
        else:
            # Mean pooling without mask
            x = embeddings.mean(dim=1)  # [batch, 768]
        return self.classifier(x)

class emb_CNN_6st_classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(emb_CNN_6st_classifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Adaptive pooling to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: (batch_size, 102, 768)
        # Sum every 6th row to reduce to (batch_size, 6, 768)
        x = torch.stack([x[:, i::6, :].sum(dim=1) for i in range(6)], dim=1)

        # Add channel dimension: (batch_size, 1, 6, 768)
        x = x.unsqueeze(1)

        # Convolution blocks
        x = F.relu(self.bn1(self.conv1(x)))  # → (batch_size, 32, 6, 768)
        x = F.relu(self.bn2(self.conv2(x)))  # → (batch_size, 64, 6, 768)

        # Adaptive average pooling
        x = self.pool(x)  # → (batch_size, 64, 1, 1)

        # Flatten and classify
        x = x.view(x.size(0), -1)  # → (batch_size, 64)
        x = self.fc(x)             # → (batch_size, num_classes)

        return x