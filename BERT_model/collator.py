import re
from transformers import AutoTokenizer

# def get_tokenizer(tokenizer_path):
#     """Load tokenizer from path."""
#     return AutoTokenizer.from_pretrained(tokenizer_path)

def collate_fn(tokenizer, batch):
    """Custom collate function."""
    seq = [re.sub('[^ACTGN]', 'N', r['seq'].upper()) for r in batch]
    ids = [r['id'] for r in batch]
    encoded_seq = tokenizer(seq, truncation=True, padding='max_length', max_length=102, return_tensors='pt')['input_ids']
    return ids, encoded_seq

class CollateFnWithTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return collate_fn(self.tokenizer, batch)