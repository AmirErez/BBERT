import re
from transformers import AutoTokenizer
import numpy as np

# def collate_fn(tokenizer, batch):
#     seq = [re.sub('[^ACTGN]', 'N', r['seq'].upper()) for r in batch]
#     ids = np.array([r['id'] for r in batch])
#     encoded = tokenizer(seq, truncation=True, padding='max_length', max_length=102, return_tensors='pt', return_attention_mask=True)
#     return ids, encoded['input_ids'], encoded['attention_mask']

REGEX = re.compile('[^ACTGN]')

def collate_fn(tokenizer, batch):
    seq = [REGEX.sub('N', r['seq'][:100].upper()) for r in batch]
    ids = np.array([r['id'] for r in batch])
    encoded = tokenizer(seq, truncation=True, padding='max_length', max_length=102, return_tensors='pt', return_attention_mask=True)
    return ids, encoded['input_ids'], encoded['attention_mask']

class CollateFnWithTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return collate_fn(self.tokenizer, batch)

# REGEX = re.compile('[^ACTGN]')  # compile once at module level

# class CollateFnWithTokenizer:
#     def __init__(self, tokenizer, max_length=102):
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __call__(self, batch):
#         # string cleaning
#         seqs = []
#         ids = []
#         for r in batch:
#             seqs.append(REGEX.sub('N', r['seq'].upper()))
#             ids.append(r['id'])  # keep as list

#         # batch tokenization
#         encoded = self.tokenizer(
#             seqs,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )

#         return ids, encoded['input_ids'], encoded['attention_mask']