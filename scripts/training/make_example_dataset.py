"""
Build the tiny example dataset committed at scripts/training/example_dataset/.

This is NOT the real BBERT pretraining pipeline (which uses InSilicoSeq reads
simulated from 1519 RefSeq genomes, see scripts/training/README.md). It's a
minimal, reproducible stand-in that has the exact on-disk shape train.py
expects (a datasets.DatasetDict with "train"/"val" splits and a "seq"
column of 100-nt strings), built from the E. coli genome already checked
into examples/data/.

Run with the "bbert" conda/micromamba env (needs the `datasets` package,
see scripts/training/README.md):

    python scripts/training/make_example_dataset.py
"""

import gzip
import random

from datasets import Dataset, DatasetDict

GENOME_PATH = "examples/data/GCF_000005845_E_coli.fasta.gz"
OUTPUT_PATH = "scripts/training/example_dataset"
READ_LEN = 100
N_TRAIN = 200
N_VAL = 50

with gzip.open(GENOME_PATH, "rt") as f:
    f.readline()  # header
    genome = "".join(line.strip() for line in f)

random.seed(0)
n_reads = N_TRAIN + N_VAL
starts = random.sample(range(len(genome) - READ_LEN), n_reads)
reads = [genome[s : s + READ_LEN] for s in starts]

dataset = DatasetDict(
    {
        "train": Dataset.from_dict({"seq": reads[:N_TRAIN]}),
        "val": Dataset.from_dict({"seq": reads[N_TRAIN:]}),
    }
)
dataset.save_to_disk(OUTPUT_PATH)
print(dataset)
