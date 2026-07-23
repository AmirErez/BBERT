# Training scripts

These scripts were used to train the BBERT model and the downstream reading-frame
classifier described in the manuscript. They are provided for transparency and
reproducibility, not as a one-command pipeline: the raw pretraining corpus is too
large to distribute via git and is not included in this repository.

## `train.py` — pretrain BBERT (masked language modeling)

Trains a `BertForMaskedLM` from scratch (or resumes from a checkpoint found in
`--save_model_path`) on a tokenized DNA sequence dataset.

A runnable example, using the tiny dataset and tokenizer committed in this repo
(see below for what each one is):

```bash
python scripts/training/train.py \
    --dataset_path scripts/training/example_dataset \
    --tokenizer_path models/diverse_bact_12_768_6_20000/checkpoint-32500 \
    --save_model_path /tmp/bbert_smoke_test
```

The real training run used to produce the released model looked the same, just
with a full-size `--dataset_path` and a plain output directory:

```bash
python train.py \
    --dataset_path /path/to/real/pretrain_dataset \
    --tokenizer_path models/diverse_bact_12_768_6_20000/checkpoint-32500 \
    --save_model_path models/my_bbert_run
```

- `--dataset_path`: a dataset saved with `datasets.save_to_disk`, containing `train`
  and `val` splits with a `seq` column (100-nt reads). For example:

  ```pycon
  >>> from datasets import load_from_disk
  >>> load_from_disk("scripts/training/example_dataset")
  DatasetDict({
      train: Dataset({
          features: ['seq'],
          num_rows: 200
      })
      val: Dataset({
          features: ['seq'],
          num_rows: 50
      })
  })
  >>> load_from_disk("scripts/training/example_dataset")["train"][0]
  {'seq': 'ACCCGTCGCTGTTCGCCCCGCTGGATTTAGGTTTTACCACGTTAAAAAACCGCGTGTTGATGGGCTCAATGCACACCGGGCTGGAGGAATACCCGGACGG'}
  ```

  [`scripts/training/example_dataset/`](example_dataset/) is committed to this
  repo as a working example you can point `--dataset_path` at directly — it's
  200 train / 50 val reads sliced from the E. coli genome already checked in at
  [`examples/data/GCF_000005845_E_coli.fasta.gz`](../../examples/data/GCF_000005845_E_coli.fasta.gz),
  built by [`make_example_dataset.py`](make_example_dataset.py). It's only
  large enough to sanity-check the pipeline runs end to end and produces a
  model — not to train anything useful. To actually use it with `train.py`,
  first lower the `train_len` constant near the top of the file (e.g. to
  `100`): it defaults to `50_000_000`, i.e. it assumes a full-size dataset —
  with only 200 rows, `dataset['train'].select(range(train_len))` raises
  `IndexError`. (Verified: with `train_len = 100` and the tokenizer below,
  `train.py --dataset_path scripts/training/example_dataset ...` runs to
  completion and writes a model to `--save_model_path/best_model`.)

  The real pretraining dataset is not included in this repo (too large for
  git). Per the manuscript's Methods, it was built by taking 2 genomes from
  each bacterial family among ~200K curated
  [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) bacterial genomes (1519
  genomes total), then simulating reads from them with
  [InSilicoSeq](https://github.com/HadrienG/InSilicoSeq) using the NextSeq
  error model, keeping reads of at least 100 nt and truncating to 100 tokens.
  The manuscript's data availability statement does not publish this dataset
  or a genome accession list, so to reproduce it at full scale you'd need to
  regenerate it from RefSeq following that recipe (`make_example_dataset.py`
  shows the `datasets.DatasetDict(...).save_to_disk(...)` shape it needs to
  end up in), or contact the authors directly.
- `--tokenizer_path`: a tokenizer compatible with `transformers.AutoTokenizer`.
  It's a fixed character-level vocabulary of 5 tokens (`A`, `C`, `T`, `G`, `N`)
  plus special tokens (`<s>`, `</s>`, `<pad>`, `<msk>`, and a few more used by
  this repo's tokenizer). It's already committed to this repo (no
  `bbert download` needed for it) at
  [`models/diverse_bact_12_768_6_20000/checkpoint-32500/`](../../models/diverse_bact_12_768_6_20000/checkpoint-32500/)
  — just clone the repo and pass that directory:
  `--tokenizer_path models/diverse_bact_12_768_6_20000/checkpoint-32500`.
  Ignored when resuming from an existing checkpoint under `--save_model_path`
  (the tokenizer saved in the checkpoint is used instead).
- `--save_model_path`: output directory for checkpoints and the final model.

Model architecture (attention heads, hidden size, number of layers, training length)
is set via constants near the top of the file rather than CLI flags.

Additional dependencies beyond [requirements.txt](../../requirements.txt):
`datasets`, `torchinfo`.

## `emb_class_train.py` — train the reading-frame classifier

Trains a small classifier head on top of frozen BBERT embeddings to predict the
reading frame (1 of 6) of a DNA read.

```bash
python emb_class_train.py --dataset /path/to/reads.fasta
```

- `--dataset` (required): FASTA/FASTQ file of reads. Example files are available
  under [examples/data/](../../examples/data/).
- `--model_path` (default `models/diverse_bact_12_768_6_20000/checkpoint-32500`):
  pretrained BBERT checkpoint. Run `bbert download` first to fetch the weights.
- `--batch_size`, `--epochs`, `--max_reads`, `--output_model`: see `--help`.

## `emb_class_inference.py` — run the trained classifier

Runs a trained reading-frame classifier over a FASTA/FASTQ file and writes
predictions to CSV.

```bash
python emb_class_inference.py --dataset /path/to/reads.fasta
```

- `--dataset` (required): FASTA/FASTQ file of reads.
- `--model_path`, `--classifier_path`: paths to the pretrained BBERT checkpoint and
  the trained classifier weights, respectively.
- `--output`, `--batch_size`, `--max_reads`: see `--help`.
