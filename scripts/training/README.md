# Training scripts

These scripts were used to train the BBERT model and the downstream reading-frame
classifier described in the manuscript. They are provided for transparency and
reproducibility, not as a one-command pipeline: the raw pretraining corpus and the
base tokenizer are too large to distribute via git and are not included in this
repository.

## `train.py` — pretrain BBERT (masked language modeling)

Trains a `BertForMaskedLM` from scratch (or resumes from a checkpoint found in
`--save_model_path`) on a tokenized DNA sequence dataset.

```bash
python train.py \
    --dataset_path /path/to/dataset \
    --tokenizer_path /path/to/base_tokenizer \
    --save_model_path /path/to/output_dir
```

- `--dataset_path`: a dataset saved with `datasets.save_to_disk`, containing `train`
  and `val` splits with a `seq` column. Not included in this repo — see the
  manuscript's data availability statement for how to obtain or regenerate it.
- `--tokenizer_path`: a tokenizer compatible with `transformers.AutoTokenizer`,
  trained on the same corpus. Not included in this repo. Ignored when resuming from
  an existing checkpoint under `--save_model_path`.
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
