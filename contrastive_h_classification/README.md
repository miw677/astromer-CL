# Contrastive H Classification

This folder contains the independent pooled-`h` downstream classification
pipeline. It does not modify the OG official classifier.

## Main script

```bash
python contrastive_h_classification/train_pooled_h_classifier.py \
  --data_root data/records/alcock/alcock/fold_0/alcock \
  --pretrained_path weights/macho_v2_2025 \
  --stage1_encoder_checkpoint weights/contrastive_stage1/encoder_epoch_5.weights.h5 \
  --stage2_encoder_checkpoint weights/contrastive_stage2/encoder_stage2_fresh_epoch_6.weights.h5 \
  --output_dir contrastive_h_classification/runs/frozen_head \
  --epochs 1000000 \
  --batch_size 128
```

By default it runs:

```text
OG encoder -> masked mean pooled h -> AstromerStyleMLPHead
Stage1 encoder -> masked mean pooled h -> AstromerStyleMLPHead
Stage2 encoder -> masked mean pooled h -> AstromerStyleMLPHead
```

The encoder is frozen by default. Use `--no-freeze_encoder` only for a later
fine-tuning ablation.

Training control is aligned with the OG official classification pipeline:

```text
epochs = 1000000
monitor = val_loss
patience = 40
restore best head before test
```

For smoke tests, use a small dataset plus explicit batch limits, for example:

```bash
python contrastive_h_classification/train_pooled_h_classifier.py \
  --data_root data/records/alcock/alcock/fold_0/alcock_20 \
  --output_dir contrastive_h_classification/runs/smoke_alcock20_cpu \
  --epochs 1 \
  --batch_size 16 \
  --max_train_batches 2 \
  --max_val_batches 2 \
  --max_test_batches 2
```

## Outputs

Each system gets its own subfolder with:

```text
metrics.json
predictions.npz
best_head.weights.h5
```

The run folder also gets:

```text
run_config.json
results_table.md
results_table.json
```

To include an external OG official anchor row in the summary table, pass:

```bash
--official_macro_f1 0.6
```

or pass the exact reproduced value after running the official pipeline.
