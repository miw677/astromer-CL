"""Extract pooled encoder embeddings (h) from trained contrastive checkpoints.

This script is checkpoint-agnostic for Stage 2 variants:
- fresh implementation checkpoints (full model or encoder-only)
- non-fresh implementation checkpoints (full model or encoder-only)

Output format per split: compressed npz with keys
- embeddings: [N, D] pooled encoder representations (h)
- labels: [N] integer class labels
- sample_ids: [N] sequential IDs within split
- lengths: [N] valid (unpadded) sequence lengths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.contrastive_astromer import build_contrastive_model_from_pretrained
from src.data.split_utils import collect_record_files
from src.data.contrastive_record_utils import detect_record_schema, parse_contrastive_sample


RECORD_SCHEMA = "dim"


def parse_labeled_tfrecord(example_proto):
    """Parse one labeled SequenceExample for encoder inference."""
    return parse_contrastive_sample(example_proto, schema=RECORD_SCHEMA)


def center_crop_to_window(sample, window_size):
    """Deterministic center crop for stable downstream embeddings."""
    seq_len = tf.shape(sample["input"])[0]

    def _crop():
        start = (seq_len - window_size) // 2
        end = start + window_size
        return {
            "input": sample["input"][start:end],
            "times": sample["times"][start:end],
            "mask_in": sample["mask_in"][start:end],
            "label": sample["label"],
        }

    return tf.cond(seq_len > window_size, _crop, lambda: sample)


def build_inference_dataset_from_raw(raw_dataset, window_size, batch_size):
    dataset = raw_dataset
    dataset = dataset.map(parse_labeled_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda s: center_crop_to_window(s, window_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    padded_shapes = {
        "input": [window_size, 1],
        "times": [window_size, 1],
        "mask_in": [window_size, 1],
        "label": [],
    }
    padding_values = {
        "input": tf.constant(0.0, dtype=tf.float32),
        "times": tf.constant(0.0, dtype=tf.float32),
        "mask_in": tf.constant(0.0, dtype=tf.float32),
        "label": tf.constant(0, dtype=tf.int32),
    }

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_inference_dataset(record_files, window_size, batch_size):
    raw_dataset = tf.data.TFRecordDataset(record_files, num_parallel_reads=tf.data.AUTOTUNE)
    return build_inference_dataset_from_raw(raw_dataset, window_size=window_size, batch_size=batch_size)


def masked_mean_pool(h_seq, mask):
    """Pool sequence output h=[B,T,D] to pooled embeddings [B,D]."""
    mask = tf.cast(mask, h_seq.dtype)
    h_sum = tf.reduce_sum(h_seq * mask, axis=1)
    denom = tf.reduce_sum(mask, axis=1)
    return h_sum / tf.maximum(denom, tf.constant(1e-8, dtype=h_seq.dtype))


def extract_split_embeddings(model, dataset, max_batches=None):
    embs = []
    labels = []
    lengths = []

    batch_count = 0
    for batch in dataset:
        if max_batches is not None and batch_count >= max_batches:
            break

        _, h_seq = model(batch, training=False, return_representation=True)
        h_pooled = masked_mean_pool(h_seq, batch["mask_in"])

        embs.append(h_pooled.numpy().astype(np.float32))
        labels.append(batch["label"].numpy().astype(np.int32))

        valid_lengths = np.sum(batch["mask_in"].numpy(), axis=(1, 2)).astype(np.int32)
        lengths.append(valid_lengths)

        batch_count += 1

    if not embs:
        return {
            "embeddings": np.zeros((0, 0), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
            "sample_ids": np.zeros((0,), dtype=np.int64),
            "lengths": np.zeros((0,), dtype=np.int32),
            "num_batches": 0,
        }

    emb_arr = np.concatenate(embs, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    length_arr = np.concatenate(lengths, axis=0)
    sample_ids = np.arange(emb_arr.shape[0], dtype=np.int64)

    return {
        "embeddings": emb_arr,
        "labels": label_arr,
        "sample_ids": sample_ids,
        "lengths": length_arr,
        "num_batches": batch_count,
    }


def _collect_split_dirs(args):
    split_dirs = {}
    if args.train_dir:
        split_dirs["train"] = args.train_dir
    if args.val_dir:
        split_dirs["val"] = args.val_dir
    if args.test_dir:
        split_dirs["test"] = args.test_dir

    if not split_dirs:
        raise ValueError("Provide at least one of --train_dir, --val_dir, --test_dir")

    return split_dirs


def _validate_split_fractions(train_frac, val_frac, test_frac):
    fractions = [float(train_frac), float(val_frac), float(test_frac)]
    if any(fr <= 0.0 for fr in fractions):
        raise ValueError("Split fractions must all be > 0")
    total = sum(fractions)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.6f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )


def _build_hash_split_filter(seed, bucket_start, bucket_end, num_buckets=10000):
    seed_suffix = tf.constant(f"|seed={int(seed)}")

    def _fn(example_proto):
        key = tf.strings.join([example_proto, seed_suffix])
        bucket = tf.strings.to_hash_bucket_fast(key, num_buckets=num_buckets)
        return tf.logical_and(bucket >= bucket_start, bucket < bucket_end)

    return _fn


def _collect_split_sources(args):
    has_explicit_split_dir = any([args.train_dir, args.val_dir, args.test_dir])
    has_full_data_dir = args.full_data_dir is not None

    if has_full_data_dir and has_explicit_split_dir:
        raise ValueError(
            "Use either --full_data_dir OR explicit split dirs (--train_dir/--val_dir/--test_dir), not both"
        )

    if has_full_data_dir:
        _validate_split_fractions(args.split_train, args.split_val, args.split_test)
        full_record_files = [str(path) for path in collect_record_files(args.full_data_dir)]
        if not full_record_files:
            raise ValueError(f"No .record files found in full data dir: {args.full_data_dir}")

        bucket_size = 10000
        train_bins = int(round(args.split_train * bucket_size))
        val_bins = int(round(args.split_val * bucket_size))
        # Keep the tail for test so train+val+test covers all buckets.
        test_bins = bucket_size - train_bins - val_bins
        if train_bins <= 0 or val_bins <= 0 or test_bins <= 0:
            raise ValueError(
                "Split bins became non-positive; adjust split fractions so each split is meaningful"
            )

        print(
            f"[SPLIT] full_data_dir mode with hash split: "
            f"train={args.split_train}, val={args.split_val}, test={args.split_test}, seed={args.split_seed}"
        )

        full_dataset = tf.data.TFRecordDataset(full_record_files, num_parallel_reads=tf.data.AUTOTUNE)
        train_raw = full_dataset.filter(
            _build_hash_split_filter(args.split_seed, 0, train_bins, num_buckets=bucket_size)
        )
        val_raw = full_dataset.filter(
            _build_hash_split_filter(
                args.split_seed, train_bins, train_bins + val_bins, num_buckets=bucket_size
            )
        )
        test_raw = full_dataset.filter(
            _build_hash_split_filter(
                args.split_seed, train_bins + val_bins, bucket_size, num_buckets=bucket_size
            )
        )

        return {
            "train": {
                "raw_dataset": train_raw,
                "input_dir": args.full_data_dir,
                "num_record_files": len(full_record_files),
                "split_mode": "hash_from_full_data_dir",
            },
            "val": {
                "raw_dataset": val_raw,
                "input_dir": args.full_data_dir,
                "num_record_files": len(full_record_files),
                "split_mode": "hash_from_full_data_dir",
            },
            "test": {
                "raw_dataset": test_raw,
                "input_dir": args.full_data_dir,
                "num_record_files": len(full_record_files),
                "split_mode": "hash_from_full_data_dir",
            },
        }

    split_sources = {}
    if args.train_dir:
        split_sources["train"] = {
            "record_files": [str(path) for path in collect_record_files(args.train_dir)],
            "input_dir": args.train_dir,
            "split_mode": "explicit_dir",
        }
    if args.val_dir:
        split_sources["val"] = {
            "record_files": [str(path) for path in collect_record_files(args.val_dir)],
            "input_dir": args.val_dir,
            "split_mode": "explicit_dir",
        }
    if args.test_dir:
        split_sources["test"] = {
            "record_files": [str(path) for path in collect_record_files(args.test_dir)],
            "input_dir": args.test_dir,
            "split_mode": "explicit_dir",
        }

    if not split_sources:
        raise ValueError(
            "Provide either --full_data_dir OR at least one explicit split dir "
            "(--train_dir, --val_dir, --test_dir)"
        )

    return split_sources


def main():
    parser = argparse.ArgumentParser(description="Extract pooled h embeddings from contrastive model.")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, default=None)
    parser.add_argument("--stage2_encoder_checkpoint", type=str, default=None)

    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--full_data_dir", type=str, default=None)
    parser.add_argument("--split_train", type=float, default=0.7)
    parser.add_argument("--split_val", type=float, default=0.15)
    parser.add_argument("--split_test", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--max_batches_per_split", type=int, default=None)

    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.stage2_checkpoint and args.stage2_encoder_checkpoint:
        raise ValueError("Use only one of --stage2_checkpoint or --stage2_encoder_checkpoint")

    split_sources = _collect_split_sources(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global RECORD_SCHEMA
    schema_source = args.full_data_dir or next(iter(split_sources.values()))["input_dir"]
    RECORD_SCHEMA = detect_record_schema(schema_source)

    print("[MODEL] Building model from pretrained architecture/config")
    model, pretrained_cfg = build_contrastive_model_from_pretrained(
        pretrained_path=args.pretrained_path,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
    )

    if args.stage2_checkpoint:
        model.load_weights(args.stage2_checkpoint)
        print(f"[MODEL] Loaded full stage2 checkpoint: {args.stage2_checkpoint}")
    elif args.stage2_encoder_checkpoint:
        model.encoder.load_weights(args.stage2_encoder_checkpoint)
        print(f"[MODEL] Loaded stage2 encoder checkpoint: {args.stage2_encoder_checkpoint}")
    else:
        print("[MODEL] No Stage 2 checkpoint provided; using pretrained encoder weights only.")

    window_size = int(pretrained_cfg["window_size"])
    print(f"[MODEL] Using window_size={window_size}")

    manifest = {
        "pretrained_path": args.pretrained_path,
        "stage2_checkpoint": args.stage2_checkpoint,
        "stage2_encoder_checkpoint": args.stage2_encoder_checkpoint,
        "mask_semantics": "contrastive mask_in is visible; model always sends 1-mask_in to encoder",
        "window_size": window_size,
        "split": {
            "mode": "full_data_dir" if args.full_data_dir else "explicit_dirs",
            "full_data_dir": args.full_data_dir,
            "fractions": {
                "train": args.split_train,
                "val": args.split_val,
                "test": args.split_test,
            },
            "seed": args.split_seed,
        },
        "splits": {},
    }

    for split_name, split_info in split_sources.items():
        print("\n" + "=" * 72)
        input_dir = split_info["input_dir"]
        print(f"[EXTRACT] split={split_name} dir={input_dir}")

        if "raw_dataset" in split_info:
            dataset = build_inference_dataset_from_raw(
                raw_dataset=split_info["raw_dataset"],
                window_size=window_size,
                batch_size=args.batch_size,
            )
            num_record_files = int(split_info.get("num_record_files", 0))
        else:
            record_files = split_info["record_files"]
            print(f"[EXTRACT] found {len(record_files)} record files")
            dataset = build_inference_dataset(
                record_files=record_files,
                window_size=window_size,
                batch_size=args.batch_size,
            )
            num_record_files = len(record_files)

        out = extract_split_embeddings(
            model=model,
            dataset=dataset,
            max_batches=args.max_batches_per_split,
        )

        split_out_path = output_dir / f"{split_name}.npz"
        np.savez_compressed(
            split_out_path,
            embeddings=out["embeddings"],
            labels=out["labels"],
            sample_ids=out["sample_ids"],
            lengths=out["lengths"],
        )

        shape = tuple(out["embeddings"].shape)
        print(f"[EXTRACT] saved {split_out_path} with embeddings shape={shape}")

        manifest["splits"][split_name] = {
            "input_dir": input_dir,
            "split_mode": split_info.get("split_mode", "unknown"),
            "num_record_files": num_record_files,
            "num_samples": int(out["embeddings"].shape[0]),
            "embedding_dim": int(out["embeddings"].shape[1]) if out["embeddings"].ndim == 2 else 0,
            "num_batches": int(out["num_batches"]),
            "output_npz": str(split_out_path),
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[DONE] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
