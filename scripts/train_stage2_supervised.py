"""
Fresh Stage 2 (Supervised Contrastive) training script.

This file intentionally does not reuse the old Stage 2 trainer logic so we can
test a clean implementation path end-to-end.

Pipeline:
    labeled TFRecords
        -> parse + random crop
        -> two views (augmented or identity)
        -> encoder + projection head
        -> supervised contrastive loss
        -> optional auxiliary CE on pooled encoder representation

Recommended usage (two-stage schedule in one run):
    python scripts/train_stage2_supervised_fresh.py \
        --pretrained_path pretrained/macho_v2_2025 \
        --data_dir data/records/alcock/fold_0/train \
        --epochs 6 --warmup_epochs 1 \
        --supcon_weight 1.0 --ce_weight 0.1
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import warnings

import tensorflow as tf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.augmentation import (  # noqa: E402
    create_contrastive_views,
    get_default_augmentation_config,
    get_strong_augmentation_config,
)
from src.losses.supcon import supervised_contrastive_loss  # noqa: E402
from src.models.contrastive_astromer import (  # noqa: E402
    build_contrastive_model,
    build_contrastive_model_from_pretrained,
)
from src.data.split_utils import collect_record_files, resolve_train_val_root  # noqa: E402
from src.data.contrastive_record_utils import detect_record_schema, parse_contrastive_sample  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message=r"Layer '.*' \(of type .*\) was passed an input with a mask attached to it.*",
    category=UserWarning,
    module=r"keras\.src\.layers\.layer",
)


RECORD_SCHEMA = "dim"


def parse_labeled_tfrecord(example_proto):
    """Parse one labeled SequenceExample into encoder-ready tensors."""
    return parse_contrastive_sample(example_proto, schema=RECORD_SCHEMA)


def truncate_to_window(sample, window_size):
    """Random crop to window_size while preserving label."""
    seq_len = tf.shape(sample["input"])[0]
    max_start = tf.maximum(seq_len - window_size, 0)
    start = tf.random.uniform((), minval=0, maxval=max_start + 1, dtype=tf.int32)
    end = tf.minimum(start + window_size, seq_len)

    return {
        "input": sample["input"][start:end],
        "times": sample["times"][start:end],
        "mask_in": sample["mask_in"][start:end],
        "label": sample["label"],
    }


def _weak_augmentation_config():
    cfg = get_default_augmentation_config()
    cfg.update(
        {
            "time_jitter_prob": 0.2,
            "time_jitter_max": 0.03,
            "amplitude_scale_prob": 0.25,
            "amplitude_scale_range": (0.98, 1.02),
            "amplitude_shift_prob": 0.1,
            "amplitude_shift_range": (-0.03, 0.03),
            "noise_prob": 0.35,
            "noise_level": 0.005,
            "masking_prob": 0.1,
            "drop_prob": 0.05,
            "crop_prob": 0.05,
            "crop_fraction": 0.9,
        }
    )
    return cfg


def resolve_augmentation_strength(strength):
    """Return (enabled, config) from a named strength."""
    key = str(strength).lower()
    if key == "none":
        return False, None
    if key == "weak":
        return True, _weak_augmentation_config()
    if key == "default":
        return True, None
    if key == "strong":
        return True, get_strong_augmentation_config()
    raise ValueError(f"Unknown augmentation strength: {strength}")


def make_view_pair(sample, apply_aug=True, aug_cfg=None):
    """Create two views plus label for supervised contrastive learning."""
    core = {
        "input": sample["input"],
        "times": sample["times"],
        "mask_in": sample["mask_in"],
    }

    if apply_aug:
        view_a, view_b = create_contrastive_views(core, config=aug_cfg)
    else:
        view_a = {
            "input": tf.identity(core["input"]),
            "times": tf.identity(core["times"]),
            "mask_in": tf.identity(core["mask_in"]),
        }
        view_b = {
            "input": tf.identity(core["input"]),
            "times": tf.identity(core["times"]),
            "mask_in": tf.identity(core["mask_in"]),
        }

    return view_a, view_b, sample["label"]


def build_dataset_pipeline(
    raw_dataset,
    window_size,
    batch_size,
    shuffle_buffer=None,
    apply_aug=True,
    aug_cfg=None,
):
    dataset = raw_dataset.map(parse_labeled_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda s: truncate_to_window(s, window_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda s: make_view_pair(s, apply_aug=apply_aug, aug_cfg=aug_cfg),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    padding_shapes = (
        {"input": [window_size, 1], "times": [window_size, 1], "mask_in": [window_size, 1]},
        {"input": [window_size, 1], "times": [window_size, 1], "mask_in": [window_size, 1]},
        [],
    )
    padding_values = (
        {"input": 0.0, "times": 0.0, "mask_in": 0.0},
        {"input": 0.0, "times": 0.0, "mask_in": 0.0},
        tf.constant(0, dtype=tf.int32),
    )

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padding_shapes,
        padding_values=padding_values,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_labeled_train_val_datasets(
    record_dir,
    batch_size,
    shuffle_buffer,
    window_size,
    allow_single_class,
    train_aug_strength,
    val_aug_strength,
):
    """Load explicit train/val split directories and build batched pipelines."""
    resolved_dir, train_dir, val_dir = resolve_train_val_root(record_dir, project_root)
    global RECORD_SCHEMA
    RECORD_SCHEMA = detect_record_schema(resolved_dir)
    train_files = [str(path) for path in collect_record_files(train_dir)]
    val_files = [str(path) for path in collect_record_files(val_dir)]

    print(f"[DATA] Train split: {train_dir} ({len(train_files)} files)")
    print(f"[DATA] Val split:   {val_dir} ({len(val_files)} files)")

    if not train_files:
        raise ValueError(
            f"No .record files found in train split under {record_dir}. "
            f"Resolved root: {resolved_dir}"
        )

    train_dataset_raw = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    val_dataset_raw = tf.data.TFRecordDataset(val_files, num_parallel_reads=tf.data.AUTOTUNE)

    def _extract_label(serialized):
        if RECORD_SCHEMA == "legacy":
            context, _ = tf.io.parse_single_sequence_example(
                serialized=serialized,
                context_features={"Label": tf.io.FixedLenFeature([], tf.int64)},
                sequence_features={},
            )
            return tf.cast(context["Label"], tf.int32)

        context, _ = tf.io.parse_single_sequence_example(
            serialized=serialized,
            context_features={"label": tf.io.FixedLenFeature([], tf.int64)},
            sequence_features={},
        )
        return tf.cast(context["label"], tf.int32)

    label_dataset = train_dataset_raw.map(_extract_label, num_parallel_calls=tf.data.AUTOTUNE)

    label_values = [int(y.numpy()) for y in label_dataset]
    label_counts = dict(sorted(Counter(label_values).items()))
    unique_labels = list(label_counts.keys())

    print(f"[DATA] Unique labels: {unique_labels}")
    print(f"[DATA] Label counts: {label_counts}")

    if len(unique_labels) < 2:
        message = (
            "Stage 2 supervised contrastive requires >=2 classes, "
            f"but found {len(unique_labels)} label(s): {unique_labels}"
        )
        if allow_single_class:
            print(f"[WARN] {message}. Continuing because allow_single_class=True")
        else:
            raise ValueError(message)

    train_apply_aug, train_aug_cfg = resolve_augmentation_strength(train_aug_strength)
    val_apply_aug, val_aug_cfg = resolve_augmentation_strength(val_aug_strength)
    print(
        f"[AUG] train={train_aug_strength} (enabled={train_apply_aug}), "
        f"val={val_aug_strength} (enabled={val_apply_aug})"
    )

    train_ds = build_dataset_pipeline(
        train_dataset_raw,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        apply_aug=train_apply_aug,
        aug_cfg=train_aug_cfg,
    )
    val_ds = build_dataset_pipeline(
        val_dataset_raw,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=None,
        apply_aug=val_apply_aug,
        aug_cfg=val_aug_cfg,
    )

    total_count = float(len(label_values))
    num_classes = max(len(unique_labels), 1)
    class_weights = {
        int(lbl): float(total_count / (num_classes * count))
        for lbl, count in label_counts.items()
    }

    info = {
        "unique_labels": unique_labels,
        "label_counts": label_counts,
        "class_weights": class_weights,
        "num_examples": len(label_values),
        "train_files": train_files,
        "val_files": val_files,
    }
    return train_ds, val_ds, info


class Stage2FreshTrainer:
    def __init__(
        self,
        model,
        optimizer,
        tau,
        supcon_weight,
        ce_weight,
        warmup_epochs,
        warmup_supcon_weight,
        warmup_ce_weight,
        checkpoint_dir,
        num_classes,
        class_weights,
        use_class_balanced_ce,
        max_train_batches,
        max_val_batches,
    ):
        self.model = model
        self.optimizer = optimizer
        self.tau = tau

        self.supcon_weight = supcon_weight
        self.ce_weight = ce_weight
        self.warmup_epochs = warmup_epochs
        self.warmup_supcon_weight = warmup_supcon_weight
        self.warmup_ce_weight = warmup_ce_weight

        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = None
        self.class_weight_table = None

        if max(self.ce_weight, self.warmup_ce_weight) > 0.0:
            if num_classes is None:
                raise ValueError("num_classes must be provided when CE is enabled")
            self.classifier = tf.keras.layers.Dense(num_classes, name="aux_classifier")

            if use_class_balanced_ce and class_weights is not None:
                keys = tf.constant(sorted(class_weights.keys()), dtype=tf.int32)
                vals = tf.constant(
                    [class_weights[int(k)] for k in sorted(class_weights.keys())],
                    dtype=tf.float32,
                )
                self.class_weight_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys, vals),
                    default_value=tf.constant(1.0, dtype=tf.float32),
                )

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_supcon = tf.keras.metrics.Mean(name="train_supcon")
        self.train_ce = tf.keras.metrics.Mean(name="train_ce")
        self.train_ce_acc = tf.keras.metrics.Mean(name="train_ce_acc")
        self.train_retr = tf.keras.metrics.Mean(name="train_retr")

        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_supcon = tf.keras.metrics.Mean(name="val_supcon")
        self.val_ce = tf.keras.metrics.Mean(name="val_ce")
        self.val_ce_acc = tf.keras.metrics.Mean(name="val_ce_acc")
        self.val_retr = tf.keras.metrics.Mean(name="val_retr")

        print(
            f"[TRAINER] tau={self.tau}, supcon_weight={self.supcon_weight}, ce_weight={self.ce_weight}, "
            f"warmup_epochs={self.warmup_epochs}"
        )
        if self.class_weight_table is not None:
            print("[TRAINER] Class-balanced CE: enabled")
        print(f"[TRAINER] checkpoints: {self.checkpoint_dir}")

    @staticmethod
    def _masked_mean_pool(h, mask):
        mask = tf.cast(mask, h.dtype)
        h_sum = tf.reduce_sum(h * mask, axis=1)
        denom = tf.reduce_sum(mask, axis=1)
        return h_sum / tf.maximum(denom, tf.constant(1e-8, dtype=h.dtype))

    @staticmethod
    def _retrieval_accuracy(z_a, z_b, labels):
        z = tf.concat([z_a, z_b], axis=0)
        y = tf.concat([labels, labels], axis=0)

        sims = tf.matmul(z, z, transpose_b=True)
        n = tf.shape(sims)[0]
        sims = sims - tf.eye(n, dtype=sims.dtype) * 1e9
        nn_idx = tf.argmax(sims, axis=1, output_type=tf.int32)
        nn_label = tf.gather(y, nn_idx)
        return tf.reduce_mean(tf.cast(tf.equal(nn_label, y), tf.float32))

    @tf.function
    def _train_step(self, view_a, view_b, labels, current_supcon_weight, current_ce_weight):
        with tf.GradientTape() as tape:
            z_a, h_a = self.model(view_a, training=True, return_representation=True)
            z_b, h_b = self.model(view_b, training=True, return_representation=True)

            supcon = supervised_contrastive_loss(z_a, z_b, labels, tau=self.tau)
            ce = tf.constant(0.0, dtype=tf.float32)
            ce_acc = tf.constant(0.0, dtype=tf.float32)

            if self.classifier is not None and current_ce_weight > 0.0:
                h_pool_a = self._masked_mean_pool(h_a, view_a["mask_in"])
                h_pool_b = self._masked_mean_pool(h_b, view_b["mask_in"])

                logits_a = self.classifier(h_pool_a, training=True)
                logits_b = self.classifier(h_pool_b, training=True)
                logits = tf.concat([logits_a, logits_b], axis=0)
                y2 = tf.concat([labels, labels], axis=0)

                ce_per = tf.keras.losses.sparse_categorical_crossentropy(
                    y2, logits, from_logits=True
                )
                if self.class_weight_table is not None:
                    sample_w = self.class_weight_table.lookup(tf.cast(y2, tf.int32))
                    ce = tf.math.divide_no_nan(
                        tf.reduce_sum(ce_per * sample_w),
                        tf.reduce_sum(sample_w),
                    )
                else:
                    ce = tf.reduce_mean(ce_per)

                pred = tf.argmax(logits, axis=1, output_type=tf.int32)
                ce_acc = tf.reduce_mean(
                    tf.cast(tf.equal(pred, tf.cast(y2, tf.int32)), tf.float32)
                )

            total = current_supcon_weight * supcon + current_ce_weight * ce

        variables = list(self.model.trainable_variables)
        if self.classifier is not None:
            variables = variables + list(self.classifier.trainable_variables)

        grads = tape.gradient(total, variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)

        retr = self._retrieval_accuracy(z_a, z_b, labels)
        return total, supcon, ce, ce_acc, retr

    @tf.function
    def _val_step(self, view_a, view_b, labels, current_supcon_weight, current_ce_weight):
        z_a, h_a = self.model(view_a, training=False, return_representation=True)
        z_b, h_b = self.model(view_b, training=False, return_representation=True)

        supcon = supervised_contrastive_loss(z_a, z_b, labels, tau=self.tau)
        ce = tf.constant(0.0, dtype=tf.float32)
        ce_acc = tf.constant(0.0, dtype=tf.float32)

        if self.classifier is not None and current_ce_weight > 0.0:
            h_pool_a = self._masked_mean_pool(h_a, view_a["mask_in"])
            h_pool_b = self._masked_mean_pool(h_b, view_b["mask_in"])
            logits_a = self.classifier(h_pool_a, training=False)
            logits_b = self.classifier(h_pool_b, training=False)
            logits = tf.concat([logits_a, logits_b], axis=0)
            y2 = tf.concat([labels, labels], axis=0)

            ce_per = tf.keras.losses.sparse_categorical_crossentropy(
                y2, logits, from_logits=True
            )
            if self.class_weight_table is not None:
                sample_w = self.class_weight_table.lookup(tf.cast(y2, tf.int32))
                ce = tf.math.divide_no_nan(
                    tf.reduce_sum(ce_per * sample_w),
                    tf.reduce_sum(sample_w),
                )
            else:
                ce = tf.reduce_mean(ce_per)

            pred = tf.argmax(logits, axis=1, output_type=tf.int32)
            ce_acc = tf.reduce_mean(
                tf.cast(tf.equal(pred, tf.cast(y2, tf.int32)), tf.float32)
            )

        total = current_supcon_weight * supcon + current_ce_weight * ce
        retr = self._retrieval_accuracy(z_a, z_b, labels)
        return total, supcon, ce, ce_acc, retr

    def _reset_metrics(self, train=True):
        if train:
            self.train_loss.reset_state()
            self.train_supcon.reset_state()
            self.train_ce.reset_state()
            self.train_ce_acc.reset_state()
            self.train_retr.reset_state()
        else:
            self.val_loss.reset_state()
            self.val_supcon.reset_state()
            self.val_ce.reset_state()
            self.val_ce_acc.reset_state()
            self.val_retr.reset_state()

    def _current_weights(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.warmup_supcon_weight, self.warmup_ce_weight
        return self.supcon_weight, self.ce_weight

    def train_epoch(self, dataset, epoch):
        self._reset_metrics(train=True)
        current_supcon_weight, current_ce_weight = self._current_weights(epoch)

        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}")
        print(
            f"[EPOCH_CFG] supcon_weight={current_supcon_weight}, ce_weight={current_ce_weight}"
        )
        print("=" * 70)

        start_time = time.time()
        num_batches = 0

        for batch_idx, (view_a, view_b, labels) in enumerate(dataset):
            if self.max_train_batches is not None and num_batches >= self.max_train_batches:
                break

            total, supcon, ce, ce_acc, retr = self._train_step(
                view_a,
                view_b,
                labels,
                tf.constant(current_supcon_weight, dtype=tf.float32),
                tf.constant(current_ce_weight, dtype=tf.float32),
            )

            self.train_loss.update_state(total)
            self.train_supcon.update_state(supcon)
            self.train_ce.update_state(ce)
            self.train_ce_acc.update_state(ce_acc)
            self.train_retr.update_state(retr)
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx + 1:4d} | "
                    f"Loss: {self.train_loss.result().numpy():.4f} | "
                    f"SupCon: {self.train_supcon.result().numpy():.4f} | "
                    f"CE: {self.train_ce.result().numpy():.4f} | "
                    f"CE-Acc: {self.train_ce_acc.result().numpy():.4f} | "
                    f"Retr@1: {self.train_retr.result().numpy():.4f}"
                )

        elapsed = time.time() - start_time
        print(f"[TRAIN] Batches: {num_batches} | Time: {elapsed:.2f}s")
        print(
            f"[TRAIN] Loss: {self.train_loss.result().numpy():.4f} | "
            f"SupCon: {self.train_supcon.result().numpy():.4f} | "
            f"CE: {self.train_ce.result().numpy():.4f} | "
            f"CE-Acc: {self.train_ce_acc.result().numpy():.4f} | "
            f"Retr@1: {self.train_retr.result().numpy():.4f}"
        )

    def validate_epoch(self, dataset, epoch):
        self._reset_metrics(train=False)
        current_supcon_weight, current_ce_weight = self._current_weights(epoch)

        num_batches = 0
        for view_a, view_b, labels in dataset:
            if self.max_val_batches is not None and num_batches >= self.max_val_batches:
                break

            total, supcon, ce, ce_acc, retr = self._val_step(
                view_a,
                view_b,
                labels,
                tf.constant(current_supcon_weight, dtype=tf.float32),
                tf.constant(current_ce_weight, dtype=tf.float32),
            )

            self.val_loss.update_state(total)
            self.val_supcon.update_state(supcon)
            self.val_ce.update_state(ce)
            self.val_ce_acc.update_state(ce_acc)
            self.val_retr.update_state(retr)
            num_batches += 1

        print(
            f"[VAL]   Loss: {self.val_loss.result().numpy():.4f} | "
            f"SupCon: {self.val_supcon.result().numpy():.4f} | "
            f"CE: {self.val_ce.result().numpy():.4f} | "
            f"CE-Acc: {self.val_ce_acc.result().numpy():.4f} | "
            f"Retr@1: {self.val_retr.result().numpy():.4f}"
        )

    def save_checkpoint(self, epoch):
        full_path = self.checkpoint_dir / f"stage2_fresh_epoch_{epoch}.weights.h5"
        self.model.save_weights(str(full_path))
        print(f"[CHECKPOINT] Saved full model to {full_path}")

        enc_path = self.checkpoint_dir / f"encoder_stage2_fresh_epoch_{epoch}.weights.h5"
        self.model.encoder.save_weights(str(enc_path))
        print(f"[CHECKPOINT] Saved encoder to {enc_path}")

        if self.classifier is not None and self.classifier.built:
            clf_path = self.checkpoint_dir / f"classifier_stage2_fresh_epoch_{epoch}"
            ckpt = tf.train.Checkpoint(classifier=self.classifier)
            ckpt.write(str(clf_path))
            print(f"[CHECKPOINT] Saved classifier checkpoint to {clf_path}")

    def train(self, train_dataset, val_dataset, epochs, save_every):
        history = {
            "train_loss": [],
            "train_supcon": [],
            "train_ce": [],
            "train_ce_acc": [],
            "train_retr": [],
            "val_loss": [],
            "val_supcon": [],
            "val_ce": [],
            "val_ce_acc": [],
            "val_retr": [],
            "epoch_supcon_weight": [],
            "epoch_ce_weight": [],
        }

        print("\n" + "=" * 70)
        print("STARTING FRESH STAGE 2 SUPERVISED CONTRASTIVE TRAINING")
        print("=" * 70)

        for epoch in range(1, epochs + 1):
            current_supcon_weight, current_ce_weight = self._current_weights(epoch)

            self.train_epoch(train_dataset, epoch)
            self.validate_epoch(val_dataset, epoch)

            history["train_loss"].append(float(self.train_loss.result().numpy()))
            history["train_supcon"].append(float(self.train_supcon.result().numpy()))
            history["train_ce"].append(float(self.train_ce.result().numpy()))
            history["train_ce_acc"].append(float(self.train_ce_acc.result().numpy()))
            history["train_retr"].append(float(self.train_retr.result().numpy()))

            history["val_loss"].append(float(self.val_loss.result().numpy()))
            history["val_supcon"].append(float(self.val_supcon.result().numpy()))
            history["val_ce"].append(float(self.val_ce.result().numpy()))
            history["val_ce_acc"].append(float(self.val_ce_acc.result().numpy()))
            history["val_retr"].append(float(self.val_retr.result().numpy()))

            history["epoch_supcon_weight"].append(float(current_supcon_weight))
            history["epoch_ce_weight"].append(float(current_ce_weight))

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

        history_path = self.checkpoint_dir / "history_stage2_fresh.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"[HISTORY] Saved to {history_path}")

        print("\n" + "=" * 70)
        print("FRESH STAGE 2 TRAINING COMPLETE")
        print("=" * 70)


def build_model_from_args(args):
    if args.pretrained_path:
        print(f"[1/5] Building model from pretrained config: {args.pretrained_path}")
        model, pt_config = build_contrastive_model_from_pretrained(
            pretrained_path=args.pretrained_path,
            projection_dim=args.projection_dim,
            projection_hidden_dim=args.projection_hidden_dim,
            encoder_mask_mode=args.encoder_mask_mode,
        )
        args.window_size = pt_config["window_size"]
        print(f"[OK] Model built with pretrained architecture (window={args.window_size})")
        return model

    print("[1/5] Building model from args")
    model = build_contrastive_model(
        window_size=args.window_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        mixer_size=args.mixer_size,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        encoder_mask_mode=args.encoder_mask_mode,
    )
    dummy = {
        "input": tf.zeros([2, args.window_size, 1]),
        "times": tf.zeros([2, args.window_size, 1]),
        "mask_in": tf.ones([2, args.window_size, 1]),
    }
    _ = model(dummy, training=False)
    print("[OK] Model built")
    return model


def maybe_load_stage1_weights(model, args):
    print("[2/5] Loading Stage 1 initialization (if provided)")
    if args.stage1_checkpoint:
        model.load_weights(args.stage1_checkpoint)
        print(f"[OK] Loaded full checkpoint: {args.stage1_checkpoint}")
        return

    if args.stage1_encoder_checkpoint:
        model.encoder.load_weights(args.stage1_encoder_checkpoint)
        print(f"[OK] Loaded encoder checkpoint: {args.stage1_encoder_checkpoint}")
        return

    print("- No Stage 1 weights provided. Training starts from current initialization.")


def main():
    parser = argparse.ArgumentParser(description="Fresh Stage 2 Supervised Contrastive Training")

    parser.add_argument("--data_dir", type=str, default="data/records/macho_subset/fold_0/train")

    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--mixer_size", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--encoder_mask_mode",
        choices=["current", "invert_visible"],
        default="current",
        help=(
            "current keeps the historical contrastive behavior; invert_visible "
            "treats mask_in as a visible mask for pooling and sends 1-mask_in "
            "to the OG encoder."
        ),
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--supcon_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_supcon_weight", type=float, default=0.0)
    parser.add_argument("--warmup_ce_weight", type=float, default=1.0)

    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--stage1_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage2_fresh_reimpl")
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--shuffle_buffer", type=int, default=1000)
    parser.add_argument("--allow_single_class", action="store_true")
    parser.add_argument(
        "--train_aug_strength",
        type=str,
        choices=["none", "weak", "default", "strong"],
        default="weak",
    )
    parser.add_argument(
        "--val_aug_strength",
        type=str,
        choices=["none", "weak", "default", "strong"],
        default="none",
    )
    parser.add_argument("--use_class_balanced_ce", action="store_true")

    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CONFIGURATION (FRESH STAGE 2)")
    print("=" * 70)
    for key, value in vars(args).items():
        print(f"{key:28s}: {value}")
    print("=" * 70)

    model = build_model_from_args(args)
    maybe_load_stage1_weights(model, args)

    if args.freeze_encoder:
        model.encoder.trainable = False
        print("- Encoder frozen")

    print("[3/5] Creating optimizer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    print(f"[OK] Adam(lr={args.learning_rate})")

    print("[4/5] Loading labeled train/val datasets")
    train_dataset, val_dataset, data_info = load_labeled_train_val_datasets(
        record_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        window_size=args.window_size,
        allow_single_class=args.allow_single_class,
        train_aug_strength=args.train_aug_strength,
        val_aug_strength=args.val_aug_strength,
    )

    if args.num_classes is None:
        args.num_classes = len(data_info["unique_labels"])
        print(f"[AUTO] Inferred num_classes={args.num_classes}")

    print("[5/5] Starting training")
    trainer = Stage2FreshTrainer(
        model=model,
        optimizer=optimizer,
        tau=args.tau,
        supcon_weight=args.supcon_weight,
        ce_weight=args.ce_weight,
        warmup_epochs=args.warmup_epochs,
        warmup_supcon_weight=args.warmup_supcon_weight,
        warmup_ce_weight=args.warmup_ce_weight,
        checkpoint_dir=args.checkpoint_dir,
        num_classes=args.num_classes,
        class_weights=data_info["class_weights"],
        use_class_balanced_ce=args.use_class_balanced_ce,
        max_train_batches=(args.max_train_batches if args.max_train_batches > 0 else None),
        max_val_batches=(args.max_val_batches if args.max_val_batches > 0 else None),
    )
    trainer.train(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
