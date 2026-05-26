"""Train Alcock classifiers on contrastive-compatible pooled h representations.

This script is intentionally parallel to, not a replacement for, the official
`presentation.pipelines.pipeline_0.classify` pipeline. The official pipeline
keeps its `skip_avg_mlp` token/layer readout. This script uses:

    encoder token outputs -> masked mean pooled h -> AstromerStyleMLPHead -> logits

The first version freezes the encoder and trains only the MLP head.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"Layer '.*' \(of type .*\) was passed an input with a mask attached to it.*",
    category=UserWarning,
    module=r"keras\.src\.layers\.layer",
)

from src.data.contrastive_record_utils import (  # noqa: E402
    detect_record_schema,
    parse_contrastive_sample,
)
from src.data.split_utils import collect_record_files, split_dir  # noqa: E402
from src.models.contrastive_astromer import (  # noqa: E402
    build_contrastive_model_from_pretrained,
)


class AstromerStyleMLPHead(tf.keras.Model):
    """Official Astromer classifier MLP capacity applied to pooled h."""

    def __init__(self, num_classes: int, name: str = "astromer_style_mlp_head"):
        super().__init__(name=name)
        self.num_classes = int(num_classes)
        self.dense_1024 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_512 = tf.keras.layers.Dense(512, activation="relu")
        self.dense_256 = tf.keras.layers.Dense(256, activation="relu")
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm")
        self.output_layer = tf.keras.layers.Dense(self.num_classes, name="output_layer")

    def call(self, h: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        x = self.dense_1024(h)
        x = self.dense_512(x)
        x = self.dense_256(x)
        x = self.layer_norm(x)
        return self.output_layer(x)

    def get_config(self) -> dict[str, object]:
        return {"num_classes": self.num_classes, "name": self.name}


class PooledHClassifier(tf.keras.Model):
    """Frozen encoder plus masked mean pooled h plus MLP logits head."""

    def __init__(
        self,
        contrastive_model: tf.keras.Model,
        head: AstromerStyleMLPHead,
        encoder_frozen: bool = True,
        expected_h_dim: int = 256,
        encoder_mask_mode: str = "current",
        name: str = "pooled_h_classifier",
    ):
        super().__init__(name=name)
        if encoder_mask_mode not in {"current", "invert_visible"}:
            raise ValueError(
                "encoder_mask_mode must be 'current' or 'invert_visible', "
                f"got {encoder_mask_mode!r}"
            )
        self.contrastive_model = contrastive_model
        self.encoder = contrastive_model.encoder
        self.head = head
        self.encoder_frozen = bool(encoder_frozen)
        self.expected_h_dim = int(expected_h_dim)
        self.encoder_mask_mode = encoder_mask_mode
        self.contrastive_model.projection_head.trainable = False
        self.encoder.trainable = not self.encoder_frozen

    @staticmethod
    def masked_mean_pool(h_seq: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(mask, h_seq.dtype)
        h_sum = tf.reduce_sum(h_seq * mask, axis=1)
        denom = tf.reduce_sum(mask, axis=1)
        return h_sum / tf.maximum(denom, tf.constant(1e-8, dtype=h_seq.dtype))

    def extract_h(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        encoder_training = bool(training and not self.encoder_frozen)
        visible_mask = tf.cast(inputs["mask_in"], tf.float32)
        if self.encoder_mask_mode == "invert_visible":
            encoder_inputs = dict(inputs)
            encoder_inputs["mask_in"] = 1.0 - visible_mask
        else:
            encoder_inputs = inputs
        h_seq = self.encoder(encoder_inputs, training=encoder_training)
        h = self.masked_mean_pool(h_seq, visible_mask)

        if h.shape.rank != 2:
            raise ValueError(f"pooled h must be rank 2 [B,D], got shape={h.shape}")
        if h.shape[-1] is not None and int(h.shape[-1]) != self.expected_h_dim:
            raise ValueError(
                f"pooled h dim mismatch: expected {self.expected_h_dim}, got {h.shape[-1]}"
            )
        return h

    def call(
        self,
        inputs: dict[str, tf.Tensor],
        training: bool = False,
        return_h: bool = False,
    ):
        h = self.extract_h(inputs, training=training)
        logits = self.head(h, training=training)

        if logits.shape.rank != 2:
            raise ValueError(f"logits must be rank 2 [B,C], got shape={logits.shape}")
        if logits.shape[-1] is not None and int(logits.shape[-1]) != self.head.num_classes:
            raise ValueError(
                f"logits dim mismatch: expected {self.head.num_classes}, got {logits.shape[-1]}"
            )

        if return_h:
            return logits, h
        return logits


@dataclass(frozen=True)
class SystemSpec:
    key: str
    display_name: str
    encoder: str
    representation: str
    checkpoint: str | None
    checkpoint_kind: str
    notes: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_class_names(data_root: Path) -> list[str]:
    objects_path = data_root / "objects.csv"
    if not objects_path.is_file():
        raise FileNotFoundError(f"Expected class mapping file: {objects_path}")

    with objects_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows or "label" not in rows[0]:
        raise ValueError(f"{objects_path} must contain a 'label' column")
    return [row["label"] for row in rows]


def center_crop_to_window(sample: dict[str, tf.Tensor], window_size: int) -> dict[str, tf.Tensor]:
    seq_len = tf.shape(sample["input"])[0]

    def crop() -> dict[str, tf.Tensor]:
        start = (seq_len - window_size) // 2
        end = start + window_size
        return {
            "input": sample["input"][start:end],
            "times": sample["times"][start:end],
            "mask_in": sample["mask_in"][start:end],
            "label": sample["label"],
        }

    return tf.cond(seq_len > window_size, crop, lambda: sample)


def build_labeled_dataset(
    record_files: Iterable[Path],
    schema: str,
    window_size: int,
    batch_size: int,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int,
) -> tf.data.Dataset:
    files = [str(path) for path in record_files]
    if not files:
        raise ValueError("No .record files found for split")

    def parse_one(serialized):
        return parse_contrastive_sample(serialized, schema=schema)

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_one, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda s: tf.shape(s["input"])[0] > 0)
    dataset = dataset.map(
        lambda s: center_crop_to_window(s, window_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        dataset = dataset.shuffle(
            shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=True,
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

    def split_xy(batch):
        y = batch.pop("label")
        return batch, y

    return dataset.map(split_xy, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def _assign_pair(layer: tf.keras.layers.Layer, group, path: str) -> int:
    if path not in group:
        raise KeyError(f"Missing checkpoint group: {path}")
    weights_group = group[path]["vars"]
    weights = [np.array(weights_group[str(i)]) for i in range(len(weights_group))]
    layer.set_weights(weights)
    return len(weights)


def load_encoder_weights_h5_by_structure(encoder: tf.keras.Model, checkpoint: Path) -> int:
    """Load encoder-only Keras H5 weights by semantic layer structure.

    Keras 3 may fail to load these checkpoints directly when auto-generated
    nested layer names differ. The saved H5 files are nevertheless regular and
    stable, so this loader maps by encoder component order.
    """

    assigned = 0
    with h5py.File(checkpoint, "r") as f:
        root = f["encoder"] if "encoder" in f else f
        assigned += _assign_pair(encoder.inp_transform, root, "inp_transform")

        enc_layers_group = root["enc_layers"]
        for idx, block in enumerate(encoder.enc_layers):
            block_name = "attention_block" if idx == 0 else f"attention_block_{idx}"
            block_group = enc_layers_group[block_name]

            assigned += _assign_pair(block.mha.wq, block_group, "mha/wq")
            assigned += _assign_pair(block.mha.wk, block_group, "mha/wk")
            assigned += _assign_pair(block.mha.wv, block_group, "mha/wv")
            assigned += _assign_pair(block.mha.dense, block_group, "mha/dense")
            assigned += _assign_pair(
                block.ffn.layers[0],
                block_group,
                "ffn/_layer_checkpoint_dependencies/dense",
            )
            assigned += _assign_pair(
                block.ffn.layers[1],
                block_group,
                "ffn/_layer_checkpoint_dependencies/dense_1",
            )
            assigned += _assign_pair(block.layernorm1, block_group, "layernorm1")
            assigned += _assign_pair(block.layernorm2, block_group, "layernorm2")

            if hasattr(block, "reshape_leak_1"):
                assigned += _assign_pair(block.reshape_leak_1, block_group, "reshape_leak_1")
            if hasattr(block, "reshape_leak_2"):
                assigned += _assign_pair(block.reshape_leak_2, block_group, "reshape_leak_2")

    return assigned


def resolve_split_paths(data_root: Path) -> dict[str, Path]:
    splits = {
        "train": split_dir(data_root, "train"),
        "val": split_dir(data_root, "val"),
        "test": split_dir(data_root, "test"),
    }
    missing = [name for name, path in splits.items() if path is None]
    if missing:
        raise ValueError(f"Missing split directories under {data_root}: {missing}")
    return {name: Path(path) for name, path in splits.items() if path is not None}


def dataset_label_distribution(dataset: tf.data.Dataset, num_classes: int) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for _, y in dataset:
        counts.update(int(v) for v in y.numpy().tolist())
    return {str(i): int(counts.get(i, 0)) for i in range(num_classes)}


def count_labels_from_records(
    record_files: Iterable[Path],
    schema: str,
    num_classes: int,
) -> dict[str, int]:
    files = [str(path) for path in record_files]
    counts: Counter[int] = Counter()

    if schema == "legacy":
        context_features = {"Label": tf.io.FixedLenFeature([], tf.int64, default_value=0)}
        label_key = "Label"
    else:
        context_features = {"label": tf.io.FixedLenFeature([], tf.int64, default_value=0)}
        label_key = "label"

    for serialized in tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE):
        context, _ = tf.io.parse_single_sequence_example(
            serialized=serialized,
            context_features=context_features,
            sequence_features={},
        )
        counts.update([int(context[label_key].numpy())])

    return {str(i): int(counts.get(i, 0)) for i in range(num_classes)}


def maybe_take(dataset: tf.data.Dataset, max_batches: int | None) -> tf.data.Dataset:
    if max_batches is None or max_batches <= 0:
        return dataset
    return dataset.take(max_batches)


def monitor_improved(current: float, best: float, monitor: str, min_delta: float) -> bool:
    if monitor == "val_macro_f1":
        return current > best + min_delta
    return current < best - min_delta


def build_model_for_system(
    spec: SystemSpec,
    pretrained_path: Path,
    projection_dim: int,
    projection_hidden_dim: int,
    num_classes: int,
    encoder_frozen: bool,
    expected_h_dim: int,
    encoder_mask_mode: str,
) -> PooledHClassifier:
    contrastive_model, _ = build_contrastive_model_from_pretrained(
        pretrained_path=str(pretrained_path),
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        encoder_mask_mode=encoder_mask_mode,
    )

    if spec.checkpoint:
        checkpoint = Path(spec.checkpoint)
        if spec.checkpoint_kind == "encoder":
            try:
                contrastive_model.encoder.load_weights(str(checkpoint))
                print(f"[LOAD] Loaded encoder checkpoint via Keras: {checkpoint}")
            except ValueError as exc:
                print(f"[LOAD] Keras encoder load failed; falling back to structural H5 load: {exc}")
                assigned = load_encoder_weights_h5_by_structure(
                    contrastive_model.encoder,
                    checkpoint,
                )
                print(f"[LOAD] Structurally assigned {assigned} encoder arrays from {checkpoint}")
        elif spec.checkpoint_kind == "full":
            contrastive_model.load_weights(str(checkpoint))
        else:
            raise ValueError(
                f"Unknown checkpoint_kind={spec.checkpoint_kind!r}; use 'encoder' or 'full'"
            )

    head = AstromerStyleMLPHead(num_classes=num_classes)
    return PooledHClassifier(
        contrastive_model=contrastive_model,
        head=head,
        encoder_frozen=encoder_frozen,
        expected_h_dim=expected_h_dim,
        encoder_mask_mode=encoder_mask_mode,
        name=spec.key,
    )


def evaluate_model(
    model: PooledHClassifier,
    dataset: tf.data.Dataset,
    num_classes: int,
    class_names: list[str],
    loss_fn: tf.keras.losses.Loss | None = None,
) -> dict[str, object]:
    y_true_batches = []
    y_pred_batches = []
    logits_batches = []
    h_shapes = []
    losses = []

    for x, y in dataset:
        logits, h = model(x, training=False, return_h=True)
        if loss_fn is not None:
            losses.append(float(loss_fn(y, logits).numpy()))

        y_true_batches.append(y.numpy().astype(np.int32))
        y_pred_batches.append(tf.argmax(logits, axis=1).numpy().astype(np.int32))
        logits_batches.append(logits.numpy().astype(np.float32))
        h_shapes.append([int(dim) for dim in h.shape])

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    logits_all = np.concatenate(logits_batches, axis=0)

    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    classwise = {}
    for idx, name in enumerate(class_names):
        classwise[name] = {
            "label_id": idx,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics = {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[1]
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "classwise": classwise,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "num_examples": int(y_true.shape[0]),
        "unique_true_labels": sorted(int(v) for v in np.unique(y_true)),
        "unique_pred_labels": sorted(int(v) for v in np.unique(y_pred)),
        "h_shapes_seen": h_shapes[:5],
        "logits_shape": [int(dim) for dim in logits_all.shape],
    }
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits_all,
    }


def train_one_system(
    spec: SystemSpec,
    datasets: dict[str, tf.data.Dataset],
    pretrained_path: Path,
    output_dir: Path,
    class_names: list[str],
    args: argparse.Namespace,
    distributions: dict[str, dict[str, int]],
) -> dict[str, object]:
    num_classes = len(class_names)
    system_dir = output_dir / spec.key
    system_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_for_system(
        spec=spec,
        pretrained_path=pretrained_path,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        num_classes=num_classes,
        encoder_frozen=args.freeze_encoder,
        expected_h_dim=args.expected_h_dim,
        encoder_mask_mode=args.encoder_mask_mode,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    best_monitor_value = -np.inf if args.monitor == "val_macro_f1" else np.inf
    best_val_macro_f1 = -np.inf
    best_epoch = None
    best_weights_path = system_dir / "best_model.weights.h5"
    history = []
    epochs_without_improvement = 0

    # Build head variables and run shape sanity checks before training.
    first_x, _ = next(iter(datasets["train"]))
    logits, h = model(first_x, training=False, return_h=True)
    if h.shape[-1] != args.expected_h_dim:
        raise ValueError(f"{spec.key}: expected h dim {args.expected_h_dim}, got {h.shape}")
    if logits.shape[-1] != num_classes:
        raise ValueError(f"{spec.key}: expected logits dim {num_classes}, got {logits.shape}")
    print(
        f"[{spec.key}] trainable_vars={len(model.trainable_variables)} "
        f"non_trainable_vars={len(model.non_trainable_variables)} "
        f"encoder_frozen={args.freeze_encoder}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_true = []
        train_pred = []

        for x, y in maybe_take(datasets["train"], args.max_train_batches):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits)

            variables = model.trainable_variables
            grads = tape.gradient(loss, variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
            optimizer.apply_gradients(grads_and_vars)

            train_losses.append(float(loss.numpy()))
            train_true.append(y.numpy().astype(np.int32))
            train_pred.append(tf.argmax(logits, axis=1).numpy().astype(np.int32))

        train_true_arr = np.concatenate(train_true, axis=0)
        train_pred_arr = np.concatenate(train_pred, axis=0)
        val_result = evaluate_model(
            model,
            maybe_take(datasets["val"], args.max_val_batches),
            num_classes,
            class_names,
            loss_fn,
        )
        val_metrics = val_result["metrics"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(accuracy_score(train_true_arr, train_pred_arr)),
            "train_macro_f1": float(
                f1_score(train_true_arr, train_pred_arr, average="macro", zero_division=0)
            ),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print(
            f"[{spec.key}] epoch={epoch:03d} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"train_macro_f1={epoch_record['train_macro_f1']:.4f} "
            f"val_loss={epoch_record['val_loss']:.4f} "
            f"val_macro_f1={epoch_record['val_macro_f1']:.4f}",
            flush=True,
        )

        monitor_value = float(epoch_record[args.monitor])
        improved = monitor_improved(
            current=monitor_value,
            best=best_monitor_value,
            monitor=args.monitor,
            min_delta=args.min_delta,
        )
        if improved:
            best_monitor_value = monitor_value
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            model.save_weights(str(best_weights_path))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience is not None and epochs_without_improvement >= args.patience:
            print(
                f"[{spec.key}] early stopping at epoch={epoch} "
                f"(best_epoch={best_epoch}, best_{args.monitor}={best_monitor_value:.6f})",
                flush=True,
            )
            break

    model.load_weights(str(best_weights_path))
    test_result = evaluate_model(
        model,
        maybe_take(datasets["test"], args.max_test_batches),
        num_classes,
        class_names,
        loss_fn,
    )

    payload = {
        "system": spec.__dict__,
        "encoder_frozen": bool(args.freeze_encoder),
        "selection_metric": args.monitor,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "best_val_macro_f1": best_val_macro_f1,
        "num_classes": num_classes,
        "class_label_mapping": {str(i): name for i, name in enumerate(class_names)},
        "class_distribution": distributions,
        "sanity_checks": {
            "pooled_h_shape_expected": [None, args.expected_h_dim],
            "mlp_input_shape_expected": [None, args.expected_h_dim],
            "mlp_output_shape_expected": [None, num_classes],
            "first_batch_h_shape": [int(dim) for dim in h.shape],
            "first_batch_logits_shape": [int(dim) for dim in logits.shape],
            "uses_projection_z": False,
            "uses_official_skip_avg_mlp": False,
            "uses_gamma_weight": False,
            "trainable_variable_count": len(model.trainable_variables),
            "non_trainable_variable_count": len(model.non_trainable_variables),
        },
        "history": history,
        "test": test_result["metrics"],
    }

    with (system_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        system_dir / "predictions.npz",
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        logits=test_result["logits"],
    )

    return payload


def default_systems(args: argparse.Namespace) -> list[SystemSpec]:
    return [
        SystemSpec(
            key="og_pooled_h_mlp",
            display_name="OG pooled h + MLP",
            encoder="Original Astromer encoder",
            representation="masked mean pooled h",
            checkpoint=None,
            checkpoint_kind="none",
            notes="Original pretrained encoder loaded from pretrained_path; no official GammaWeight readout.",
        ),
        SystemSpec(
            key="stage1_pooled_h_mlp",
            display_name="Stage1 pooled h + MLP",
            encoder="Stage1 contrastive encoder",
            representation="masked mean pooled h",
            checkpoint=args.stage1_encoder_checkpoint,
            checkpoint_kind="encoder",
            notes="Stage1 encoder checkpoint; projection z is not used.",
        ),
        SystemSpec(
            key="stage2_pooled_h_mlp",
            display_name="Stage2 pooled h + MLP",
            encoder="Stage2 contrastive encoder",
            representation="masked mean pooled h",
            checkpoint=args.stage2_encoder_checkpoint,
            checkpoint_kind="encoder",
            notes="Stage2 encoder checkpoint; main target system.",
        ),
    ]


def write_summary_table(
    output_dir: Path,
    results: list[dict[str, object]],
    official_macro_f1: float | None,
    official_accuracy: float | None,
) -> None:
    rows = []
    if official_macro_f1 is not None or official_accuracy is not None:
        rows.append(
            {
                "System": "OG official pipeline",
                "Encoder": "Original Astromer",
                "Representation": "official skip_avg_mlp token/layer readout",
                "Head": "official MLP after GammaWeight",
                "Encoder frozen?": "",
                "Test Macro F1": official_macro_f1,
                "Test Accuracy": official_accuracy,
                "Notes": "External anchor baseline; not run by this script.",
            }
        )

    for result in results:
        spec = result["system"]
        test = result["test"]
        rows.append(
            {
                "System": spec["display_name"],
                "Encoder": spec["encoder"],
                "Representation": spec["representation"],
                "Head": "AstromerStyleMLPHead",
                "Encoder frozen?": result["encoder_frozen"],
                "Test Macro F1": test["macro_f1"],
                "Test Accuracy": test["accuracy"],
                "Notes": spec["notes"],
            }
        )

    md_lines = [
        "| System | Encoder | Representation | Head | Encoder frozen? | Test Macro F1 | Test Accuracy | Notes |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {System} | {Encoder} | {Representation} | {Head} | {Frozen} | {MacroF1} | {Accuracy} | {Notes} |".format(
                System=row["System"],
                Encoder=row["Encoder"],
                Representation=row["Representation"],
                Head=row["Head"],
                Frozen=row["Encoder frozen?"],
                MacroF1="" if row["Test Macro F1"] is None else f"{row['Test Macro F1']:.6f}",
                Accuracy="" if row["Test Accuracy"] is None else f"{row['Test Accuracy']:.6f}",
                Notes=row["Notes"],
            )
        )

    (output_dir / "results_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with (output_dir / "results_table.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen encoder + pooled h + Astromer-style MLP classification on Alcock."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/records/alcock/alcock/fold_0/alcock",
        help="Alcock package root containing train/val/test and objects.csv.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="weights/macho_v2_2025",
        help="Original Astromer pretrained checkpoint/config directory.",
    )
    parser.add_argument(
        "--stage1_encoder_checkpoint",
        type=str,
        default="weights/contrastive_stage1/encoder_epoch_5.weights.h5",
    )
    parser.add_argument(
        "--stage2_encoder_checkpoint",
        type=str,
        default="weights/contrastive_stage2/encoder_stage2_epoch_6.weights.h5",
    )
    parser.add_argument("--output_dir", type=str, default="contrastive_h_classification/runs/frozen_head")
    parser.add_argument("--systems", nargs="*", default=["og", "stage1", "stage2"])
    parser.add_argument("--epochs", type=int, default=1000000)
    parser.add_argument("--monitor", choices=["val_loss", "val_macro_f1"], default="val_loss")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--shuffle_buffer", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--expected_h_dim", type=int, default=256)
    parser.add_argument(
        "--encoder_mask_mode",
        choices=["current", "invert_visible"],
        default="current",
        help=(
            "current keeps historical behavior; invert_visible treats mask_in "
            "as visible mask for pooling and passes 1-mask_in to the OG encoder."
        ),
    )
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official_macro_f1", type=float, default=None)
    parser.add_argument("--official_accuracy", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    pretrained_path = Path(args.pretrained_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(data_root)
    num_classes = len(class_names)
    split_paths = resolve_split_paths(data_root)
    schema = detect_record_schema(data_root)
    split_record_files = {
        split: collect_record_files(path)
        for split, path in split_paths.items()
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "schema": schema,
                "num_classes": num_classes,
                "class_label_mapping": {str(i): name for i, name in enumerate(class_names)},
                "split_paths": {k: str(v) for k, v in split_paths.items()},
            },
            f,
            indent=2,
        )

    _, pretrained_cfg = build_contrastive_model_from_pretrained(
        pretrained_path=str(pretrained_path),
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
    )
    window_size = int(pretrained_cfg["window_size"])

    print("[DATA] Building datasets", flush=True)
    datasets = {
        split: build_labeled_dataset(
            record_files=split_record_files[split],
            schema=schema,
            window_size=window_size,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )
        for split, path in split_paths.items()
    }
    print("[DATA] Counting class distributions from TFRecord labels", flush=True)
    distributions = {
        split: count_labels_from_records(split_record_files[split], schema, num_classes)
        for split in split_paths
    }

    system_map = {
        "og": "og_pooled_h_mlp",
        "original": "og_pooled_h_mlp",
        "stage1": "stage1_pooled_h_mlp",
        "stage2": "stage2_pooled_h_mlp",
    }
    requested = {system_map.get(item, item) for item in args.systems}
    specs = [spec for spec in default_systems(args) if spec.key in requested]
    if not specs:
        raise ValueError(f"No systems selected from --systems={args.systems}")

    results = []
    for spec in specs:
        print(f"\n[RUN] {spec.display_name}", flush=True)
        result = train_one_system(
            spec=spec,
            datasets=datasets,
            pretrained_path=pretrained_path,
            output_dir=output_dir,
            class_names=class_names,
            args=args,
            distributions=distributions,
        )
        results.append(result)

    write_summary_table(
        output_dir=output_dir,
        results=results,
        official_macro_f1=args.official_macro_f1,
        official_accuracy=args.official_accuracy,
    )
    print(f"\n[DONE] Results written to {output_dir}")


if __name__ == "__main__":
    main()
