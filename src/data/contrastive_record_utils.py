from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def detect_record_schema(record_dir):
    record_dir = Path(record_dir)
    candidates = [record_dir, record_dir.parent, record_dir.parent.parent]

    for candidate in candidates:
        if (candidate / "config.toml").is_file():
            return "legacy"

    for candidate in candidates:
        if (candidate / "objects.csv").is_file():
            return "dim"

    return "dim"


def _parse_feature_lists(sequence, feature_names):
    tensors = []
    for name in feature_names:
        dense = tf.sparse.to_dense(sequence[name])
        dense = tf.reshape(tf.cast(dense, tf.float32), [1, -1])
        tensors.append(dense)
    return tf.stack(tensors, axis=2)[0]


def parse_contrastive_sample(example_proto, schema="dim"):
    if schema == "legacy":
        context_features = {
            "Label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        sequence_features = {
            "mjd": tf.io.VarLenFeature(tf.float32),
            "mag": tf.io.VarLenFeature(tf.float32),
            "errmag": tf.io.VarLenFeature(tf.float32),
        }
        context, sequence = tf.io.parse_single_sequence_example(
            serialized=example_proto,
            context_features=context_features,
            sequence_features=sequence_features,
        )
        stacked = _parse_feature_lists(sequence, ["mjd", "mag", "errmag"])
        label = tf.cast(context["Label"], tf.int32)
        return {
            "input": stacked[:, 1:2],
            "times": stacked[:, 0:1],
            "mask_in": tf.ones_like(stacked[:, 1:2], dtype=tf.float32),
            "label": label,
        }

    context_features = {
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "length": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    sequence_features = {
        "dim_0": tf.io.VarLenFeature(tf.float32),
        "dim_1": tf.io.VarLenFeature(tf.float32),
        "dim_2": tf.io.VarLenFeature(tf.float32),
    }
    context, sequence = tf.io.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features,
    )
    stacked = _parse_feature_lists(sequence, ["dim_0", "dim_1", "dim_2"])
    label = tf.cast(context["label"], tf.int32)
    return {
        "input": stacked[:, 1:2],
        "times": stacked[:, 0:1],
        "mask_in": tf.ones_like(stacked[:, 1:2], dtype=tf.float32),
        "label": label,
    }
