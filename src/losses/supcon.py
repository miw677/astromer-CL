"""
Supervised Contrastive Loss (SupCon)

For a batch of N samples with two augmented views each (2N embeddings),
positives for anchor i are all embeddings with the same class label as i,
excluding itself.

Loss:
    L = sum_i [ -1/|P(i)| * sum_{p in P(i)} log( exp(s(i,p)) / sum_{a!=i} exp(s(i,a)) ) ]
where:
    s(i,j) = (z_i^T z_j) / tau
"""

import tensorflow as tf


def supervised_contrastive_loss(z_a, z_b, labels, tau=0.1):
    """
    Compute supervised contrastive loss for two-view batches.

    Args:
        z_a: Tensor [batch_size, embedding_dim], embeddings for view A
        z_b: Tensor [batch_size, embedding_dim], embeddings for view B
        labels: Tensor [batch_size], integer class labels
        tau: Temperature

    Returns:
        Scalar loss tensor
    """
    tau = tf.cast(tau, z_a.dtype)
    z_a = tf.math.l2_normalize(z_a, axis=1)
    z_b = tf.math.l2_normalize(z_b, axis=1)

    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    labels_2n = tf.concat([labels, labels], axis=0)
    z = tf.concat([z_a, z_b], axis=0)

    logits = tf.matmul(z, z, transpose_b=True) / tau

    batch_2n = tf.shape(z)[0]
    self_mask = tf.eye(batch_2n, dtype=logits.dtype)
    non_self_mask = tf.ones_like(self_mask) - self_mask

    label_equal = tf.cast(
        tf.equal(
            tf.expand_dims(labels_2n, axis=1),
            tf.expand_dims(labels_2n, axis=0),
        ),
        logits.dtype,
    )
    positive_mask = label_equal * non_self_mask

    logits_masked = logits - self_mask * 1e9
    log_denom = tf.reduce_logsumexp(logits_masked, axis=1, keepdims=True)
    log_prob = logits - log_denom

    positive_counts = tf.reduce_sum(positive_mask, axis=1)
    per_anchor = -tf.reduce_sum(positive_mask * log_prob, axis=1)
    per_anchor = tf.math.divide_no_nan(per_anchor, positive_counts)

    valid_anchors = positive_counts > 0
    per_anchor = tf.boolean_mask(per_anchor, valid_anchors)

    # In extreme edge cases (e.g., all labels unique in tiny batch), return 0.
    loss = tf.cond(
        tf.size(per_anchor) > 0,
        lambda: tf.reduce_mean(per_anchor),
        lambda: tf.constant(0.0, dtype=logits.dtype),
    )
    return loss
