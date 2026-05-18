"""
InfoNCE Loss (NT-Xent) for Unsupervised Contrastive Learning

THE FORMULA:
──────────────────────────

Given a batch of N light curves, we create 2 augmented views per sample.
So we have 2N embeddings total: z_1, z_2, ..., z_{2N}

For a random sample, say z_i, its "positive" is the other view of the same light curve.
Its "negatives" are all other 2N-2 embeddings.

The InfoNCE loss for anchor z_i is:

    L_i = -log(  exp(s(i, p(i)) / τ)  /  sum_a≠i exp(s(i, a) / τ)  )

Where:
  • s(i, j) = z_i · z_j / ||z_i|| ||z_j||  (cosine similarity, already normalized)
  • τ (tau) = temperature parameter (controls hardness of negatives)
  • p(i) = positive index (the other view of sample i)

  
Batch Size vs. τ:
──────────────────────────
Rule of thumb:
  • Small batch (16-32): Use τ ≥ 0.5 for stability
  • Medium batch (64-128): τ = 0.1-0.3 is good
  • Large batch (256+): τ = 0.05 works well
"""

import tensorflow as tf
import numpy as np


def infonce_loss(z_a, z_b, tau=0.1):
    """
    Args:
        z_a: Tensor of shape (batch_size, embedding_dim)
        z_b: Tensor of shape (batch_size, embedding_dim)
        tau: Temperature parameter
    
    Returns:
        loss: Scalar tensor
    """
    batch_size = tf.shape(z_a)[0]
    tau = tf.cast(tau, z_a.dtype)
    
    z_a = tf.math.l2_normalize(z_a, axis=1)
    z_b = tf.math.l2_normalize(z_b, axis=1)
    
    z = tf.concat([z_a, z_b], axis=0)
    
    similarity = tf.matmul(z, z, transpose_b=True) / tau
    
    eye = tf.eye(batch_size, dtype=similarity.dtype)
    zeros = tf.zeros_like(eye)
    positive_mask = tf.concat(
        [tf.concat([zeros, eye], axis=1),
         tf.concat([eye, zeros], axis=1)],
        axis=0
    )
    
    self_mask = tf.eye(2 * batch_size, dtype=similarity.dtype)
    
    similarity_masked = similarity - self_mask * 1e8
    
    pos = tf.reduce_sum(positive_mask * similarity, axis=1)
    log_denom = tf.reduce_logsumexp(similarity_masked, axis=1)
    
    log_prob = pos - log_denom
    
    loss = -tf.reduce_mean(log_prob)
    
    return loss
