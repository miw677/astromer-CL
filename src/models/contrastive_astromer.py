"""
Contrastive Learning Wrapper for Astromer Encoder

This module implements the architecture for unsupervised contrastive learning:

    Light Curve (x)
         ↓
    ENCODER (Astromer transformer)
         ↓
    h (representation, shape: [batch, seq_len, d_model])
         ↓
    POOLING (average across time)
         ↓
    h_pooled (shape: [batch, d_model])
         ↓
    PROJECTION HEAD (MLP: Dense → ReLU → Dense)
         ↓
    z (contrastive embedding, shape: [batch, proj_dim])
         ↓
    L2 NORMALIZATION
         ↓
    z_normalized (unit sphere embeddings)

Usage:
    # Build model
    model = ContrastiveAstromer(
        encoder_config={...},  # Astromer hyperparameters
        projection_dim=128,    # Output dimension of projection head
        hidden_dim=256         # Hidden layer size in projection MLP
    )
    
    # Forward pass (training)
    z_a, h_a = model(view_a, training=True, return_representation=True)
    z_b, h_b = model(view_b, training=True, return_representation=True)
    
    # Compute loss
    loss = infonce_loss(z_a, z_b, tau=0.1)
    
    # Downstream tasks: extract encoder for downstream tasks
    encoder = model.encoder
"""

import tensorflow as tf
import toml
import re
import inspect
import os
from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

from src.layers import Encoder
# from src.layers.input import AddMSKToken


def _weight_path(weight):
    if hasattr(weight, 'path'):
        return weight.path.split(':')[0]
    return weight.name.split(':')[0]


class ProjectionHead(Layer):
    """
    MLP Projection Head for Contrastive Learning
    
    Architecture:
        input → Dense(hidden_dim) → ReLU → Dense(output_dim) → L2 Norm
    
    Args:
        hidden_dim: Size of hidden layer (typically 2x - 4x the input dimension)
        output_dim: Final embedding dimension (typically 128 or 256)
        name: Layer name
    """
    def __init__(self, hidden_dim=256, output_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # MLP layers
        self.dense1 = Dense(hidden_dim, activation='relu', name='proj_hidden')
        self.dense2 = Dense(output_dim, name='proj_output')
        
    def call(self, inputs, training=False):
        """
        Forward pass through projection head
        
        Args:
            inputs: Pooled representation h_pooled, shape [batch, d_model]
            training: Whether in training mode
            
        Returns:
            z_normalized: L2-normalized embeddings, shape [batch, output_dim]
        """
        x = self.dense1(inputs, training=training)
        x = self.dense2(x, training=training)
        
        # L2 normalization: z = z / ||z||_2
        z_normalized = tf.math.l2_normalize(x, axis=-1)
        
        return z_normalized


def _filter_layer_kwargs(layer_cls, kwargs):
    """Keep explicit layer args plus Keras base Layer kwargs."""
    signature = inspect.signature(layer_cls.__init__)
    explicit = set(signature.parameters)
    keras_base = {'name', 'trainable', 'dtype', 'dynamic'}
    return {
        key: value
        for key, value in kwargs.items()
        if key in explicit or key in keras_base
    }


def _build_encoder(**kwargs):
    """Build Encoder while tolerating small ASTROMER API differences."""
    encoder_kwargs = _filter_layer_kwargs(Encoder, kwargs)
    dropped = sorted(set(kwargs) - set(encoder_kwargs))
    if dropped:
        print(f"[COMPAT] Encoder does not accept {dropped}; skipping them.")
    return Encoder(**encoder_kwargs)


class ContrastiveAstromer(Model):
    """
    Astromer Model for Unsupervised Contrastive Learning
    
    This model wraps the standard Astromer encoder and adds:
    1. Pooling layer (average across time)
    2. Projection head (2-layer MLP)
    3. L2 normalization
    
    Architecture Flow:
        Input → Encoder → Pooling → Projection Head → L2 Norm
        
    Outputs:
        z: Normalized contrastive embeddings (for InfoNCE loss)
        h: Raw encoder representations (for downstream tasks)
        
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        mixer_size: FFN hidden dimension
        dropout: Dropout rate
        projection_dim: Output dimension of projection head (default: 128)
        projection_hidden_dim: Hidden dimension of projection MLP (default: 256)
        window_size: Maximum sequence length
        trainable_mask: Whether to add learnable [MSK] token
        **kwargs: Additional encoder parameters
    """
    def __init__(self,
                 num_layers=2,
                 num_heads=2,
                 head_dim=64,
                 mixer_size=256,
                 dropout=0.1,
                 pe_base=1000,
                 pe_dim=128,
                 pe_c=1,
                 window_size=100,
                 m_alpha=-0.5,
                 mask_format='Q',
                 use_leak=False,
                 temperature=0.,
                 projection_dim=128,
                 projection_hidden_dim=256,
                 trainable_mask=False,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Store hyperparameters
        self.window_size = window_size
        self.projection_dim = projection_dim
        self.projection_hidden_dim = projection_hidden_dim
        # self.trainable_mask = trainable_mask
        self.trainable_mask = False
        
        # Build encoder (Astromer transformer)
        self.encoder = _build_encoder(
            window_size=window_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            mixer_size=mixer_size,
            dropout=dropout,
            pe_base=pe_base,
            pe_dim=pe_dim,
            pe_c=pe_c,
            m_alpha=m_alpha,
            mask_format=mask_format,
            use_leak=use_leak,
            temperature=temperature,
            use_cache=False,  # No caching during contrastive training
            latent_dim=None,
            name='encoder'
        )
        
        # # Optional: Add trainable [MSK] token
        # if trainable_mask:
        #     self.msk_token_layer = AddMSKToken(
        #         trainable=True,
        #         window_size=window_size,
        #         on=['input'],
        #         name='msk_token'
        #     )
        
        # Build projection head
        self.projection_head = ProjectionHead(
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            name='projection_head'
        )
        
    def call(self, inputs, training=False, return_representation=False):
        """
        Forward pass through contrastive model
        
        Args:
            inputs: Dictionary with keys:
                - 'input': magnitudes, shape [batch, seq_len, 1]
                - 'times': timestamps, shape [batch, seq_len, 1]
                - 'mask_in': attention mask, shape [batch, seq_len, 1]
            training: Whether in training mode
            return_representation: If True, return both (z, h). If False, return only z.
            
        Returns:
            If return_representation=False (default):
                z: Normalized embeddings, shape [batch, projection_dim]
            If return_representation=True:
                (z, h): Tuple of (normalized embeddings, raw representations)
                    z: shape [batch, projection_dim]
                    h: shape [batch, window_size, d_model]
        """
        # # Add [MSK] token if enabled
        # if self.trainable_mask:
        #     x = self.msk_token_layer(inputs)
        # else:
        #     x = inputs
        x = inputs
        
        # Encode: x → h
        # h has shape [batch, seq_len, d_model]
        h = self.encoder(x, training=training)
        
        # Pool across time dimension
        # Average pooling: h_pooled = mean(h, axis=1)
        # Shape: [batch, d_model]
        mask = tf.cast(x['mask_in'], h.dtype)    # [B, T, 1]
        h_sum = tf.reduce_sum(h * mask, axis=1)  # [B, D]
        denom = tf.reduce_sum(mask, axis=1)      # [B, 1]
        # h_pooled = tf.reduce_mean(h, axis=1)
        h_pooled = h_sum / tf.maximum(denom, tf.constant(1e-8, dtype=h.dtype))
        
        # Project: h_pooled → z (normalized)
        # Shape: [batch, projection_dim]
        z = self.projection_head(h_pooled, training=training)
        
        if return_representation:
            return z, h
        else:
            return z
    
    def get_encoder(self):
        """
        Extract the encoder for downstream tasks
        
        Returns:
            encoder: The trained Astromer transformer encoder
        """
        return self.encoder


def build_contrastive_model(window_size=100,
                            num_layers=2,
                            num_heads=2,
                            head_dim=64,
                            mixer_size=256,
                            dropout=0.1,
                            projection_dim=128,
                            projection_hidden_dim=256,
                            **kwargs):
    """
    Convenience function to build ContrastiveAstromer model
    
    This creates the model with standard hyperparameters.
    For custom configurations, instantiate ContrastiveAstromer directly.
    
    Args:
        window_size: Maximum sequence length
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        mixer_size: FFN hidden dimension
        dropout: Dropout rate
        projection_dim: Output dimension of projection head
        projection_hidden_dim: Hidden dimension in projection MLP
        **kwargs: Additional encoder parameters
        
    Returns:
        model: ContrastiveAstromer instance
        
    Example:
        model = build_contrastive_model(
            window_size=200,
            num_layers=4,
            num_heads=4,
            projection_dim=128
        )
    """
    model = ContrastiveAstromer(
        window_size=window_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        mixer_size=mixer_size,
        dropout=dropout,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        **kwargs
    )
    
    return model


# ============================================================================
# PRETRAINED WEIGHT LOADING
# ============================================================================

def _canonical_keys_for_model_layer(model_weights):
    """
    Assign canonical keys to model weights within one attention layer.
    
    Keras auto-numbers sub-layers globally (dense_4, sequential_1, etc.)
    so we can't rely on the exact names. Instead, we classify each weight
    by its sub-component prefix and use counters to disambiguate 1st vs 2nd
    occurrences (the creation order within each layer is deterministic).
    
    Returns list of (canonical_key, weight) where canonical_key matches
    the checkpoint naming pattern (e.g. 'mha/wq/kernel', 'ffn/layer_with_weights-0/kernel').
    """
    result = []
    seq_dense_idx = 0       # counts individual weights in sequential/dense
    ln_idx = 0              # counts individual weights in layer_normalization
    leak_dense_idx = 0      # counts individual weights in standalone dense

    for w in model_weights:
        # Get sub-path after att_layer_N/
        subpath = re.sub(r'^.*/att_layer_\d+/', '', _weight_path(w))
        leaf = subpath.split('/')[-1]  # kernel, bias, gamma, or beta

        if 'head_attention_multi' in subpath:
            if '/WQ/' in subpath:
                key = f'mha/wq/{leaf}'
            elif '/WK/' in subpath:
                key = f'mha/wk/{leaf}'
            elif '/WV/' in subpath:
                key = f'mha/wv/{leaf}'
            elif '/attmerge/' in subpath:
                key = f'mha/dense/{leaf}'
            else:
                key = None

        elif 'sequential' in subpath:
            # FFN: 2 dense layers, each with kernel + bias = 4 weights
            layer_i = seq_dense_idx // 2   # 0 or 1
            key = f'ffn/layer_with_weights-{layer_i}/{leaf}'
            seq_dense_idx += 1

        elif 'layer_normalization' in subpath:
            # LayerNorm: 2 norms, each with gamma + beta = 4 weights
            norm_i = ln_idx // 2 + 1       # 1 or 2
            key = f'layernorm{norm_i}/{leaf}'
            ln_idx += 1

        elif re.match(r'(dense|reshape_leak_\d+)', subpath):
            # Standalone dense = reshape_leak
            explicit = re.match(r'reshape_leak_(\d+)/', subpath)
            leak_i = int(explicit.group(1)) if explicit else leak_dense_idx // 2 + 1
            key = f'reshape_leak_{leak_i}/{leaf}'
            leak_dense_idx += 1

        else:
            key = None

        result.append((key, w))

    return result


def _leaf_name(weight, fallback):
    leaf = _weight_path(weight).split('/')[-1]
    return leaf if leaf in {'kernel', 'bias', 'gamma', 'beta'} else fallback


def _dense_pairs(prefix, layer):
    pairs = []
    for i, w in enumerate(layer.weights):
        fallback = 'kernel' if i == 0 else 'bias'
        pairs.append((f"{prefix}/{_leaf_name(w, fallback)}", w))
    return pairs


def _norm_pairs(prefix, layer):
    pairs = []
    for i, w in enumerate(layer.weights):
        fallback = 'gamma' if i == 0 else 'beta'
        pairs.append((f"{prefix}/{_leaf_name(w, fallback)}", w))
    return pairs


def _canonical_keys_for_attention_block(block):
    """Map an AttentionBlock's own sublayers to checkpoint keys without path parsing."""
    pairs = []

    if all(hasattr(block.mha, attr) for attr in ('wq', 'wk', 'wv', 'dense')):
        pairs.extend(_dense_pairs('mha/wq', block.mha.wq))
        pairs.extend(_dense_pairs('mha/wk', block.mha.wk))
        pairs.extend(_dense_pairs('mha/wv', block.mha.wv))
        pairs.extend(_dense_pairs('mha/dense', block.mha.dense))

    if hasattr(block, 'ffn') and hasattr(block.ffn, 'layers') and len(block.ffn.layers) >= 2:
        pairs.extend(_dense_pairs('ffn/layer_with_weights-0', block.ffn.layers[0]))
        pairs.extend(_dense_pairs('ffn/layer_with_weights-1', block.ffn.layers[1]))

    if hasattr(block, 'layernorm1'):
        pairs.extend(_norm_pairs('layernorm1', block.layernorm1))
    if hasattr(block, 'layernorm2'):
        pairs.extend(_norm_pairs('layernorm2', block.layernorm2))

    if hasattr(block, 'reshape_leak_1'):
        pairs.extend(_dense_pairs('reshape_leak_1', block.reshape_leak_1))
    if hasattr(block, 'reshape_leak_2'):
        pairs.extend(_dense_pairs('reshape_leak_2', block.reshape_leak_2))

    return pairs


def _audit_enabled(value):
    if value is not None:
        return bool(value)
    return os.environ.get('ASTROMER_PRETRAIN_AUDIT', '').lower() in {'1', 'true', 'yes', 'y'}


def _assign_checkpoint_weight(w, weights_path, ckpt_name):
    value = tf.train.load_variable(weights_path, ckpt_name)
    w.assign(value)
    diff = tf.reduce_max(
        tf.abs(tf.cast(w, tf.float32) - tf.cast(value, tf.float32))
    )
    return float(diff.numpy())


def load_pretrained_encoder(model, pretrained_path, audit=None):
    """
    Load ASTROMER v1 pretrained encoder weights into a ContrastiveAstromer model.
    
    Uses canonical-key matching: both checkpoint variable names and model weight
    paths are mapped to a common canonical form (layer_idx, subpath) and matched.
    This is robust to Keras global auto-numbering of sub-layers.
    
    Args:
        model: A ContrastiveAstromer instance (must be built, i.e. called once)
        pretrained_path: Path to the pretrained directory containing
                        config.toml, weights.index, weights.data-*
    
    Returns:
        int: Number of weights successfully loaded
    """
    weights_path = str(Path(pretrained_path) / 'weights')

    if not Path(weights_path + '.index').exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {weights_path}. "
            f"Expected weights.index and weights.data-* files."
        )

    # ---- Build checkpoint lookup: (layer_idx, canonical_subpath) → (name, shape) ----
    ckpt_all = tf.train.list_variables(weights_path)
    ckpt_lookup = {}

    for name, shape in ckpt_all:
        if not name.startswith('layer_with_weights-0/'):
            continue  # skip regression head (layer_with_weights-1)

        if '/inp_transform/' in name:
            subpath = re.sub(
                r'^layer_with_weights-0/inp_transform/', '', name
            ).replace('/.ATTRIBUTES/VARIABLE_VALUE', '')
            ckpt_lookup[(-1, subpath)] = (name, shape)
        else:
            m = re.search(r'/enc_layers/(\d+)/', name)
            if m:
                layer_idx = int(m.group(1))
                subpath = re.sub(
                    r'^layer_with_weights-0/enc_layers/\d+/', '', name
                ).replace('/.ATTRIBUTES/VARIABLE_VALUE', '')
                ckpt_lookup[(layer_idx, subpath)] = (name, shape)

    audit = _audit_enabled(audit)
    loaded = 0
    errors = []
    loaded_records = []
    loaded_keys = []

    # ---- Match inp_transform by leaf name (kernel / bias) ----
    for i, w in enumerate(model.encoder.inp_transform.weights):
        leaf = _leaf_name(w, 'kernel' if i == 0 else 'bias')
        key = (-1, leaf)
        if key in ckpt_lookup:
            ckpt_name, ckpt_shape = ckpt_lookup[key]
            if list(ckpt_shape) == list(w.shape):
                max_abs_diff = _assign_checkpoint_weight(w, weights_path, ckpt_name)
                loaded += 1
                loaded_keys.append(key)
                loaded_records.append((key, ckpt_name, _weight_path(w), list(w.shape), max_abs_diff))
            else:
                errors.append(
                    f"  inp_transform/{leaf}: shape mismatch "
                    f"ckpt={list(ckpt_shape)} model={list(w.shape)}"
                )
        else:
            errors.append(f"  inp_transform/{leaf}: not found in checkpoint")

    # ---- Match attention layers by canonical key ----
    for layer_idx, block in enumerate(model.encoder.enc_layers):
        canonical_pairs = _canonical_keys_for_attention_block(block)
        if not canonical_pairs:
            canonical_pairs = _canonical_keys_for_model_layer(block.weights)

        for canon_key, w in canonical_pairs:
            if canon_key is None:
                errors.append(
                    f"  att_layer_{layer_idx}: unmapped weight {_weight_path(w)}"
                )
                continue

            full_key = (layer_idx, canon_key)
            if full_key in ckpt_lookup:
                ckpt_name, ckpt_shape = ckpt_lookup[full_key]
                if list(ckpt_shape) == list(w.shape):
                    max_abs_diff = _assign_checkpoint_weight(w, weights_path, ckpt_name)
                    loaded += 1
                    loaded_keys.append(full_key)
                    loaded_records.append((full_key, ckpt_name, _weight_path(w), list(w.shape), max_abs_diff))
                else:
                    errors.append(
                        f"  att_layer_{layer_idx}/{canon_key}: shape mismatch "
                        f"ckpt={list(ckpt_shape)} model={list(w.shape)}"
                    )
            else:
                errors.append(
                    f"  att_layer_{layer_idx}/{canon_key}: not in checkpoint"
                )

    total = len(model.encoder.weights)
    print(f"[PRETRAINED] Loaded {loaded}/{total} encoder weights from {pretrained_path}")
    if audit:
        print("[PRETRAINED][AUDIT] Loaded weight mapping:")
        for key, ckpt_name, model_path, shape, max_abs_diff in loaded_records:
            print(
                f"  key={key} | ckpt={ckpt_name} | "
                f"model={model_path} | shape={shape} | max_abs_diff={max_abs_diff:.3e}"
            )

        duplicate_keys = sorted(
            key for key in set(loaded_keys) if loaded_keys.count(key) > 1
        )
        missing_keys = sorted(set(ckpt_lookup) - set(loaded_keys))
        bad_diffs = [
            (key, diff) for key, _, _, _, diff in loaded_records
            if diff > 0.0
        ]
        print(
            f"[PRETRAINED][AUDIT] Summary: loaded_keys={len(loaded_keys)}, "
            f"checkpoint_encoder_keys={len(ckpt_lookup)}, duplicates={len(duplicate_keys)}, "
            f"missing_checkpoint_keys={len(missing_keys)}, nonzero_diffs={len(bad_diffs)}"
        )
        if duplicate_keys:
            print("[PRETRAINED][AUDIT] Duplicate model mappings:")
            for key in duplicate_keys:
                print(f"  {key}")
        if missing_keys:
            print("[PRETRAINED][AUDIT] Checkpoint encoder keys not loaded:")
            for key in missing_keys:
                print(f"  {key}")
        if bad_diffs:
            print("[PRETRAINED][AUDIT] Nonzero assignment diffs:")
            for key, diff in bad_diffs:
                print(f"  {key}: {diff:.3e}")
    if errors:
        print(f"[PRETRAINED] Errors ({len(errors)}):")
        for e in errors:
            print(e)

    return loaded


def build_contrastive_model_from_pretrained(pretrained_path,
                                            projection_dim=128,
                                            projection_hidden_dim=256):
    """
    Build a ContrastiveAstromer model with encoder initialized from
    ASTROMER v1 pretrained weights.
    
    Reads the architecture hyperparameters from the pretrained config.toml,
    builds a matching ContrastiveAstromer, loads the encoder weights,
    and leaves the projection head randomly initialized.
    
    Args:
        pretrained_path: Path to pretrained directory (e.g. './pretrained/macho-clean')
        projection_dim: Output dimension of contrastive projection head
        projection_hidden_dim: Hidden dimension of projection MLP
        
    Returns:
        model: ContrastiveAstromer with pretrained encoder
        config: Dict of pretrained config values
    """
    # Read pretrained config
    config_file = Path(pretrained_path) / 'config.toml'
    with open(config_file, 'r') as f:
        config = toml.load(f)
    
    print(f"[PRETRAINED] Config: {config['num_layers']}L, "
          f"{config['num_heads']}H, head_dim={config['head_dim']}, "
          f"mixer={config['mixer']}, pe_dim={config['pe_dim']}")
    
    # Build model with matching architecture
    model = ContrastiveAstromer(
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        mixer_size=config['mixer'],
        dropout=config['dropout'],
        pe_base=config['pe_base'],
        pe_dim=config['pe_dim'],
        pe_c=config['pe_exp'],
        window_size=config['window_size'],
        m_alpha=config['m_alpha'],
        mask_format=config['mask_format'],
        use_leak=config['use_leak'],
        temperature=config.get('temperature', 0.0),
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        # trainable_mask=not config.get('no_msk_token', False),
        trainable_mask=False,
    )
    
    # Build the model by doing a dummy forward pass
    dummy = {
        'input': tf.zeros([2, config['window_size'], 1]),
        'times': tf.zeros([2, config['window_size'], 1]),
        'mask_in': tf.ones([2, config['window_size'], 1])
    }
    _ = model(dummy, training=False)
    
    # Load pretrained encoder weights
    n_loaded = load_pretrained_encoder(model, pretrained_path)
    
    total_encoder = len(model.encoder.weights)
    total_proj = len(model.projection_head.weights)
    print(f"[PRETRAINED] Encoder: {n_loaded}/{total_encoder} weights loaded (pretrained)")
    print(f"[PRETRAINED] Projection head: {total_proj} weights (randomly initialized)")
    
    return model, config
