"""
Data Augmentation for Light Curves (Astronomical Time Series)

This module provides label-preserving augmentation strategies for contrastive learning.
All augmentations maintain the physical characteristics that define the star class.

For SimCLR-style contrastive learning, we need to create two different "views" 
of each light curve that represent the same astronomical object.
"""

import tensorflow as tf


# ============================================================================
# INDIVIDUAL AUGMENTATION FUNCTIONS
# ============================================================================

@tf.function
def time_jitter(magnitudes, times, mask, max_shift=0.1, seed=None):
    """
    Shift time coordinates by a small random amount.
    
    This simulates observing the same star at slightly different phases
    or with a small timing offset in the observation schedule.
    
    Args:
        magnitudes: (batch, window_size, 1) - brightness values
        times: (batch, window_size, 1) - time coordinates
        mask: (batch, window_size, 1) - visibility mask
        max_shift: Maximum fraction of time range to shift
        seed: Random seed for reproducibility
    
    Returns:
        magnitudes (unchanged), shifted times, mask (unchanged)
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    batch_size = tf.shape(times)[0]
    
    # Calculate time range for each sample
    time_range = tf.reduce_max(times, axis=1, keepdims=True) - \
                 tf.reduce_min(times, axis=1, keepdims=True)
    
    # Random shift for each sample in batch
    shift = tf.random.uniform(
        shape=(batch_size, 1, 1),
        minval=-max_shift,
        maxval=max_shift,
        dtype=times.dtype
    ) * time_range
    
    times_shifted = times + shift
    
    return magnitudes, times_shifted, mask


@tf.function
def amplitude_scaling(magnitudes, times, mask, scale_range=(0.95, 1.05), seed=None):
    """
    Scale the magnitude amplitude by a small random factor.
    
    This simulates small variations in observing conditions, atmospheric effects,
    or intrinsic brightness variations that don't change the star's type.
    
    Note: In astronomy, magnitudes are logarithmic and inverted 
    (brighter = smaller number), so we scale around the mean.
    
    Args:
        magnitudes: (batch, window_size, 1) - brightness values
        times: (batch, window_size, 1) - time coordinates
        mask: (batch, window_size, 1) - visibility mask
        scale_range: (min, max) scaling factors
        seed: Random seed
    
    Returns:
        scaled magnitudes, times (unchanged), mask (unchanged)
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    batch_size = tf.shape(magnitudes)[0]
    
    # Random scale factor for each sample
    scale = tf.random.uniform(
        shape=(batch_size, 1, 1),
        minval=scale_range[0],
        maxval=scale_range[1],
        dtype=magnitudes.dtype
    )
    
    # Scale around the mean magnitude
    mean_mag = tf.reduce_mean(magnitudes, axis=1, keepdims=True)
    magnitudes_scaled = mean_mag + (magnitudes - mean_mag) * scale
    
    return magnitudes_scaled, times, mask


@tf.function
def add_gaussian_noise(magnitudes, times, mask, noise_level=0.02, seed=None):
    """
    Add Gaussian noise to magnitude measurements.
    
    This simulates observational noise, photon counting statistics,
    atmospheric turbulence, and instrumental effects.
    
    Args:
        magnitudes: (batch, window_size, 1) - brightness values
        times: (batch, window_size, 1) - time coordinates
        mask: (batch, window_size, 1) - visibility mask
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed
    
    Returns:
        noisy magnitudes, times (unchanged), mask (unchanged)
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    noise = tf.random.normal(
        shape=tf.shape(magnitudes),
        mean=0.0,
        stddev=noise_level,
        dtype=magnitudes.dtype
    )
    
    magnitudes_noisy = magnitudes + noise
    
    return magnitudes_noisy, times, mask


@tf.function
def random_masking(magnitudes, times, mask, drop_prob=0.1, seed=None):
    """
    Randomly mask (hide) some observations.
    
    This simulates:
    - Missing observations due to weather/technical issues
    - Sparse sampling in time
    - Partial phase coverage
    
    Args:
        magnitudes: (batch, window_size, 1) - brightness values
        times: (batch, window_size, 1) - time coordinates
        mask: (batch, window_size, 1) - visibility mask (1=visible, 0=masked)
        drop_prob: Probability of masking each observation
        seed: Random seed
    
    Returns:
        magnitudes (unchanged), times (unchanged), updated mask
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    # Generate random dropout mask
    dropout_mask = tf.cast(
        tf.random.uniform(tf.shape(mask), dtype=tf.float32) > drop_prob,
        dtype=mask.dtype
    )
    
    # Combine with existing mask (keep points that were already masked)
    mask_augmented = mask * dropout_mask
    
    return magnitudes, times, mask_augmented


@tf.function
def temporal_crop(magnitudes, times, mask, crop_fraction=0.8, seed=None):
    """
    Randomly crop a continuous segment of the time series.
    
    This simulates observing only part of the star's variability cycle
    or having observations limited to a specific time window.
    
    Args:
        magnitudes: (batch, window_size, 1) - brightness values
        times: (batch, window_size, 1) - time coordinates
        mask: (batch, window_size, 1) - visibility mask
        crop_fraction: Fraction of time series to keep (0.8 = keep 80%)
        seed: Random seed
    
    Returns:
        magnitudes (unchanged), times (unchanged), updated mask
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    batch_size = tf.shape(times)[0]
    window_size = tf.shape(times)[1]
    
    # Calculate crop window size
    crop_size = tf.cast(tf.cast(window_size, tf.float32) * crop_fraction, tf.int32)
    
    # Random start position for each sample in batch
    max_start = window_size - crop_size
    start_idx = tf.random.uniform(
        shape=(batch_size,),
        minval=0,
        maxval=max_start + 1,
        dtype=tf.int32
    )
    
    # Create mask for cropped region
    indices = tf.range(window_size)
    indices_broadcast = tf.expand_dims(tf.expand_dims(indices, 0), -1)  # (1, window_size, 1)
    start_broadcast = tf.reshape(start_idx, (batch_size, 1, 1))
    end_broadcast = start_broadcast + crop_size
    
    crop_mask = tf.cast(
        tf.logical_and(
            indices_broadcast >= start_broadcast,
            indices_broadcast < end_broadcast
        ),
        dtype=mask.dtype
    )
    
    # Apply crop to existing mask
    mask_cropped = mask * crop_mask
    
    return magnitudes, times, mask_cropped


@tf.function
def amplitude_shift(magnitudes, times, mask, shift_range=(-0.1, 0.1), seed=None):
    """
    Shift all magnitudes by a constant offset.
    
    This simulates:
    - Calibration offsets between different telescopes
    - Systematic photometric zero-point errors
    - Different reference standards
    
    Args:
        magnitudes: (batch, window_size, 1) - brightness values
        times: (batch, window_size, 1) - time coordinates
        mask: (batch, window_size, 1) - visibility mask
        shift_range: (min, max) magnitude shift
        seed: Random seed
    
    Returns:
        shifted magnitudes, times (unchanged), mask (unchanged)
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    batch_size = tf.shape(magnitudes)[0]
    
    # Random shift for each sample
    shift = tf.random.uniform(
        shape=(batch_size, 1, 1),
        minval=shift_range[0],
        maxval=shift_range[1],
        dtype=magnitudes.dtype
    )
    
    magnitudes_shifted = magnitudes + shift
    
    return magnitudes_shifted, times, mask



# ============================================================================
# CONTRASTIVE VIEW CREATION (for separated input/times/mask_in format)
# ============================================================================

def augment_one_view(mag, times, mask, config=None):
    """
    Apply a random combination of augmentations to create one contrastive view.

    Args:
        mag:    (batch, seq_len, 1) magnitudes
        times:  (batch, seq_len, 1) timestamps
        mask:   (batch, seq_len, 1) visibility mask
        config: Optional dict. If None, uses default values.

    Returns:
        (mag, times, mask) - augmented tensors, same shapes as input
    """
    if config is None:
        config = get_default_augmentation_config()
        # config = get_strong_augmentation_config()

    if tf.random.uniform(()) < config.get('time_jitter_prob', 0.5):
        mag, times, mask = time_jitter(
            mag, times, mask,
            max_shift=config.get('time_jitter_max', 0.1))

    if tf.random.uniform(()) < config.get('amplitude_scale_prob', 0.5):
        mag, times, mask = amplitude_scaling(
            mag, times, mask,
            scale_range=config.get('amplitude_scale_range', (0.95, 1.05)))

    if tf.random.uniform(()) < config.get('amplitude_shift_prob', 0.3):
        mag, times, mask = amplitude_shift(
            mag, times, mask,
            shift_range=config.get('amplitude_shift_range', (-0.1, 0.1)))

    if tf.random.uniform(()) < config.get('noise_prob', 0.8):
        mag, times, mask = add_gaussian_noise(
            mag, times, mask,
            noise_level=config.get('noise_level', 0.02))

    if tf.random.uniform(()) < config.get('masking_prob', 0.3):
        mag, times, mask = random_masking(
            mag, times, mask,
            drop_prob=config.get('drop_prob', 0.1))

    if tf.random.uniform(()) < config.get('crop_prob', 0.2):
        mag, times, mask = temporal_crop(
            mag, times, mask,
            crop_fraction=config.get('crop_fraction', 0.8))

    return mag, times, mask


def create_contrastive_views(sample, config=None):
    """
    Create two independently-augmented views for contrastive learning.

    Designed for the separated-tensor format used by the ASTROMER encoder:
        sample = {'input': [seq_len,1], 'times': [seq_len,1], 'mask_in': [seq_len,1]}

    Args:
        sample: Dict with 'input', 'times', 'mask_in' (each [seq_len, 1])
        config: Optional augmentation config dict.  None = default.

    Returns:
        (view_a, view_b): Tuple of dicts, each with 'input', 'times', 'mask_in'
    """
    mag  = tf.expand_dims(sample['input'],   0)   # [1, seq_len, 1]
    times = tf.expand_dims(sample['times'],  0)
    mask  = tf.expand_dims(sample['mask_in'], 0)

    mag_a, time_a, mask_a = augment_one_view(mag, times, mask, config)
    mag_b, time_b, mask_b = augment_one_view(mag, times, mask, config)

    view_a = {'input': mag_a[0], 'times': time_a[0], 'mask_in': mask_a[0]}
    view_b = {'input': mag_b[0], 'times': time_b[0], 'mask_in': mask_b[0]}

    return view_a, view_b


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

def get_default_augmentation_config():
    """
    Get default augmentation configuration for light curves.
    
    Returns:
        Dict with augmentation parameters
    """
    return {
        # Time jitter
        'time_jitter_prob': 0.5,
        'time_jitter_max': 0.1,
        
        # Amplitude scaling
        'amplitude_scale_prob': 0.5,
        'amplitude_scale_range': (0.95, 1.05),
        
        # Amplitude shift
        'amplitude_shift_prob': 0.3,
        'amplitude_shift_range': (-0.1, 0.1),
        
        # Gaussian noise
        'noise_prob': 0.8,
        'noise_level': 0.01,
        
        # Random masking
        'masking_prob': 0.3,
        'drop_prob': 0.1,
        
        # Temporal crop
        'crop_prob': 0.2,
        'crop_fraction': 0.8,
    }


def get_strong_augmentation_config():
    """
    Stronger augmentation for more robust representations.
    """
    return {
        # 'time_jitter_prob': 0.7,
        # 'time_jitter_max': 0.2,
        # 'amplitude_scale_prob': 0.7,
        # 'amplitude_scale_range': (0.9, 1.1),
        # 'amplitude_shift_prob': 0.5,
        # 'amplitude_shift_range': (-0.2, 0.2),
        # 'noise_prob': 0.9,
        # 'noise_level': 0.02,
        # 'masking_prob': 0.5,
        # 'drop_prob': 0.15,
        # 'crop_prob': 0.3,
        # 'crop_fraction': 0.7,
        
        'time_jitter_prob': 0.6,
        'time_jitter_max': 0.1,
        'amplitude_scale_prob': 0.7,
        'amplitude_scale_range': (0.9, 1.1),
        'amplitude_shift_prob': 0.5,
        'amplitude_shift_range': (-0.2, 0.2),
        'noise_prob': 0.9,
        'noise_level': 0.02,
        'masking_prob': 0.5,
        'drop_prob': 0.15,
        'crop_prob': 0.2,
        'crop_fraction': 0.8,
    }



# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=== Light Curve Augmentation Module ===")
    print("\nThis module provides astronomy-aware augmentations for contrastive learning.")
    print("\nAvailable augmentations:")
    print("  1. time_jitter - Shift time coordinates")
    print("  2. amplitude_scaling - Scale brightness amplitude")
    print("  3. amplitude_shift - Shift brightness baseline")
    print("  4. add_gaussian_noise - Add measurement noise")
    print("  5. random_masking - Hide some observations")
    print("  6. temporal_crop - Keep only part of time series")
    print("\nUsage:")
    print("  config = None for `get_default_augmentation_config()` or type for `get_strong_augmentation_config()`")
    print("  view_a, view_b = create_contrastive_views(sample, config)")
