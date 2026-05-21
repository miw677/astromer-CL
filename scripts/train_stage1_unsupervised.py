"""
Unsupervised Contrastive Learning Training Script

    Data (TFRecords)
         ↓
    Augmentation (create two views)
         ↓
    Encoder + Projection Head
         ↓
    InfoNCE Loss
         ↓
    Backpropagation
         ↓
    Update Weights

Usage:
    python scripts/train_stage1_unsupervised.py --epochs 10 --batch_size 64 --tau 0.07
"""

import tensorflow as tf
import time
import json
from pathlib import Path
import argparse
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.contrastive_astromer import (
    build_contrastive_model,
    build_contrastive_model_from_pretrained,
)
from src.losses.infonce import infonce_loss
from src.data.augmentation import create_contrastive_views
from src.data.split_utils import collect_record_files, resolve_train_val_root
from src.data.contrastive_record_utils import detect_record_schema, parse_contrastive_sample

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Layer '.*' \(of type .*\) was passed an input with a mask attached to it.*",
    category=UserWarning,
    module=r"keras\.src\.layers\.layer",
)


RECORD_SCHEMA = "dim"

# DATA LOADING PIPELINE

def parse_tfrecord(example_proto):
    """Parse a single TFRecord example into contrastive-ready tensors."""
    sample = parse_contrastive_sample(example_proto, schema=RECORD_SCHEMA)
    return {
        'input': sample['input'],
        'times': sample['times'],
        'mask_in': sample['mask_in'],
    }


def truncate_to_window(sample, window_size=200):
    """Randomly crop to window_size timesteps (or keep all if shorter)."""
    seq_len = tf.shape(sample['input'])[0]
    max_start = tf.maximum(seq_len - window_size, 0)
    start = tf.random.uniform((), minval=0, maxval=max_start + 1, dtype=tf.int32)
    end = tf.minimum(start + window_size, seq_len)
    return {k: sample[k][start:end] for k in sample}


def create_view_pair(sample):
    """Create two augmented views (delegates to augmentation.py)."""
    return create_contrastive_views(sample)


def _build_pipeline(dataset, window_size, batch_size, shuffle_buffer=None):
    """Shared pipeline: parse -> truncate -> augment -> (shuffle) -> batch -> prefetch."""
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda s: tf.shape(s['input'])[0] > 0)
    dataset = dataset.map(
        lambda s: truncate_to_window(s, window_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(create_view_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    padding_shapes = (
        {'input': [window_size, 1], 'times': [window_size, 1], 'mask_in': [window_size, 1]},
        {'input': [window_size, 1], 'times': [window_size, 1], 'mask_in': [window_size, 1]}
    )
    padding_values = (
        {'input': 0.0, 'times': 0.0, 'mask_in': 0.0},
        {'input': 0.0, 'times': 0.0, 'mask_in': 0.0}
    )
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=padding_shapes, padding_values=padding_values
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_train_val_datasets(
    record_dir, batch_size=32, shuffle_buffer=1000,
    window_size=200
):
    """
    Load TFRecords from explicit train/val split directories.

    Args:
        record_dir: Package root containing train/ and val/ directories.

    Returns:
        (train_dataset, val_dataset)
    """
    resolved_dir, train_dir, val_dir = resolve_train_val_root(record_dir, project_root)
    global RECORD_SCHEMA
    RECORD_SCHEMA = detect_record_schema(resolved_dir)
    train_files = [str(path) for path in collect_record_files(train_dir)]
    val_files = [str(path) for path in collect_record_files(val_dir)]

    print(f"[DATA] Train split: {train_dir} ({len(train_files)} files)")
    print(f"[DATA] Val split:   {val_dir} ({len(val_files)} files)")

    if len(train_files) == 0:
        raise ValueError(
            f"No .record files found in train split under {record_dir}. "
            f"Resolved root: {resolved_dir}"
        )

    train_ds = tf.data.TFRecordDataset(
        train_files, num_parallel_reads=tf.data.AUTOTUNE
    )
    val_ds = tf.data.TFRecordDataset(
        val_files, num_parallel_reads=tf.data.AUTOTUNE
    )

    train_dataset = _build_pipeline(train_ds, window_size, batch_size, shuffle_buffer)
    val_dataset   = _build_pipeline(val_ds, window_size, batch_size, shuffle_buffer=None)

    return train_dataset, val_dataset


# TRAINING LOOP

class ContrastiveTrainer:
    """
    Trainer for unsupervised contrastive learning
    
    """
    
    def __init__(self, model, optimizer, tau=0.1, checkpoint_dir='checkpoints/stage1'):
        """
        Args:
            model: ContrastiveAstromer instance
            optimizer: TensorFlow optimizer (e.g., Adam)
            tau: Temperature parameter for InfoNCE loss
            checkpoint_dir: Where to save model checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.tau = tau
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc_tracker = tf.keras.metrics.Mean(name='train_acc')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.val_acc_tracker = tf.keras.metrics.Mean(name='val_acc')
        
        print(f"[TRAINER] Initialized with tau={tau}")
        print(f"[TRAINER] Checkpoints will be saved to: {self.checkpoint_dir}")
    
    @tf.function
    def train_step(self, view_a, view_b):
        """
        Args:
            view_a: First augmented view, shape [batch, seq_len, 1]
            view_b: Second augmented view, shape [batch, seq_len, 1]
            
        Returns:
            loss: InfoNCE loss value
            acc: Contrastive accuracy (higher = better learning)
        """
        with tf.GradientTape() as tape:
            z_a = self.model(view_a, training=True)  # [batch, projection_dim]
            z_b = self.model(view_b, training=True)  # [batch, projection_dim]
            
            loss = infonce_loss(z_a, z_b, tau=self.tau)
        
        # BACKWARD PASS: Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        logits = tf.matmul(z_a, z_b, transpose_b=True) / self.tau  # [batch, batch]
        labels = tf.range(tf.shape(z_a)[0], dtype=tf.int64)  # [0, 1, 2, ..., batch_size-1]
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        
        return loss, accuracy
    
    def train_epoch(self, dataset, epoch):
        """
        Args:
            dataset: tf.data.Dataset yielding (view_a, view_b) batches
            epoch: Current epoch number (for logging)
            
        Returns:
            avg_loss: Average loss over epoch
            avg_acc: Average accuracy over epoch
        """
        self.train_loss_tracker.reset_state()
        self.train_acc_tracker.reset_state()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}")
        print(f"{'='*70}")
        
        start_time = time.time()
        num_batches = 0
        
        for batch_idx, (view_a, view_b) in enumerate(dataset):
            # Train on this batch
            loss, acc = self.train_step(view_a, view_b)
            
            # Update metrics
            self.train_loss_tracker.update_state(loss)
            self.train_acc_tracker.update_state(acc)
            
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_loss = self.train_loss_tracker.result().numpy()
                current_acc = self.train_acc_tracker.result().numpy()
                print(f"  Batch {batch_idx+1:4d} | Loss: {current_loss:.4f} | Contrastive Matching Acc: {current_acc:.4f}")
        
        epoch_time = time.time() - start_time
        avg_loss = self.train_loss_tracker.result().numpy()
        avg_acc = self.train_acc_tracker.result().numpy()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*70}")
        print(f"Average Training Loss:     {avg_loss:.4f}")
        print(f"Average Batch-Wise Positive-Pair Matching Accuracy (InfoNCE): {avg_acc:.4f}")
        print(f"Batches:          {num_batches}")
        print(f"Time:             {epoch_time:.2f}s")
        print(f"{'='*70}\n")
        
        return avg_loss, avg_acc

    @tf.function
    def val_step(self, view_a, view_b):
        """Forward pass only (no gradient update)."""
        z_a = self.model(view_a, training=False)
        z_b = self.model(view_b, training=False)
        loss = infonce_loss(z_a, z_b, tau=self.tau)

        logits = tf.matmul(z_a, z_b, transpose_b=True) / self.tau
        labels = tf.range(tf.shape(z_a)[0], dtype=tf.int64)
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        return loss, accuracy

    def validate_epoch(self, val_dataset, epoch):
        """Run one pass over the validation set."""
        self.val_loss_tracker.reset_state()
        self.val_acc_tracker.reset_state()

        for view_a, view_b in val_dataset:
            loss, acc = self.val_step(view_a, view_b)
            self.val_loss_tracker.update_state(loss)
            self.val_acc_tracker.update_state(acc)

        avg_loss = self.val_loss_tracker.result().numpy()
        avg_acc  = self.val_acc_tracker.result().numpy()

        print(f"  [VAL]  Loss: {avg_loss:.4f}  |  Acc: {avg_acc:.4f}")
        return avg_loss, avg_acc

    def save_checkpoint(self, epoch, avg_loss, avg_acc):
        """       
        Args:
            epoch: Current epoch number
            avg_loss: Average loss for this epoch
            avg_acc: Average accuracy for this epoch
        """
        checkpoint_path = self.checkpoint_dir / f'epoch_{epoch}_loss_{avg_loss:.4f}_acc_{avg_acc:.4f}.weights.h5'
        self.model.save_weights(str(checkpoint_path))
        print(f"[CHECKPOINT] Saved to {checkpoint_path}")
        
        # Also save the encoder separately (for easy loading in Stage 2/3)
        encoder_path = self.checkpoint_dir / f'encoder_epoch_{epoch}.weights.h5'
        self.model.encoder.save_weights(str(encoder_path))
        print(f"[CHECKPOINT] Saved encoder to {encoder_path}")
    
    def train(self, train_dataset, num_epochs=5, save_every=1, val_dataset=None):
        """       
        Args:
            train_dataset: tf.data.Dataset for training
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            val_dataset: Optional validation dataset
        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Epochs:      {num_epochs}")
        print(f"Temperature: {self.tau}")
        print(f"Optimizer:   {self.optimizer.__class__.__name__}")
        print(f"Learning Rate: {self.optimizer.learning_rate.numpy()}")
        print(f"Validation:  {'Yes' if val_dataset is not None else 'No'}")
        print("="*70 + "\n")

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [],   'val_acc': [],
        }
        
        for epoch in range(1, num_epochs + 1):
            avg_loss, avg_acc = self.train_epoch(train_dataset, epoch)
            history['train_loss'].append(float(avg_loss))
            history['train_acc'].append(float(avg_acc))

            if val_dataset is not None:
                val_loss, val_acc = self.validate_epoch(val_dataset, epoch)
                history['val_loss'].append(float(val_loss))
                history['val_acc'].append(float(val_acc))
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, avg_loss, avg_acc)

        # ---- Save history & plot ----
        history_path = self.checkpoint_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[HISTORY] Saved to {history_path}")

        self._plot_curves(history)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print("Next steps:")
        print("1. Load encoder weights for Stage 2 (supervised contrastive)")
        print("2. Or use encoder for Stage 3 (Gaussian Process classifier)")
        print("="*70 + "\n")

    def _plot_curves(self, history):
        """Plot training & validation loss/accuracy curves and save to checkpoint dir."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[PLOT] matplotlib not installed -- skipping plot.")
            return

        epochs = range(1, len(history['train_loss']) + 1)
        has_val = len(history['val_loss']) > 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Loss ---
        ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train')
        if has_val:
            ax1.plot(epochs, history['val_loss'], 'r-o', markersize=3, label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('InfoNCE Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Accuracy ---
        ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train')
        if has_val:
            ax2.plot(epochs, history['val_acc'], 'r-o', markersize=3, label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Contrastive Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Stage 1 Unsupervised Contrastive Learning', fontsize=14)
        fig.tight_layout()

        plot_path = self.checkpoint_dir / 'training_curves.png'
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"[PLOT] Saved to {plot_path}")


# MAIN TRAINING SCRIPT

def main():
    parser = argparse.ArgumentParser(description='Unsupervised Contrastive Learning')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default='data/records/macho_subset/fold_0/train',
                       help='Directory containing TFRecord files')
    
    # Model parameters
    parser.add_argument('--window_size', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=2,
                       help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=64,
                       help='Dimension per attention head')
    parser.add_argument('--mixer_size', type=int, default=256,
                       help='FFN hidden dimension')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Projection head output dimension')
    parser.add_argument('--projection_hidden_dim', type=int, default=256,
                       help='Projection head hidden dimension')
    parser.add_argument(
        '--encoder_mask_mode',
        choices=['current', 'invert_visible'],
        default='current',
        help=(
            'How to pass contrastive mask_in to the OG encoder. current keeps '
            'the historical behavior; invert_visible treats mask_in as a '
            'visible mask for pooling and sends 1-mask_in to the encoder.'
        ),
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Temperature parameter for InfoNCE loss')
    
    # Pretrained model
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained ASTROMER v1 directory (e.g. pretrained/macho-clean). '
                            'When set, architecture args are read from config.toml and encoder '
                            'weights are loaded from checkpoint.')

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/stage1',
                       help='Directory to save checkpoints')
    parser.add_argument('--shuffle_buffer', type=int, default=1000,
                       help='Shuffle buffer size')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print("="*70 + "\n")
    
    # 1. BUILD MODEL
    if args.pretrained_path:
        print(f"[1/4] Building model from pretrained: {args.pretrained_path}")
        model, pt_config = build_contrastive_model_from_pretrained(
            pretrained_path=args.pretrained_path,
            projection_dim=args.projection_dim,
            projection_hidden_dim=args.projection_hidden_dim,
            encoder_mask_mode=args.encoder_mask_mode,
        )
        # Override window_size from pretrained config
        args.window_size = pt_config['window_size']
        print(f"[OK] Pretrained model built (window_size={args.window_size})\n")
    else:
        print("[1/4] Building model from scratch...")
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
        print("[OK] Model built successfully\n")
    
    # 2. CREATE OPTIMIZER
    print("[2/4] Creating optimizer...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    print(f"[OK] Optimizer: Adam(lr={args.learning_rate})\n")
    
    # 3. LOAD DATA (with train/val split)
    print("[3/4] Loading dataset...")
    train_dataset, val_dataset = load_train_val_datasets(
        record_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        window_size=args.window_size,
    )
    print("[OK] Train & Val datasets loaded\n")
    
    # 4. TRAIN MODEL
    print("[4/4] Starting training...")
    trainer = ContrastiveTrainer(
        model=model,
        optimizer=optimizer,
        tau=args.tau,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        save_every=1,
        val_dataset=val_dataset,
    )
    
    print("\n[OK] Training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()

