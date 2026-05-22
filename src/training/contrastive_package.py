"""Utilities for packaging Contrastive Astromer pretrained artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

import toml


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _as_tomlable(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(k): _as_tomlable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_tomlable(v) for v in value]
    return _as_jsonable(value)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _copy_if_exists(src: Path | None, dst: Path) -> str | None:
    if src is None:
        return None
    src = Path(src)
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst.name


def _latest_by_epoch(files: list[Path], pattern: str) -> Path | None:
    regex = re.compile(pattern)
    candidates: list[tuple[int, Path]] = []
    for path in files:
        match = regex.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def infer_contrastive_checkpoints(checkpoint_dir: str | Path, stage: str) -> dict[str, Path | None]:
    """Infer the latest full/encoder/classifier checkpoints from a training directory."""
    root = Path(checkpoint_dir)
    files = [p for p in root.iterdir() if p.is_file()] if root.exists() else []

    if stage == "stage1":
        full = _latest_by_epoch(files, r"epoch_(\d+)_loss_.*_acc_.*\.weights\.h5$")
        encoder = _latest_by_epoch(files, r"encoder_epoch_(\d+)\.weights\.h5$")
        classifier = None
    elif stage == "stage2":
        full = _latest_by_epoch(files, r"stage2_epoch_(\d+)\.weights\.h5$")
        encoder = _latest_by_epoch(files, r"encoder_stage2_epoch_(\d+)\.weights\.h5$")
        classifier = _latest_by_epoch(files, r"classifier_stage2_epoch_(\d+)\.index$")
        if full is None:
            full = _latest_by_epoch(files, r"stage2_fresh_epoch_(\d+)\.weights\.h5$")
        if encoder is None:
            encoder = _latest_by_epoch(files, r"encoder_stage2_fresh_epoch_(\d+)\.weights\.h5$")
        if classifier is None:
            classifier = _latest_by_epoch(files, r"classifier_stage2_fresh_epoch_(\d+)\.index$")
    else:
        raise ValueError(f"Unknown stage={stage!r}; expected 'stage1' or 'stage2'")

    return {"full": full, "encoder": encoder, "classifier_index": classifier}


def package_contrastive_pretrained(
    *,
    stage: str,
    checkpoint_dir: str | Path,
    package_dir: str | Path,
    config: dict[str, Any],
    full_checkpoint: str | Path | None = None,
    encoder_checkpoint: str | Path | None = None,
    classifier_checkpoint_index: str | Path | None = None,
    history_path: str | Path | None = None,
    plot_path: str | Path | None = None,
) -> Path:
    """Create a stable, publishable Contrastive Astromer pretrained package.

    The package intentionally keeps both the full contrastive checkpoint and the
    encoder-only checkpoint. The former is the scientific counterpart to a
    pretrained contrastive model; the latter is the downstream convenience
    artifact used by classifiers.
    """
    checkpoint_dir = Path(checkpoint_dir)
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    inferred = infer_contrastive_checkpoints(checkpoint_dir, stage)
    full_src = Path(full_checkpoint) if full_checkpoint else inferred["full"]
    encoder_src = Path(encoder_checkpoint) if encoder_checkpoint else inferred["encoder"]
    classifier_idx_src = (
        Path(classifier_checkpoint_index)
        if classifier_checkpoint_index
        else inferred["classifier_index"]
    )

    if full_src is None or not full_src.exists():
        raise FileNotFoundError(
            f"Could not find full contrastive checkpoint for {stage} in {checkpoint_dir}"
        )
    if encoder_src is None or not encoder_src.exists():
        raise FileNotFoundError(
            f"Could not find encoder checkpoint for {stage} in {checkpoint_dir}"
        )

    artifacts: dict[str, str | None] = {}
    artifacts["full_model"] = _copy_if_exists(full_src, package_dir / "full_model.weights.h5")
    artifacts["encoder"] = _copy_if_exists(encoder_src, package_dir / "encoder.weights.h5")

    if history_path is None:
        if stage == "stage1":
            default_history = checkpoint_dir / "history.json"
        else:
            default_history = checkpoint_dir / "history_stage2.json"
            if not default_history.exists():
                default_history = checkpoint_dir / "history_stage2_fresh.json"
        history_path = default_history if default_history.exists() else None
    artifacts["history"] = _copy_if_exists(
        Path(history_path) if history_path else None,
        package_dir / "history.json",
    )

    if plot_path is None:
        candidates = [
            checkpoint_dir / "training_curves.png",
            checkpoint_dir / "stage2_val_loss_retrieval.png",
            checkpoint_dir / "stage2_fresh_val_loss_retrieval.png",
        ]
        plot_path = next((p for p in candidates if p.exists()), None)
    artifacts["training_plot"] = _copy_if_exists(
        Path(plot_path) if plot_path else None,
        package_dir / "training_curves.png",
    )

    if classifier_idx_src is not None and classifier_idx_src.exists():
        prefix = classifier_idx_src.with_suffix("")
        data_src = Path(str(prefix) + ".data-00000-of-00001")
        idx_dst = package_dir / "aux_classifier.index"
        data_dst = package_dir / "aux_classifier.data-00000-of-00001"
        artifacts["aux_classifier_index"] = _copy_if_exists(classifier_idx_src, idx_dst)
        artifacts["aux_classifier_data"] = _copy_if_exists(data_src, data_dst)
    else:
        artifacts["aux_classifier_index"] = None
        artifacts["aux_classifier_data"] = None

    package_config = {
        "package_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "ContrastiveAstromer",
        "stage": stage,
        "architecture": {
            "shared_with_official": "src.layers.Encoder",
            "full_model": "src.models.contrastive_astromer.ContrastiveAstromer",
            "projection_head": "Dense(projection_hidden_dim, relu) -> Dense(projection_dim) -> l2_normalize",
            "pooling": "visible_masked_mean over encoder token outputs",
        },
        "config": _as_jsonable(config),
        "artifacts": artifacts,
        "source_checkpoints": {
            "checkpoint_dir": str(checkpoint_dir),
            "full_model": str(full_src),
            "encoder": str(encoder_src),
            "aux_classifier_index": str(classifier_idx_src) if classifier_idx_src else None,
        },
    }

    with open(package_dir / "contrastive_config.toml", "w", encoding="utf-8") as f:
        toml.dump(_as_tomlable(package_config), f)
    with open(package_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(package_config, f, indent=2)

    readme = f"""# Contrastive Astromer {stage}

This directory is a publishable pretrained package for `ContrastiveAstromer`.

## Contents

- `contrastive_config.toml`: architecture, training, and artifact metadata.
- `full_model.weights.h5`: full contrastive model weights (`encoder + projection_head`).
- `encoder.weights.h5`: encoder-only weights for downstream classifiers.
- `history.json`: training history, when available.
- `training_curves.png`: training plot, when available.
- `aux_classifier.*`: Stage 2 auxiliary CE classifier checkpoint, when available.

## Architecture

`ContrastiveAstromer` uses the same `src.layers.Encoder` architecture as official
Astromer, then applies visible-mask mean pooling and a contrastive projection
head. It is not the same full architecture as official Astromer, which uses a
masked reconstruction head.

## Loading

Use `full_model.weights.h5` when you want the full contrastive model, including
the projection head. Use `encoder.weights.h5` when you only need the pretrained
encoder for downstream classification or export into the official pipeline.
"""
    _write_text(package_dir / "README.md", readme)

    print(f"[PACKAGE] Wrote Contrastive Astromer package to: {package_dir}")
    print(f"[PACKAGE] Full model: {package_dir / 'full_model.weights.h5'}")
    print(f"[PACKAGE] Encoder:    {package_dir / 'encoder.weights.h5'}")
    return package_dir
