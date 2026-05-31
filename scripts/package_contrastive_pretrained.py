"""Package trained Contrastive Astromer checkpoints for publication/loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import toml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.contrastive_package import package_contrastive_pretrained  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Contrastive Astromer pretrained package.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--package_dir", required=True)
    parser.add_argument("--full_checkpoint", default=None)
    parser.add_argument("--encoder_checkpoint", default=None)
    parser.add_argument("--classifier_checkpoint_index", default=None)
    parser.add_argument("--history_path", default=None)
    parser.add_argument("--plot_path", default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--stage1_encoder_checkpoint", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--metadata_json",
        default=None,
        help="Optional JSON file with extra config/metadata to include.",
    )
    args = parser.parse_args()

    config = {
        "packaging_script": "scripts/package_contrastive_pretrained.py",
        "data_dir": args.data_dir,
        "pretrained_path": args.pretrained_path,
        "mask_semantics": "contrastive mask_in is visible; model always sends 1-mask_in to encoder",
        "projection_dim": args.projection_dim,
        "projection_hidden_dim": args.projection_hidden_dim,
        "stage1_checkpoint": args.stage1_checkpoint,
        "stage1_encoder_checkpoint": args.stage1_encoder_checkpoint,
        "notes": args.notes,
    }
    if args.pretrained_path:
        config_path = Path(args.pretrained_path) / "config.toml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config["pretrained_config"] = toml.load(f)

    if args.metadata_json:
        with open(args.metadata_json, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    package_contrastive_pretrained(
        stage=args.stage,
        checkpoint_dir=args.checkpoint_dir,
        package_dir=args.package_dir,
        config=config,
        full_checkpoint=args.full_checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        classifier_checkpoint_index=args.classifier_checkpoint_index,
        history_path=args.history_path,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
