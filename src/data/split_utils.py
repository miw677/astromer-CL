from __future__ import annotations

from pathlib import Path


def collect_record_files(base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    direct_files = sorted(base_dir.glob("*.record"))
    if direct_files:
        return direct_files

    return sorted(base_dir.rglob("*.record"))


def dataset_name_from_path(record_dir):
    parts = Path(record_dir).parts
    if "records" in parts:
        idx = parts.index("records")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(record_dir).name


def suffix_from_path(record_dir):
    parts = Path(record_dir).parts
    if "records" in parts:
        idx = parts.index("records")
        if idx + 2 < len(parts):
            return Path(*parts[idx + 2 :])
    return Path()


def split_dir(base_dir, split_name):
    base_dir = Path(base_dir)
    candidates = [base_dir / split_name]
    if split_name == "val":
        candidates.extend([base_dir / "validation", base_dir / "valid"])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def resolve_train_val_root(record_dir, project_root):
    requested_dir = Path(record_dir)
    dataset_name = dataset_name_from_path(requested_dir)
    suffix = suffix_from_path(requested_dir)

    dataset_root = project_root / "data" / "records" / dataset_name
    nested_dataset_root = dataset_root / dataset_name
    old_dataset_root = project_root / "data_old" / "records" / dataset_name
    old_nested_dataset_root = old_dataset_root / dataset_name

    candidate_roots = [
        requested_dir,
        requested_dir.parent,
        requested_dir.parent.parent,
        dataset_root / suffix,
        nested_dataset_root / suffix,
        old_dataset_root / suffix,
        old_nested_dataset_root / suffix,
    ]

    for root in candidate_roots:
        train_dir = split_dir(root, "train")
        val_dir = split_dir(root, "val")
        if train_dir is not None and val_dir is not None:
            return root, train_dir, val_dir

    raise ValueError(
        f"Could not find train/val split directories under {record_dir}. "
        "Expected a package root with train/ and val/ subdirectories."
    )