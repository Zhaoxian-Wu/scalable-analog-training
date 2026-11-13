#!/usr/bin/env python3
"""Restore a frozen ImageNet experiment source tree from this repository.

The default output preserves the exact Python bytes used by a recorded run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


RELEASE_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((RELEASE_DIR / "source_manifest.json").read_text())
POOL = RELEASE_DIR / "frozen_source" / "code"
MP_DIGEST_FILES = (
    "{entry}",
    "imagenet_reference/train.py",
    "imagenet_reference/train_analog_resnet{depth}.py",
    "imagenet_reference/train_analog_resnet18_adamw.py",
    "imagenet_reference/upstream/presets.py",
    "sout_resnet_cifar.py",
    "E1-nanoGPT/MixedPrecisionOptim/LogicalAdamW.py",
    "E1-nanoGPT/MixedPrecisionOptim/LogicalOptimizers.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_digest(code_dir: Path, recipe_name: str, recipe: dict) -> str:
    if recipe["source_digest_scope"] == "all_python_files":
        paths = sorted(recipe["source_files"])
    elif recipe["source_digest_scope"] == "selected_imports":
        depth = recipe_name.split("-", 1)[0].removeprefix("resnet")
        paths = [name.format(entry=recipe["entry_point"], depth=depth) for name in MP_DIGEST_FILES]
    else:
        raise ValueError(f"Unknown digest scope: {recipe['source_digest_scope']}")

    digest = hashlib.sha256()
    for relative in paths:
        path = code_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(f"code/{relative}".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_pool() -> None:
    for relative, expected in MANIFEST["source_files"].items():
        path = POOL / relative
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen source changed: {path}")
    for relative, expected in MANIFEST["digital_reference"]["files"].items():
        path = RELEASE_DIR / "digital_reference" / relative
        if sha256(path) != expected:
            raise RuntimeError(f"Digital source changed: {path}")
    continuation = RELEASE_DIR / "entrypoints" / "resnet18_samp_continuation.py"
    expected = MANIFEST["recipes"]["resnet18-samp"]["continuation_entry_sha256"]
    if sha256(continuation) != expected:
        raise RuntimeError(f"Continuation driver changed: {continuation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=sorted(MANIFEST["recipes"]))
    parser.add_argument("--output-dir", type=Path, help="New directory that will contain source/code")
    parser.add_argument("--verify-only", action="store_true", help="Verify repository files and historical digest")
    args = parser.parse_args()
    if not args.verify_only and args.output_dir is None:
        parser.error("--output-dir is required unless --verify-only is used")

    verify_pool()
    recipe = MANIFEST["recipes"][args.recipe]
    pooled_digest = recipe_digest(POOL, args.recipe, recipe)
    if pooled_digest != recipe["source_digest"]:
        raise RuntimeError(f"Historical source digest mismatch for {args.recipe}: {pooled_digest}")

    if args.verify_only:
        print(f"{args.recipe}: verified {pooled_digest}")
        return

    destination = args.output_dir.resolve()
    if destination.exists():
        if not destination.is_dir():
            parser.error(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            parser.error(f"Output directory is not empty: {destination}")
    code_dir = destination / "source" / "code"
    for relative in recipe["source_files"]:
        target = code_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(POOL / relative, target)
    for extra in ("LICENSE", "README.md"):
        relative = Path("imagenet_reference/upstream") / extra
        target = code_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(POOL / relative, target)

    restored_digest = recipe_digest(code_dir, args.recipe, recipe)
    if restored_digest != pooled_digest:
        raise RuntimeError(f"Restored digest mismatch for {args.recipe}: {restored_digest}")
    print(f"{args.recipe}: restored {code_dir} ({restored_digest})")


if __name__ == "__main__":
    main()
