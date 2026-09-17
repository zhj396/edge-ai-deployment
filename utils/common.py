import argparse
import random
from pathlib import Path
from typing import List, Union, Optional

from . import get_logger

logger = get_logger(__name__)


# =========================================================
# File / Directory Existence Validator
# =========================================================
class PathValidationError(argparse.ArgumentTypeError, ValueError):
    """Path-validation failure, surfaced verbatim by both call paths.

    Inherits ``ValueError`` so library callers (and their tests) keep the
    familiar contract, and ``argparse.ArgumentTypeError`` so argparse
    (``type=existing_validation``) prints this message instead of its generic
    "invalid existing_validation value: '...'" wrapper.
    """


# Git-ignored artifact directories (see ARTIFACTS.md): a missing path under
# one of them usually means the Release assets were never downloaded, not
# that the caller passed a wrong path.
_ARTIFACT_DIRS = ("models", "data")


def _artifact_hint(p: Path) -> str:
    """Return an ARTIFACTS.md pointer when ``p`` looks like a repo artifact."""
    if any(part in _ARTIFACT_DIRS for part in p.parts):
        return (
            "\nHint: models/ and data/ are git-ignored. If you expected a "
            "Release artifact here, run the download + sha256-verify steps "
            "in ARTIFACTS.md first."
        )
    return ""


def existing_validation(path: Union[str, Path]) -> Path:
    """
    Validate that the path exists as a file (image or model) or directory.
    Supported formats:
        - Images: .jpg, .jpeg, .png, .bmp, .tiff, .webp
        - Models: .pt, .onnx
    Missing paths under ``models/`` or ``data/`` additionally point at
    ARTIFACTS.md, which documents how to obtain those artifacts.
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise PathValidationError(
            f"Path does not exist: {p}{_artifact_hint(p)}"
        )

    if p.is_dir():
        return p

    if not p.is_file():
        raise PathValidationError(f"Path is neither a file nor a directory: {p}")

    suffix = p.suffix.lower()
    image_suffix = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    model_suffix = {'.pt', '.pth', '.onnx'}

    if suffix in image_suffix or suffix in model_suffix:
        return p
    else:
        raise PathValidationError(
            f"Unsupported file format: {suffix}\n"
            f"Supported: images (.jpg/.png etc), models (.pt/.onnx)"
        )


# =========================================================
# Image Loading
# =========================================================
def load_images(
    image_paths: Union[str, Path, List[Union[str, Path]]],
    max_images: Optional[int] = None,
    recursive: bool = True,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> List[Path]:
    """
    Load image paths from files, directories, or a mix of both.

    Args:
        image_paths: Single file path, directory, or list (mixed types supported)
        max_images: Maximum number of images to return
        recursive: Whether to recurse into subdirectories
        shuffle: Whether to shuffle the result
        seed: Random seed for reproducible shuffling
    """
    valid_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if isinstance(image_paths, (str, Path)):
        paths = [Path(image_paths)]
    else:
        paths = [Path(p) for p in image_paths]

    expanded_paths: List[Path] = []

    for p in paths:
        p = p.expanduser().resolve()

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

        if p.is_dir():
            glob_func = p.rglob if recursive else p.glob
            found = [
                f for f in glob_func("*")
                if f.is_file() and f.suffix.lower() in valid_suffix
            ]
            expanded_paths.extend(found)
            logger.debug(f"Directory {p.name}: found {len(found)} images")

        elif p.is_file():
            if p.suffix.lower() in valid_suffix:
                expanded_paths.append(p)
            else:
                logger.warning(f"Unsupported file format, skipped: {p}")

    result = sorted(set(expanded_paths))  # Deduplicate + sort

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(result)

    if max_images is not None and len(result) > max_images:
        result = result[:max_images]

    logger.info(f"Loaded {len(result)} images (recursive={recursive}, shuffle={shuffle})")

    return result
