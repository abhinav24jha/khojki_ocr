"""Trusted training checkpoint loading and interruption-safe snapshot writes."""

from pathlib import Path
import os
import tempfile

from .vocabulary import VOCAB


def load_single_checkpoint(path):
    """Load one model for training; inference ensemble files are not warm starts.

    Training snapshots include Python/NumPy RNG state and optimizer metadata,
    so pickle loading is necessary. Only use a checkpoint from a trusted source.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "ensemble_states" in saved:
        raise ValueError("Training requires a single-model checkpoint, not an ensemble")
    if saved.get("vocab") != VOCAB:
        raise ValueError("Checkpoint vocabulary mismatch")
    if "model" not in saved:
        raise ValueError("Checkpoint does not contain model weights")
    return saved


def save_checkpoint(checkpoint, path):
    """Replace a checkpoint only after its complete new snapshot is on disk."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
