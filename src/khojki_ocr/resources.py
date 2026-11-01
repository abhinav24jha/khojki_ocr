"""Locate bundled resources in a source checkout or an installed wheel."""

from pathlib import Path


def resource_path(group: str, filename: str) -> Path:
    """Return an existing resource, preferring the editable source checkout."""
    if group not in {"assets", "models"} or Path(filename).name != filename:
        raise ValueError("Expected an assets/models resource with a simple filename")
    package = Path(__file__).resolve().parent
    candidates = (
        package.parents[1] / group / filename,
        package / "resources" / group / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Bundled resource not found: {group}/{filename}")


def default_checkpoint() -> Path:
    return resource_path("models", "printed-khojki.pt")


def review_font() -> Path:
    return resource_path("assets", "NotoSansKhojki-Regular.ttf")
