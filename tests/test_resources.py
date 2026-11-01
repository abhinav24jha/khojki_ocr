import hashlib
import json

import pytest

from khojki_ocr.resources import default_checkpoint, resource_path, review_font


def test_default_model_matches_published_card(project_root):
    card = json.loads((project_root / "models/model-card.json").read_text())
    path = default_checkpoint()
    assert path.stat().st_size == card["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == card["sha256"]
    assert review_font().is_file()
    assert resource_path("assets", "OFL.txt").is_file()


@pytest.mark.parametrize(
    "group,filename", [("unknown", "font.ttf"), ("assets", "../outside")]
)
def test_resource_lookup_rejects_invalid_paths(group, filename):
    with pytest.raises(ValueError):
        resource_path(group, filename)


def test_missing_resource_has_clear_error():
    with pytest.raises(FileNotFoundError, match="Bundled resource not found"):
        resource_path("assets", "does-not-exist.ttf")
