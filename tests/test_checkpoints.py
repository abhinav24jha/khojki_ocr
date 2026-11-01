"""Small trusted checkpoint fixtures test save/load behavior, not model accuracy."""

import pytest

from khojki_ocr.vocabulary import VOCAB

pytestmark = pytest.mark.integration


def test_atomic_checkpoint_roundtrip(tmp_path):
    import torch

    from khojki_ocr.checkpoints import load_single_checkpoint, save_checkpoint

    path = tmp_path / "nested/last.pt"
    state = {"model": {"weight": torch.tensor([1.0, 2.0])}, "vocab": VOCAB, "step": 1}
    save_checkpoint(state, path)
    loaded = load_single_checkpoint(path)
    assert loaded["step"] == 1
    assert torch.equal(loaded["model"]["weight"], state["model"]["weight"])
    assert sorted(item.name for item in path.parent.iterdir()) == ["last.pt"]


def test_failed_save_preserves_previous_checkpoint_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    import torch

    from khojki_ocr.checkpoints import save_checkpoint

    path = tmp_path / "last.pt"
    path.write_bytes(b"previous good checkpoint")

    def interrupted_save(value, stream):
        stream.write(b"partial replacement")
        raise OSError("Simulated disk interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="disk interruption"):
        save_checkpoint({"step": 2}, path)
    assert path.read_bytes() == b"previous good checkpoint"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["last.pt"]


@pytest.mark.parametrize(
    "saved,message",
    [
        ({"ensemble_states": [], "vocab": VOCAB}, "single-model checkpoint"),
        ({"model": {}, "vocab": "incorrect"}, "vocabulary mismatch"),
        ({"vocab": VOCAB}, "does not contain model weights"),
    ],
)
def test_invalid_training_snapshots_have_clear_errors(tmp_path, saved, message):
    import torch

    from khojki_ocr.checkpoints import load_single_checkpoint

    path = tmp_path / "bad.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match=message):
        load_single_checkpoint(path)
