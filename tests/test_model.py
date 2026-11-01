"""CPU tensor/CTC contracts; run on CI, not the memory-constrained laptop."""

import pytest
from PIL import Image

from khojki_ocr.vocabulary import VOCAB

pytestmark = pytest.mark.integration


@pytest.fixture
def torch_api():
    import torch

    torch.set_num_threads(2)
    return torch


def test_greedy_ctc_repeated_letter_requires_blank(torch_api):
    from khojki_ocr.model import decode

    logits = torch_api.full((4, 1, len(VOCAB) + 1), -20.0)
    for timestep, token in enumerate([1, 1, 0, 1]):
        logits[timestep, 0, token] = 0
    assert decode(logits) == [VOCAB[0] * 2]
    assert decode(logits, torch_api.tensor([2])) == [VOCAB[0]]


def test_image_tensor_maps_white_to_zero_and_preserves_aspect_ratio(torch_api):
    from khojki_ocr.model import image_tensor

    white = image_tensor(Image.new("RGB", (80, 40), "white"))
    black = image_tensor(Image.new("L", (80, 40), 0))
    assert tuple(white.shape) == (1, 48, 96)
    assert white.dtype == torch_api.float32
    assert torch_api.count_nonzero(white) == 0
    assert torch_api.all(black == 1)


@pytest.mark.parametrize("height", [0, 15, 48.5, True])
def test_image_tensor_rejects_invalid_height(torch_api, height):
    from khojki_ocr.model import image_tensor

    with pytest.raises(ValueError, match="height"):
        image_tensor(Image.new("L", (20, 20)), height=height)


def test_collate_pads_white_and_uses_valid_unpadded_lengths(torch_api):
    from khojki_ocr.model import collate

    short = torch_api.ones((1, 48, 17))
    long = torch_api.ones((1, 48, 29))
    batch, labels, lengths, label_lengths, texts = collate(
        [
            (short, VOCAB[0] * 2),
            (long, VOCAB[1]),
        ]
    )
    assert tuple(batch.shape) == (2, 1, 48, 48)
    assert lengths.tolist() == [4, 7]
    assert label_lengths.tolist() == [2, 1]
    assert labels.tolist() == [1, 1, 2]
    assert texts == (VOCAB[0] * 2, VOCAB[1])
    assert not batch[0, :, :, 17:].any()
    assert not batch[1, :, :, 29:].any()


def test_collate_rejects_empty_mixed_height_and_unknown_labels(torch_api):
    from khojki_ocr.model import collate

    with pytest.raises(ValueError, match="empty"):
        collate([])
    with pytest.raises(ValueError, match="Mixed image heights"):
        collate([(torch_api.zeros(1, 48, 20), ""), (torch_api.zeros(1, 32, 20), "")])
    with pytest.raises(ValueError, match="unsupported codepoints"):
        collate([(torch_api.zeros(1, 48, 20), "😀")])


def test_packed_recurrence_ignores_extra_batch_padding(torch_api):
    from khojki_ocr.model import CRNN

    torch_api.manual_seed(9)
    model = CRNN(packed=True).eval()
    short = torch_api.zeros(1, 1, 48, 80)
    short[:, :, :, :32] = torch_api.rand(1, 1, 48, 32)
    wider = torch_api.zeros(1, 1, 48, 128)
    wider[:, :, :, :80] = short
    with torch_api.no_grad():
        first = model(short, torch_api.tensor([8]))[:8]
        padded = model(wider, torch_api.tensor([8]))[:8]
    assert torch_api.allclose(first, padded, atol=1e-6)
    with pytest.raises(ValueError, match="requires valid sequence lengths"):
        model(short)


def test_one_cpu_ctc_backward_step_has_finite_nonzero_gradients(torch_api):
    from khojki_ocr.model import CRNN, collate

    torch_api.manual_seed(17)
    model = CRNN(packed=True)
    batch, labels, lengths, target_lengths, _ = collate(
        [
            (torch_api.rand(1, 48, 48), VOCAB[0] * 2),
            (torch_api.rand(1, 48, 64), VOCAB[1]),
        ]
    )
    probabilities = model(batch, lengths)
    assert probabilities.shape == (20, 2, len(VOCAB) + 1)
    loss = torch_api.nn.CTCLoss()(probabilities, labels, lengths, target_lengths)
    assert torch_api.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(
        gradient is not None and torch_api.isfinite(gradient).all()
        for gradient in gradients
    )
    assert model.head.weight.grad.abs().sum() > 0
