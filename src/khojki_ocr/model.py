"""Compact convolutional/recurrent CTC model and its checkpoint-compatible inputs."""

import numpy as np
from PIL import Image
import torch
from torch import nn

from .vocabulary import VOCAB


class CRNN(nn.Module):
    """Map N×1×H×W images to T×N×C log probabilities (T = W // 4).

    Keep the nested module layout stable: released state dictionaries use these
    layer names. Packed recurrence excludes batch padding from the LSTM.
    """

    def __init__(self, nclass=len(VOCAB) + 1, packed=False):
        super().__init__()
        self.packed = packed

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.cnn = nn.Sequential(
            block(1, 32),
            nn.MaxPool2d(2),
            block(32, 64),
            nn.MaxPool2d(2),
            block(64, 128),
            nn.MaxPool2d((2, 1)),
            block(128, 192),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(
            192, 128, 2, bidirectional=True, batch_first=True, dropout=0.1
        )
        self.head = nn.Linear(256, nclass)

    def forward(self, x, lengths=None):
        x = self.cnn(x).mean(2).transpose(1, 2)
        if self.packed:
            if lengths is None:
                raise ValueError("Packed recurrence requires valid sequence lengths")
            sequence = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            sequence, _ = self.rnn(sequence)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                sequence, batch_first=True, total_length=x.shape[1]
            )
        else:
            x, _ = self.rnn(x)
        return self.head(x).log_softmax(-1).transpose(0, 1)


def image_tensor(image, height=48):
    """Preserve aspect ratio and return a float32 1×H×W tensor with white = 0."""
    if not isinstance(height, int) or isinstance(height, bool) or height < 16:
        raise ValueError("Image height must be an integer of at least 16 pixels")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image must have positive dimensions")
    image = image.convert("L")
    w = max(8, round(image.width * height / image.height))
    a = (
        np.asarray(image.resize((w, height), Image.Resampling.BILINEAR)).astype(
            "float32"
        )
        / 255
    )
    return torch.from_numpy((1 - a)[None])


def collate(samples):
    """Pad images with white and encode labels for CTC training or inference.

    The extra 16 pixels are historical checkpoint input context. Valid sequence
    lengths exclude both that margin and padding introduced by other samples.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    xs, ys = zip(*samples)
    if any(x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] < 4 for x in xs):
        raise ValueError(
            "Each sample must have shape (1, height, width), with width at least 4"
        )
    width = (max(x.shape[-1] for x in xs) + 3) // 4 * 4 + 16
    height = xs[0].shape[-2]
    if any(x.shape[-2] != height for x in xs):
        raise ValueError("Mixed image heights in a batch")
    batch = torch.zeros(len(xs), 1, height, width)
    for i, x in enumerate(xs):
        batch[i, :, :, : x.shape[-1]] = x
    mapping = {c: i + 1 for i, c in enumerate(VOCAB)}
    unsupported = {c for text in ys for c in text if c not in mapping}
    if unsupported:
        names = ", ".join(f"U+{ord(c):04X}" for c in sorted(unsupported))
        raise ValueError(f"Training labels contain unsupported codepoints: {names}")
    labels = torch.tensor([mapping[c] for s in ys for c in s], dtype=torch.long)
    return (
        batch,
        labels,
        torch.tensor([x.shape[-1] // 4 for x in xs]),
        torch.tensor([len(y) for y in ys]),
        ys,
    )


def decode(log_probs, lengths=None):
    """Greedy CTC collapse; a blank separates otherwise repeated characters."""
    ids = log_probs.argmax(-1).transpose(0, 1).cpu().tolist()
    out = []
    for i, row in enumerate(ids):
        if lengths is not None:
            row = row[: int(lengths[i])]
        prev = 0
        text = ""
        for v in row:
            if v and v != prev:
                text += VOCAB[v - 1]
            prev = v
        out.append(text)
    return out
