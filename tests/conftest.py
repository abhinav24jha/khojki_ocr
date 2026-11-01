"""Small generated scans keep the default suite independent of private material."""

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def printed_page():
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    for y in (60, 140, 220):
        for x in range(60, 430, 22):
            draw.rectangle((x, y, x + 12, y + 19), fill="black")
    return image


@pytest.fixture
def word_strip():
    image = Image.new("RGB", (180, 48), "white")
    draw = ImageDraw.Draw(image)
    for left, right in ((10, 42), (85, 123)):
        draw.rectangle((left, 15, right, 33), fill="black")
    for x in (60, 145):
        for y in (18, 30):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")
    return image, "\U00011208\U0001123a\U00011219\U0001123a"
