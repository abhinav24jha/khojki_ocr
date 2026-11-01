import json
from html.parser import HTMLParser

import pytest
from PIL import Image
from pypdf import PdfWriter

from khojki_ocr.pipeline import page_images, process_page, write_review
from khojki_ocr.unicode import inspect_text


def test_blank_page_writes_empty_artifacts_without_recognition(tmp_path):
    def unexpected_recognition(image):
        pytest.fail("A blank page must not call the recognizer")

    report = process_page(
        Image.new("RGB", (600, 800), "white"), unexpected_recognition, tmp_path
    )
    assert report["line_count"] == 0
    assert (tmp_path / "transcription.txt").read_text() == ""
    assert json.loads((tmp_path / "result.json").read_text())["lines"] == []
    assert (tmp_path / "review.html").is_file()
    assert (tmp_path / "overlay.png").is_file()


def test_page_exports_raw_utf8_and_auditable_line_coordinates(printed_page, tmp_path):
    raw = "\U00011208\U0001122c\U0001123a૧"
    report = process_page(printed_page, lambda image: raw, tmp_path, prefix="scan")
    assert report["line_count"] == 3
    assert (tmp_path / "transcription.txt").read_text(encoding="utf-8") == (
        raw + "\n"
    ) * 3
    saved = json.loads((tmp_path / "result.json").read_text())
    for row in saved["lines"]:
        assert row["text"] == raw
        assert row["review_required"] is True
        left, top, right, bottom = row["box"]
        with Image.open(tmp_path / row["crop"]) as crop:
            assert crop.size == (right - left, bottom - top)
    assert saved["recognition_seconds"] >= 0


def test_review_escapes_untrusted_text_and_attributes(tmp_path):
    text = '<script>\U00011208</script> & "value" </textarea>'
    identifier = 'line" onfocus="bad'
    report = {
        "lines": [
            {
                "id": identifier,
                "crop": 'scan" onerror="bad.png',
                "text": text,
                "validation": inspect_text(text),
            }
        ]
    }
    write_review(report, tmp_path)
    document = (tmp_path / "review.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in document
    assert "<script>\U00011208" not in document
    assert "reviewed-transcriptions.json" in document

    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append((tag, dict(attrs)))

    parser = Elements()
    parser.feed(document)
    assert sum(tag == "script" for tag, _ in parser.tags) == 1
    assert not any("onerror" in attrs or "onfocus" in attrs for _, attrs in parser.tags)
    assert (
        next(attrs["data-id"] for tag, attrs in parser.tags if tag == "textarea")
        == identifier
    )


@pytest.fixture
def image_pdf(tmp_path):
    path = tmp_path / "two-pages.pdf"
    first = Image.new("RGB", (200, 300), "white")
    first.save(
        path,
        "PDF",
        resolution=72,
        save_all=True,
        append_images=[Image.new("RGB", (240, 320), "white")],
    )
    return path


@pytest.mark.parametrize("native", [False, True])
def test_pdf_page_selection_is_one_based_and_preserves_requested_order(
    image_pdf, native
):
    pages = list(page_images(image_pdf, [2, 1], native=native))
    assert [number for number, _ in pages] == [2, 1]
    if native:
        assert [image.size for _, image in pages] == [(240, 320), (200, 300)]
    else:
        assert pages[0][1].width > 240


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("number", [0, -1, 3])
def test_pdf_out_of_range_pages_are_rejected(image_pdf, native, number):
    with pytest.raises(ValueError, match="outside PDF|positive one-based"):
        list(page_images(image_pdf, [number], native=native))


def test_native_mode_rejects_vector_only_page(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=300)
    source = tmp_path / "vector.pdf"
    writer.write(source)
    with pytest.raises(ValueError, match="exactly one embedded image"):
        list(page_images(source, native=True))


def test_image_loader_applies_exif_orientation(tmp_path):
    image = Image.new("RGB", (20, 30), "white")
    exif = image.getexif()
    exif[274] = 6
    source = tmp_path / "rotated.jpg"
    image.save(source, exif=exif)
    [(number, loaded)] = list(page_images(source))
    assert number == 1
    assert loaded.mode == "RGB"
    assert loaded.size == (30, 20)


@pytest.mark.parametrize("pages", [[], [True], [1.5]])
def test_invalid_page_selections_fail_before_opening_source(tmp_path, pages):
    with pytest.raises(ValueError, match="positive one-based"):
        list(page_images(tmp_path / "unused.pdf", pages))


def test_image_page_selection_rejects_nonexistent_page(tmp_path):
    with pytest.raises(ValueError, match="only page 1"):
        list(page_images(tmp_path / "unused.png", [2]))


@pytest.mark.parametrize(
    "prefix", ["", "..", "../escape", "nested/name", "nested\\name"]
)
def test_prefix_cannot_escape_output_directory(tmp_path, prefix):
    with pytest.raises(ValueError, match="simple filename"):
        process_page(
            Image.new("L", (20, 20), 255), lambda image: "", tmp_path, prefix=prefix
        )


@pytest.mark.parametrize(
    "size,expected", [((200, 300), RuntimeError), ((10000, 10000), ValueError)]
)
def test_pdf_cleanup_on_render_failure_and_size_guard(
    monkeypatch, tmp_path, size, expected
):
    import sys
    from types import SimpleNamespace

    state = {"page_closed": False, "document_closed": False, "rendered": False}

    class Page:
        def get_size(self):
            return size

        def render(self, **kwargs):
            state["rendered"] = True
            raise RuntimeError("Simulated rasterizer failure")

        def close(self):
            state["page_closed"] = True

    class Document:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return Page()

        def close(self):
            state["document_closed"] = True

    monkeypatch.setitem(
        sys.modules,
        "pypdfium2",
        SimpleNamespace(PdfDocument=lambda source: Document()),
    )
    with pytest.raises(expected):
        list(page_images(tmp_path / "mock.pdf"))
    assert state["page_closed"] and state["document_closed"]
    assert state["rendered"] == (expected is RuntimeError)
