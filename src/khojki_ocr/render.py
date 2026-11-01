"""Shape Unicode with HarfBuzz and rasterize the actual glyph IDs with FreeType."""

from pathlib import Path
import freetype
import uharfbuzz as hb
import numpy as np
from PIL import Image


class Renderer:
    def __init__(self, font_path, size=48):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("Font size must be a positive integer")
        self.path = str(font_path)
        self.face = freetype.Face(self.path)
        self.face.set_pixel_sizes(0, size)
        self.font = hb.Font(hb.Face(Path(font_path).read_bytes()))
        self.font.scale = (size * 64, size * 64)
        hb.ot_font_set_funcs(self.font)

    def render(self, text, pad=4):
        if not isinstance(pad, int) or isinstance(pad, bool) or pad < 0:
            raise ValueError("Rendering padding must be a nonnegative integer")
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf)
        pen_x = pen_y = 0
        pieces = []
        for info, pos in zip(buf.glyph_infos or [], buf.glyph_positions or []):
            if info.codepoint == 0:
                raise ValueError(f"Font lacks a shaped glyph for {text!r}")
            self.face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            slot = self.face.glyph
            b = slot.bitmap
            if b.width and b.rows:
                a = np.asarray(b.buffer, dtype=np.uint8).reshape(b.rows, abs(b.pitch))[
                    :, : b.width
                ]
                x = round((pen_x + pos.x_offset) / 64) + slot.bitmap_left
                y = -round((pen_y + pos.y_offset) / 64) - slot.bitmap_top
                pieces.append((x, y, a.copy()))
            pen_x += pos.x_advance
            pen_y += pos.y_advance
        if not pieces:
            return Image.new("L", (max(1, round(pen_x / 64)), 16), 255)
        x0 = min(p[0] for p in pieces)
        y0 = min(p[1] for p in pieces)
        x1 = max(x + a.shape[1] for x, y, a in pieces)
        y1 = max(y + a.shape[0] for x, y, a in pieces)
        canvas = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), dtype=np.uint8)
        for x, y, a in pieces:
            xx, yy = x - x0 + pad, y - y0 + pad
            canvas[yy : yy + a.shape[0], xx : xx + a.shape[1]] = np.maximum(
                canvas[yy : yy + a.shape[0], xx : xx + a.shape[1]], a
            )
        return Image.fromarray(255 - canvas)
