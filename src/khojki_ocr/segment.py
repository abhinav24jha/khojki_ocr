"""Conservative connected-component and projection segmentation for printed pages."""

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Line:
    id: str
    box: list[int]


def ink_mask(image):
    if image.width < 1 or image.height < 1:
        raise ValueError("Image must have positive dimensions")
    gray = np.asarray(image.convert("L"))
    bg = cv2.GaussianBlur(gray, (0, 0), 9)
    mask = ((bg.astype(float) - gray) > 24).astype("uint8")
    n, lab, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            mask[lab == i] = 0
    return mask


def frame_region(mask):
    """Locate an ornamental print frame from edge density and wide top ornament.

    No page ID or reference transcription is used. Unframed pages fall back to
    ordinary full-page segmentation; the caller must review those results.
    """
    h, w = mask.shape
    # The fixed ornament margins cannot describe a frame on a tiny crop.
    if h < 40 or w < 80:
        return None
    col = gaussian_filter1d(mask.mean(0), 4)
    left = int(0.015 * w) + int(np.argmax(col[int(0.015 * w) : int(0.2 * w)]))
    right = int(0.8 * w) + int(np.argmax(col[int(0.8 * w) : int(0.975 * w)]))
    # Remove side ornaments before measuring horizontal ornaments.
    l, r = left + 15, right - 15
    if r - l < 0.6 * w:
        return None
    joined = cv2.morphologyEx(mask[:, l:r], cv2.MORPH_CLOSE, np.ones((3, 19), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    top = [
        s
        for s in stats[1:]
        if s[2] > 0.94 * (r - l) and s[3] < 0.045 * h and 0.05 * h < s[1] < 0.22 * h
    ]
    if not top:
        return None
    upper = min(top, key=lambda s: s[1])
    y0 = int(upper[1] + upper[3] + 3)
    bottom = [
        s
        for s in stats[1:]
        if s[2] > 0.94 * (r - l) and s[3] < 0.065 * h and s[1] > 0.88 * h
    ]
    if bottom:
        y1 = int(max(bottom, key=lambda s: s[1])[1] - 3)
    else:
        profile = gaussian_filter1d(mask[:, l:r].mean(1), 2)
        peaks, _ = find_peaks(
            profile,
            height=max(0.04, profile.max() * 0.22),
            prominence=0.02,
            distance=12,
        )
        candidates = [y for y in peaks if 0.85 * h < y < 0.97 * h]
        if not candidates:
            return None
        y1 = int(candidates[-1] - 12)
    if y1 - y0 < 0.6 * h:
        return None
    fine = gaussian_filter1d(mask.mean(0), 1.5)
    la, lb = left + 8, min(w, left + max(25, int(0.065 * w)))
    ra, rb = max(0, right - max(25, int(0.065 * w))), right - 8
    if lb <= la or rb <= ra:
        return None
    trim_left = la + int(np.argmin(fine[la:lb]))
    trim_right = ra + int(np.argmin(fine[ra:rb]))
    return [trim_left, y0, trim_right, y1]


def detect_lines(
    image, prefix="line", mode="adaptive", region=None, automatic_frame=True
):
    if mode not in {"adaptive", "robust", "projection"}:
        raise ValueError("Unknown line segmentation mode")
    if region is not None:
        if len(region) != 4 or any(
            not isinstance(v, (int, np.integer)) for v in region
        ):
            raise ValueError("Region must contain four integer coordinates")
        left, top, right, bottom = region
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("Region must be a nonempty rectangle within the image")
    mask = ink_mask(image)
    h, w = mask.shape
    # Connected ornamental frames are distinct from text: retain their interior,
    # but remove their ink before finding baselines. Generic fallback is full page.
    n, lab, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    frames = [
        i for i in range(1, n) if stats[i, 2] > 0.65 * w and stats[i, 3] > 0.6 * h
    ]
    if mode == "robust":
        frames = [
            i
            for i in frames
            if stats[i, 0] < 0.1 * w and stats[i, 0] + stats[i, 2] > 0.9 * w
        ]
    roi = [int(0.04 * w), int(0.03 * h), int(0.96 * w), int(0.97 * h)]
    if frames:
        i = max(frames, key=lambda i: stats[i, 4])
        x, y, ww, hh, _ = stats[i]
        roi = [int(x + 14), int(y + 18), int(x + ww - 14), int(y + hh - 18)]
        for i in frames:
            mask[lab == i] = 0
    if region is not None:
        roi = [int(value) for value in region]
    elif automatic_frame:
        inferred = frame_region(mask)
        if inferred is not None:
            if mode != "robust" or (
                inferred[0] < 0.12 * w
                and inferred[2] > 0.88 * w
                and inferred[1] < 0.2 * h
            ):
                roi = inferred
    if mode == "robust" and region is None and (roi[0] > 0.12 * w or roi[2] < 0.88 * w):
        roi = [int(0.025 * w), int(0.02 * h), int(0.975 * w), int(0.98 * h)]
    # Frame heuristics use fixed margins; clamp their result for small images.
    roi = [max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])]
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        return [], roi
    # Project the remaining body ink onto rows to locate line bodies.
    body = mask[y0:y1, x0:x1]
    profile = gaussian_filter1d(body.mean(1), 1.8)
    if mode == "projection":
        active = profile > 0.045
        for a in range(1, len(active) - 5):
            if active[a - 1] and active[a : a + 5].any():
                active[a : a + np.argmax(active[a : a + 5])] = True
        edges = np.diff(np.r_[0, active.astype(int), 0])
        spans = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    else:
        # Main ink peaks represent line bodies; smaller detached marks stay in
        # the same line's valley-to-valley crop rather than becoming new lines.
        peak_height = (
            max(0.015, profile.max() * 0.08)
            if mode == "robust"
            else max(0.025, profile.max() * 0.17)
        )
        prominence = 0.012 if mode == "robust" else 0.018
        peaks, _ = find_peaks(
            profile,
            height=peak_height,
            distance=max(18, int(h * 0.022)),
            prominence=prominence,
        )
        if not len(peaks):
            return [], roi
        cuts = (
            [0]
            + [
                int(a + np.argmin(profile[a : b + 1]))
                for a, b in zip(peaks[:-1], peaks[1:])
            ]
            + [len(profile)]
        )
        spans = [(cuts[j], cuts[j + 1]) for j in range(len(peaks))]
    lines = []
    for a, b in spans:
        ys, xs = np.where(body[a:b])
        if len(xs) < 35:
            continue
        # Keep only substantial row groups, excluding isolated noise at crop edges.
        rows = np.bincount(ys, minlength=b - a)
        substantial = np.where(rows >= max(4, 0.012 * (x1 - x0)))[0]
        if not len(substantial):
            continue
        top, bottom = int(substantial[0]), int(substantial[-1]) + 1
        if bottom - top < 10:
            continue
        valid = (ys >= top) & (ys < bottom)
        left, right = int(xs[valid].min()), int(xs[valid].max()) + 1
        if right - left < 0.12 * w:
            continue
        box = [
            max(x0, x0 + left - 5),
            max(y0 + a, y0 + a + top - 4),
            min(x1, x0 + right + 5),
            min(y0 + b, y0 + a + bottom + 4),
        ]
        lines.append(Line(f"{prefix}-{len(lines)+1:02d}", box))
    return lines, roi


def overlay(image, lines):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for line in lines:
        draw.rectangle(line.box, outline="red", width=2)
        draw.text(
            (line.box[0], line.box[1]),
            line.id,
            fill="blue",
            stroke_width=1,
            stroke_fill="white",
        )
    return out
