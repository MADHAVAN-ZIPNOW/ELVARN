"""A 3D chart picture for the team e-mail: an exploded 3D pie and 3D columns (e-mail clients cannot draw 3D, so it is a PNG)."""
from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, S = 480, 300, 3
BG_TOP, BG_BOTTOM = (30, 27, 75), (76, 29, 149)
FONT_FILE = Path(__file__).resolve().parent / "fonts" / "DejaVuSans-Bold.ttf"      # shipped with the app: the server has no fonts of its own


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_FILE), size * S)
    except OSError:
        return ImageFont.load_default(size=size * S)      # still scalable, never a tiny bitmap font


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _shade(rgb: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(v * f))) for v in rgb)  # type: ignore[return-value]


def _text(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: int, fill, anchor="mm") -> None:
    d.text((xy[0] * S, xy[1] * S), text, font=_font(size), fill=fill, anchor=anchor)


def render(parts: list[tuple[str, int, str]], pct: int) -> bytes:
    """parts = [(label, count, '#rrggbb'), ...] (counts > 0); pct = the share of items that are done. Returns PNG bytes."""
    parts = [p for p in parts if p[1] > 0]
    img = Image.new("RGB", (W * S, H * S))
    d = ImageDraw.Draw(img)
    for y in range(H * S):                                   # background: deep indigo to violet
        t = y / (H * S)
        d.line([(0, y), (W * S, y)], fill=tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)))
    img = img.convert("RGBA")
    if not parts:
        return _png(img)
    total = sum(p[1] for p in parts)

    # ---------------------------------------------------------------- the exploded 3D pie (left)
    cx, cy, R, r, depth = 120.0, 156.0, 76.0, 38.0, 20
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([(cx - R - 6) * S, (cy + depth - r + 10) * S, (cx + R + 14) * S, (cy + depth + r + 16) * S], fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(9 * S)))
    d = ImageDraw.Draw(img)

    slices, a0 = [], -math.pi / 2                            # start at the top, go clockwise
    for label, n, col in parts:
        a1 = a0 + 2 * math.pi * n / total
        slices.append((label, n, _hex(col), a0, a1))
        a0 = a1

    def point(a: float, rr: float, ox: float, oy: float, dz: float) -> tuple[float, float]:
        return ((cx + ox + math.cos(a) * rr) * S, (cy + oy + math.sin(a) * rr * (r / R) + dz) * S)

    def explode(a0_: float, a1_: float, label: str) -> tuple[float, float]:
        if label != parts[0][0] or len(parts) == 1:          # only the first slice (Done) is pulled out of the pie
            return 0.0, 0.0
        mid = (a0_ + a1_) / 2
        return math.cos(mid) * 14, math.sin(mid) * 7

    order = sorted(slices, key=lambda s_: math.sin((s_[3] + s_[4]) / 2))   # far slices first, near slices last (painter's order)
    for label, n, rgb, a0_, a1_ in order:
        ox, oy = explode(a0_, a1_, label)
        steps = max(8, int((a1_ - a0_) / 0.05))
        angles = [a0_ + (a1_ - a0_) * i / steps for i in range(steps + 1)]
        # the curved side wall: only the parts that face the viewer (sin > 0)
        for i in range(steps):
            if math.sin((angles[i] + angles[i + 1]) / 2) <= 0:
                continue
            quad = [point(angles[i], R, ox, oy, 0), point(angles[i + 1], R, ox, oy, 0),
                    point(angles[i + 1], R, ox, oy, depth), point(angles[i], R, ox, oy, depth)]
            shade = 0.50 + 0.18 * math.cos((angles[i] + angles[i + 1]) / 2)
            d.polygon(quad, fill=_shade(rgb, shade))
        for a_, face in ((a0_, 0.62), (a1_, 0.46)):          # the flat cut faces of the slice, seen from the side
            if math.cos(a_) * (1 if a_ == a1_ else -1) > 0.15 or len(slices) == 1:
                d.polygon([point(a_, 0, ox, oy, 0), point(a_, R, ox, oy, 0), point(a_, R, ox, oy, depth), point(a_, 0, ox, oy, depth)],
                          fill=_shade(rgb, face))
        top = [point(0, 0, ox, oy, 0)] + [point(a_, R, ox, oy, 0) for a_ in angles]
        d.polygon(top, fill=_shade(rgb, 1.0), outline=(255, 255, 255), width=S)
        gloss = [point(0, 0, ox, oy, 0)] + [point(a_, R * 0.98, ox, oy, 0) for a_ in angles[: max(2, steps // 2)]]
        gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).polygon(gloss, fill=(255, 255, 255, 36))
        img = Image.alpha_composite(img, gl)
        d = ImageDraw.Draw(img)
    _text(d, (cx, cy - r - 34), f"{pct}% DONE", 18, (255, 255, 255))

    # ---------------------------------------------------------------- the 3D columns (right)
    left, right, floor, top_y = 244.0, 468.0, 262.0, 62.0
    slot = (right - left) / max(1, len(parts))
    bw = min(38.0, slot * 0.6)
    dx, dy = 11.0, -7.0                                      # the depth offset: the columns are seen from the corner
    peak = max(p[1] for p in parts)
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([left * S, (floor - 2) * S, right * S, (floor + 10) * S], fill=(0, 0, 0, 90))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(6 * S)))
    d = ImageDraw.Draw(img)
    d.polygon([(left * S, floor * S), (right * S, floor * S), ((right + dx) * S, (floor + dy) * S), ((left + dx) * S, (floor + dy) * S)], fill=(112, 92, 205))
    for i, (label, n, col) in enumerate(parts):
        rgb = _hex(col)
        x0 = left + slot * i + (slot - bw) / 2
        h = 18 + (floor - top_y - 18) * n / peak
        y0 = floor - h
        d.polygon([((x0 + bw) * S, floor * S), ((x0 + bw + dx) * S, (floor + dy) * S), ((x0 + bw + dx) * S, (y0 + dy) * S), ((x0 + bw) * S, y0 * S)], fill=_shade(rgb, 0.55))   # side
        for k in range(int(h * S)):                                                                                                                                      # front, lit from the top
            t = k / max(1, h * S)
            d.line([(x0 * S, (y0 * S) + k), ((x0 + bw) * S, (y0 * S) + k)], fill=_shade(rgb, 1.18 - 0.38 * t))
        d.polygon([(x0 * S, y0 * S), ((x0 + dx) * S, (y0 + dy) * S), ((x0 + bw + dx) * S, (y0 + dy) * S), ((x0 + bw) * S, y0 * S)], fill=_shade(rgb, 1.38))               # top
        d.line([(x0 * S, y0 * S), ((x0 + bw) * S, y0 * S)], fill=(255, 255, 255, 120), width=S)
        _text(d, (x0 + bw / 2 + dx / 2, y0 + dy - 11), str(n), 22, (255, 255, 255))
        _text(d, (x0 + bw / 2 + dx / 2, floor + 20), label, 10, (225, 218, 255))
    return _png(img)


def _png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.convert("RGB").resize((W, H), Image.LANCZOS).save(out, "PNG", optimize=True)
    return out.getvalue()


if __name__ == "__main__":      # python -m app.chart3d  ->  writes chart3d_sample.png
    data = render([("Done", 2, "#059669"), ("In progress", 1, "#6366f1"), ("Next", 1, "#d97706"), ("Blocked", 1, "#dc2626")], 40)
    open("chart3d_sample.png", "wb").write(data)
    print(len(data) // 1024, "KB")
