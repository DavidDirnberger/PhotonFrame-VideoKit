#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

import helpers as he

# --- Resampling: neue (Resampling.*) + Fallback auf alte (Image.BICUBIC) ---
try:
    from PIL.Image import Resampling

    RESAMPLE_BICUBIC: Any = Resampling.BICUBIC
except Exception:
    # Pillow < 9.1: BICUBIC hängt direkt an PIL.Image
    RESAMPLE_BICUBIC = Image.BICUBIC  # type: ignore[attr-defined]

from ffmpeg_perf import autotune_final_cmd

# ---------- ffmpeg helpers ----------


def extract_frame(
    path: Path,
    filter_chain: Optional[str],
    out_path: Path,
    pos: float = 0.5,
    duration: Optional[float] = None,
    *,
    precise: bool = True,
) -> None:
    """
    Extrahiert ein einzelnes Frame an Position pos∈[0..1].

    - Standard: präzise (langsamere Seek), -ss NACH -i, mit autotune_final_cmd
    - Für Previews: precise=False → schneller Seek, -ss VOR -i, ohne autotune_final_cmd

    duration:
        Optional bereits bekannte Dauer (in Sekunden), um ffprobe-Aufrufe zu sparen.
    """
    # Dauer nur ermitteln, wenn nicht von außen übergeben
    dur = (
        float(duration)
        if duration is not None
        else (he.get_duration_seconds(path) or 0.0)
    )
    if dur <= 0:
        raise subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg (invalid duration)"
        )

    # Timestamp clampen
    ts = max(0.0, min(float(pos) * dur, max(dur - 0.001, 0.0)))

    scale = "scale=min(640\\,iw):-2"
    vf = scale if not filter_chain else f"{filter_chain},{scale}"

    # Basis-Command (ohne Tuning)
    base_cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
    ]

    if precise:
        # präzise Seek-Variante (dekodiert ab Start, dafür exakter)
        base_cmd += [
            "-i",
            str(path),
            "-ss",
            f"{ts:.3f}",
        ]
    else:
        # schnelle Seek-Variante: -ss VOR -i (für Previews völlig ausreichend)
        base_cmd += [
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(path),
        ]

    base_cmd += [
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-q:v",
        "2",
        str(out_path),
    ]

    # Nur für "echte" präzise Operationen das große Tuning anschmeißen
    cmd = autotune_final_cmd(path, base_cmd) if precise else base_cmd

    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )


def grab_frame_at(
    src: Path, time_sec: float, out_path: Path, scale_w: int = 640
) -> bool:
    """Greift robust EIN Frame bei absoluter Zeit (präzise, -ss nach -i)."""
    dur = he.get_duration_seconds(src) or 0.0
    ts = max(0.0, min(float(time_sec), max(dur - 1e-3, 0.0))) if dur > 0 else 0.0
    vf = f"scale=min({scale_w}\\,iw):-2"

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ss",
        f"{ts:.3f}",
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-q:v",
        "2",
        str(out_path),
    ]
    ok = (
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )
    return ok and out_path.exists() and out_path.stat().st_size > 0


# ---------- Pillow helpers ----------


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Robuster Font-Loader mit Fallbacks."""
    for fname in ("impact.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(fname, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    stroke: int,
) -> tuple[float, float]:
    """
    Misst Textbreite/-höhe bevorzugt via draw.textbbox (inkl. stroke_width).
    Fallback: font.getbbox(text) + stroke-Zugaben.
    Gibt floats zurück (Pillow-Stubs erlauben float).
    """
    try:
        # Neuere Pillow-Versionen: exakte bbox inkl. Stroke
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        return (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    except TypeError:
        # Ältere Version: ohne stroke_width; Stroke manuell addieren
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0]) + 2 * stroke, (bbox[3] - bbox[1]) + 2 * stroke
        except Exception:
            pass

    # Letzter Fallback: Font-API
    try:
        fb = font.getbbox(text)  # (l,t,r,b)
        return (fb[2] - fb[0]) + 2 * stroke, (fb[3] - fb[1]) + 2 * stroke
    except Exception:
        # Minimalwerte, falls alles fehlschlägt
        return float(10 + 2 * stroke), float(10 + 2 * stroke)


def make_labeled(img_path: Path | str, label: str, out_path: Path | str) -> None:
    """
    Fügt ein Label mittig unten ein. Schriftgröße wird per binärer Suche maximiert.
    Keine Nutzung von draw.textsize (Pylance-freundlich).
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    text = str(label)

    # Ränder
    pad_x = max(8, int(w * 0.04))
    pad_y = max(8, int(h * 0.04))
    max_text_width = max(10, w - 2 * pad_x)

    start_size = max(8, int(h * 0.07))
    min_size = max(8, int(h * 0.02))

    def measure_at(
        px: int,
    ) -> tuple[int, int, ImageFont.ImageFont | ImageFont.FreeTypeFont, int]:
        font = _load_font(px)
        stroke = max(1, int(round(px * 0.07)))  # ~7%
        twf, thf = _measure_text(draw, text, font, stroke)
        tw, th = int(round(twf)), int(round(thf))  # → ints für Layout
        return tw, th, font, stroke

    # Binäre Suche
    lo, hi = min_size, start_size
    best: tuple[int, int, ImageFont.ImageFont | ImageFont.FreeTypeFont, int] | None = (
        None
    )
    while lo <= hi:
        mid = (lo + hi) // 2
        tw, th, font, stroke = measure_at(mid)
        if tw <= max_text_width:
            best = (tw, th, font, stroke)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        tw, th, font, stroke = measure_at(min_size)
    else:
        tw, th, font, stroke = best

    x = (w - tw) // 2
    y = h - th - pad_y

    # Zeichnen
    try:
        draw.text(
            (x, y),
            text,
            font=font,
            fill="white",
            stroke_width=stroke,
            stroke_fill="black",
        )
    except TypeError:
        # Fallback: manuelle Outline
        r = stroke
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")

    img.save(out_path)


def montage(img1: Path | str, img2: Path | str, out_path: Path | str) -> None:
    """Setzt zwei Bilder nebeneinander; Höhe wird angeglichen (ohne Verzerren)."""
    im1 = Image.open(img1).convert("RGB")
    im2 = Image.open(img2).convert("RGB")

    # Zielhöhe = max Höhe; Breiten proportional skalieren
    target_h = max(im1.height, im2.height)

    def scale_to_h(im: Image.Image, h: int) -> Image.Image:
        if im.height == h:
            return im
        w = max(1, int(round(im.width * (h / im.height))))
        return im.resize((w, h), resample=RESAMPLE_BICUBIC)

    im1 = scale_to_h(im1, target_h)
    im2 = scale_to_h(im2, target_h)

    out = Image.new("RGB", (im1.width + im2.width, target_h))
    out.paste(im1, (0, 0))
    out.paste(im2, (im1.width, 0))
    out.save(out_path)
