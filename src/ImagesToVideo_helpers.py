# ImagesToVideo_helpers.py
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union, cast

# local modules
import definitions as defin
import helpers as he
import VideoEncodersCodecs as vec  # für Geometrie-Probes

_RES_PLAIN_RE = re.compile(r"^\s*(\d+)\s*[:x]\s*(\d+)\s*$", re.I)
_HMS_RE = re.compile(r"^\s*(\d{1,2}:)?\d{1,2}:\d{2}\s*$")
_FPS_TAG_RE = re.compile(r"\s*(fps?|fp|f)\s*$", re.I)
_UNIT_RE = re.compile(r"\s*(ns|ms|s|m|h)\s*$", re.I)
_NUM_RE = re.compile(r"^\s*[+-]?\d+(?:[.,]\d+)?\s*$")
_DUR_WITH_UNIT_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*(ns|ms|s|m|h)\s*$", re.I)


# ----------------------- Natural Sort & Parsing -----------------------


def _natural_sort_fallback(paths: List[str]) -> List[str]:
    ns = getattr(he, "natural_sort", None)
    if callable(ns):
        try:
            res = ns(paths)  # type: ignore[no-untyped-call]
            if isinstance(res, list):
                return [str(x) for x in cast(List[Any], res)]
            if isinstance(res, tuple):
                return [str(x) for x in cast(Tuple[Any], res)]
            return [str(x) for x in cast(Iterable[Any], res)]
        except Exception:
            pass
    import re as _re

    def _key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]

    return sorted(paths, key=_key)


def safe_parse_rational(s: str) -> float:
    s2 = str(s).strip().replace(",", ".")
    parse_fn = getattr(he, "parse_rational", None)
    if callable(parse_fn):
        try:
            res_any = parse_fn(s2)  # type: ignore[no-untyped-call]
            if isinstance(res_any, (int, float, str)):
                return float(res_any)
            return float(str(res_any))
        except Exception:
            pass
    if "/" in s2:
        n_str, d_str = s2.split("/", 1)
        n_f = float(n_str.strip())
        d_f = float(d_str.strip())
        if d_f == 0.0:
            raise ValueError("division by zero in rational")
        return n_f / d_f
    return float(s2)


def _pretty_seconds(secs: float) -> str:
    if secs < 1:
        ms = round(secs * 1000.0, 3)
        s = f"{ms:g}ms"
    else:
        s = f"{secs:.3f}s".rstrip("0").rstrip(".")
    return s


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)\s*$", re.IGNORECASE)


def parse_fps_or_duration(
    user_value: str,
) -> Tuple[str, Optional[str], Optional[float]]:
    if not user_value:
        return ("fps", "25", None)
    m = _DURATION_RE.match(str(user_value))
    if m:
        val, unit = m.groups()
        x = float(val)
        return ("dur", None, x / 1000.0 if unit.lower() == "ms" else x)
    fps_str = str(user_value).strip()
    try:
        _ = safe_parse_rational(fps_str)
        return ("fps", fps_str, None)
    except Exception:
        return ("fps", "25", None)


def parse_resolution_arg_images(val: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Akzeptiert: 'original', 'custom', 'WxH' (z.B. '300x200' oder '300:200').
    Rückgabe: ('custom', 'W:H') bei WxH, sonst (val or 'original', None).
    """
    if not val:
        return ("original", None)
    s = str(val).strip()
    m = _RES_PLAIN_RE.match(s)
    if m:
        w, h = m.group(1), m.group(2)
        return ("custom", f"{w}:{h}")
    return (s, None)


def compute_image_size_stats(files: List[str]) -> Tuple[int, int, int, int, int, int]:
    """
    Liefert (min_w, min_h, avg_w, avg_h, max_w, max_h) über alle Eingabebilder.
    Nutzt ffprobe pro Bild (vec.ffprobe_geometry).
    """
    ws: List[int] = []
    hs: List[int] = []
    for p in files:
        try:
            w, h, _fmt = vec.ffprobe_geometry(Path(p))
            if w and h:
                ws.append(int(w))
                hs.append(int(h))
        except Exception:
            pass
    if not ws or not hs:
        return (0, 0, 0, 0, 0, 0)
    min_w, min_h = min(ws), min(hs)
    max_w, max_h = max(ws), max(hs)
    avg_w = int(round(sum(ws) / len(ws)))
    avg_h = int(round(sum(hs) / len(hs)))
    return (min_w, min_h, avg_w, avg_h, max_w, max_h)


# ----------------------- Inputs & Table -----------------------


def prepare_inputs_images(args: Any) -> Tuple[bool, List[str]]:
    image_exts = tuple(
        getattr(
            defin,
            "IMAGE_EXTENSIONS",
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
        )
    )
    bm_raw, fl_any = he.prepare_inputs(args, image_exts)
    files_list: List[str] = [str(Path(p)) for p in cast(List[Any], fl_any)]
    files_list = _natural_sort_fallback(files_list)
    return bool(bm_raw), files_list


def _resolution_display(
    res_key: str, res_defs: Mapping[str, Any], scale_val: Optional[str]
) -> Union[str, Dict[str, str]]:
    if res_key == "custom":
        return {
            "de": f"Benutzerdefiniert ({scale_val or '—'})",
            "en": f"Custom ({scale_val or '—'})",
        }
    if res_key in ("original", None):
        return {"de": "Original", "en": "Original"}
    spec = res_defs.get(res_key)
    return spec["name"] if spec and "name" in spec else res_key


def target_container(format_choice: Optional[str], files: List[str]) -> str:
    if (format_choice or "mp4") != "keep":
        return str(format_choice or "mp4").lower()
    return "mp4"


def build_params_for_table_images(
    *,
    files: List[str],
    preset_choice: Optional[str],
    format_choice: Optional[str],
    resolution_key: Optional[str],
    resolution_scale: Optional[str],
    codec_choice_by_container: Dict[str, str],
    presets: Mapping[str, Any],
    res_defs: Mapping[str, Any],
    fr_mode: str,
    fps_rational: Optional[str],
    dur_seconds: Optional[float],
    scale_exact: Optional[bool] = None,
    total_duration: Optional[str] = None,
    is_vfr: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tgt_container: str = target_container(format_choice or "mp4", files)

    pc = preset_choice or "casual"
    fc = format_choice or "mp4"

    fmt_spec = defin.CONVERT_FORMAT_DESCRIPTIONS.get(fc)
    format_val: str = fmt_spec["name"] if fmt_spec and "name" in fmt_spec else fc

    preset_spec = presets.get(pc)
    preset_val: str = (
        preset_spec["name"] if preset_spec and "name" in preset_spec else pc
    )

    if resolution_key == "custom":
        scale_val: Optional[str] = resolution_scale
    elif resolution_key in ("original", None):
        scale_val = None
    else:
        res_spec = res_defs.get(resolution_key or "")
        scale_val = res_spec["scale"] if res_spec and "scale" in res_spec else None

    resolution_val = _resolution_display(
        resolution_key or "original", res_defs, scale_val
    )

    if (format_choice or "mp4") != "keep":
        video_codec_val: str = codec_choice_by_container.get(tgt_container, "h264")
    else:
        video_codec_val = (
            ", ".join(f"{c}:{v}" for c, v in codec_choice_by_container.items())
            or "h264"
        )

    # (neu – ersetze den ganzen Block, inkl. der früheren doppelten Zuweisung)
    if fr_mode == "fps" and fps_rational:
        framerate_val: Union[str, Dict[str, str]] = f"{fps_rational} fps"
    elif fr_mode == "dur" and (dur_seconds is not None):
        framerate_val = {
            "de": f"{_pretty_seconds(dur_seconds)}/Bild",
            "en": f"{_pretty_seconds(dur_seconds)}/img",
        }
    elif is_vfr:
        framerate_val = {"de": "VFR (variabel)", "en": "VFR (variable)"}
    else:
        framerate_val = {"de": "25 fps (Standard)", "en": "25 fps (default)"}

    faststart_on = bool(
        (tgt_container == "mp4")
        and (preset_spec is not None and preset_spec.get("faststart", False))
    )

    params: Dict[str, Any] = {
        "files": [str(Path(f)) for f in files],
        "preset": preset_val,
        "format": format_val,
        "video_codec": video_codec_val,
        "resolution": resolution_val,
        "framerate": framerate_val,
        "total_duration": total_duration,
    }
    if scale_val:
        params["scale"] = scale_val
    if faststart_on:
        params["faststart"] = {"de": "an", "en": "on"}

    if (resolution_key not in ("original", None)) and (scale_exact is not None):
        params["resize_mode"] = (
            {"de": "Skalieren (ohne AR)", "en": "Scale (no AR)"}
            if scale_exact
            else {"de": "Crop/Pad (AR erhalten)", "en": "Crop/Pad (keep AR)"}
        )

    labels: Dict[str, Any] = {
        "preset": {"de": "Voreinstellung", "en": "Preset"},
        "format": {"de": "Format", "en": "Format"},
        "video_codec": {"de": "Video-Codec", "en": "Video codec"},
        "resolution": {"de": "Zielauflösung", "en": "Target resolution"},
        "scale": {"de": "Skalierung", "en": "Scale"},
        "framerate": {"de": "Bildrate", "en": "Framerate"},
        "faststart": {"de": "Faststart", "en": "Faststart"},
        "resize_mode": {"de": "Auflösungsanpassung", "en": "Resolution adjustment"},
        "total_duration": {"de": "Gesamtdauer", "en": "Total duration"},
    }
    return params, labels


# ----------------------- Common Dir & ffconcat -----------------------


def get_common_dir(files: List[str]) -> Path:
    try:
        common_dir = Path(os.path.commonpath([str(Path(f).resolve()) for f in files]))
        if common_dir.is_file():
            common_dir = common_dir.parent
    except Exception:
        common_dir = Path(files[0]).resolve().parent
    return common_dir


def _ffconcat_quote(
    path: Union[str, Path], *, absolute: bool = True, base_dir: Optional[Path] = None
) -> str:
    p = Path(path)
    if absolute:
        s = p.resolve().as_posix()
    else:
        root = (base_dir or Path.cwd()).resolve()
        try:
            s = Path(os.path.relpath(p.resolve(), start=root)).as_posix()
        except Exception:
            s = p.resolve().as_posix()
    s = s.replace("\\", "\\\\").replace("'", r"\'")
    return f"'{s}'"


def write_ffconcat_file(
    image_files: List[str],
    dur_each: float,
    *,
    base_dir: Optional[Path] = None,
    filename_prefix: str = "img2vid_",
    use_relative: bool = True,
) -> Path:
    base_dir = base_dir or get_common_dir(image_files)
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=filename_prefix, suffix=".ffconcat", dir=str(base_dir)
        )
        os.close(fd)
    except Exception:
        fd, tmp_path = tempfile.mkstemp(prefix=filename_prefix, suffix=".ffconcat")
        os.close(fd)

    tmp_file = Path(tmp_path)
    with tmp_file.open("w", encoding="utf-8") as fh:
        fh.write("ffconcat version 1.0\n")
        for img in image_files:
            q = _ffconcat_quote(
                img, absolute=not use_relative, base_dir=tmp_file.parent
            )
            if use_relative:
                q = _ffconcat_quote(img, absolute=False, base_dir=tmp_file.parent)
            fh.write(f"file {q}\n")
            fh.write(f"duration {dur_each:.6f}\n")
        q_last = (
            _ffconcat_quote(image_files[-1], absolute=False, base_dir=tmp_file.parent)
            if use_relative
            else _ffconcat_quote(image_files[-1])
        )
        fh.write(f"file {q_last}\n")
    return tmp_file


def inject_concat_demuxer_for_input(
    cmd: List[str], list_path: Union[str, Path]
) -> None:
    lp = str(list_path)
    i = 0
    while i < len(cmd) - 1:
        if cmd[i] == "-i" and cmd[i + 1] == lp:
            if i >= 2 and cmd[i - 2 : i] == ["-f", "concat"]:
                return
            cmd[i:i] = ["-f", "concat", "-safe", "0"]
            return
        i += 1


def insert_output_opts(
    cmd: List[str], out_path: Union[str, Path], opts: List[str]
) -> None:
    """
    Setzt 'opts' unmittelbar VOR die Ausgabedatei ein.
    Fällt bei Nichtauffinden von out_path auf Anhängen zurück (best effort).
    """
    out_str = str(out_path)
    try:
        # Index der letzten Übereinstimmung (normalerweise die Ausgabedatei)
        r_index = cmd[::-1].index(out_str)
        insert_at = len(cmd) - 1 - r_index
    except ValueError:
        # Falls out_path aus unerfindlichen Gründen noch nicht im cmd ist:
        cmd += opts
        return

    for k, tok in enumerate(opts):
        cmd.insert(insert_at + k, tok)


# === Croppad-Helpers für Images->Video (gleiche Achs-Logik wie im croppad-Modul) ===


def parse_wh_from_scale(scale_val: str | None) -> Tuple[int, int]:
    """
    Erwartet 'W:H' oder 'WxH' etc. (robust). Gibt (W,H) als int zurück.
    """
    if not scale_val:
        return (0, 0)
    s = str(scale_val).strip().lower()
    for sep in ["x", "X", ",", ".", ":", "/", "-", "#", "+", "*"]:
        s = s.replace(sep, "x")
    parts = [p.strip() for p in s.split("x") if p.strip()]
    if len(parts) != 2:
        return (0, 0)
    try:
        w = int(parts[0])
        h = int(parts[1])
        return (w, h) if w > 0 and h > 0 else (0, 0)
    except Exception:
        return (0, 0)


def _pixfmt_has_alpha(pf: Optional[str]) -> bool:
    if not pf:
        return False
    p = pf.lower()
    return (
        p.startswith(("rgba", "bgra", "argb", "abgr", "gbrap", "yuva", "ya"))
        or "a" in p  # pragmatischer Fallback
    )


def detect_alpha_from_first(files: List[str]) -> bool:
    """
    Prüft erstes Bild auf Alpha (per ffprobe pix_fmt).
    """
    if not files:
        return False
    try:
        _w, _h, fmt = vec.ffprobe_geometry(Path(files[0]))
        return _pixfmt_has_alpha(fmt)
    except Exception:
        return False


def build_croppad_vf_images(
    *,
    target_w: int,
    target_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    transparent_pad: bool = False,
) -> str:
    """
    Erzeugt eine VF-Chain:
      1) crop: nur wenn Quelle größer als Ziel (achsweise), zentriert + Offset, geclamped
      2) pad : nur wenn Quelle kleiner als Ziel (achsweise), zentriert + Offset
    Kommas in FFmpeg-Expressions werden ge-escaped (,) damit der Parser nicht trennt.
    """
    if target_w <= 0 or target_h <= 0:
        return ""

    tw, th = int(target_w), int(target_h)
    ox, oy = int(offset_x), int(offset_y)

    def esc_commas(s: str) -> str:
        return s.replace(",", r"\,")  # NUR Kommas innerhalb von Expressions escapen

    # --- Crop-Koordinaten: sicher via if(gte(...), clip(...), 0) ---
    # X: nur croppen wenn iw >= tw, sonst 0
    cx_expr = f"if(gte(iw,{tw}),clip(((iw-{tw})/2)+{ox},0,iw-{tw}),0)"
    cy_expr = f"if(gte(ih,{th}),clip(((ih-{th})/2)+{oy},0,ih-{th}),0)"

    # Zielbreiten/-höhen: achsweise min(i*,t*)
    cw_expr = f"min(iw,{tw})"
    ch_expr = f"min(ih,{th})"

    crop = (
        "crop="
        f"{esc_commas(cw_expr)}:"
        f"{esc_commas(ch_expr)}:"
        f"{esc_commas(cx_expr)}:"
        f"{esc_commas(cy_expr)}"
    )

    # --- Pad-Koordinaten: nach Crop gelten iw/ih als post-crop Maße ---
    # zentriert + Offsets, geclamped in gültige Range
    px_expr = f"clip((({tw}-iw)/2)+{ox},0,{tw}-iw)"
    py_expr = f"clip((({th}-ih)/2)+{oy},0,{th}-ih)"
    pad_color = "black@0" if transparent_pad else "black"

    pad = f"pad={tw}:{th}:{esc_commas(px_expr)}:{esc_commas(py_expr)}:{pad_color}"

    return f"{crop},{pad}"


def ensure_merge_vf(cmd: List[str], out_path: Union[str, Path], vf: str) -> None:
    """
    Hängt '-vf <vf>' VOR der Ausgabedatei an oder merged mit bestehender VF-Chain.
    Nutzt bei Bedarf vec._vf_join(), fällt sonst auf einfache Verkettung zurück.
    """
    if not vf:
        return
    out_str = str(out_path)
    # gibt es bereits ein -vf?
    try:
        j = cmd.index("-vf")
        if j + 1 < len(cmd):
            try:
                new_vf = vec.vf_join(cmd[j + 1], vf)
            except Exception:
                new_vf = f"{cmd[j + 1]},{vf}"
            cmd[j + 1] = new_vf
            return
    except ValueError:
        pass
    # sonst: vor out_path einfügen
    insert_output_opts(cmd, out_str, ["-vf", vf])


def parse_total_duration_to_seconds(s: str) -> float:
    """
    Interpretiert eine Gesamtdauer:
      - "HH:MM:SS", "MM:SS"
      - "SS" (plain number → Sekunden als float)
      - mit Einheiten (s, ms, m, h, ns) → entsprechende Sekunden
    """
    if s is None:
        raise ValueError("empty total duration")
    t = str(s).strip()
    if not t:
        raise ValueError("empty total duration")

    low = t.lower().replace(",", ".")
    # Kolonformat → HH:MM:SS / MM:SS
    if ":" in low or _HMS_RE.match(low):
        try:
            # he.to_seconds versteht HH:MM:SS / MM:SS / SS
            return float(he.to_seconds(low))
        except Exception:
            pass

    # Einheit?
    m = _UNIT_RE.search(low)
    if m:
        unit = m.group(1).lower()
        num = float(low[: m.start(1)].strip())
        if unit == "ns":
            return num / 1_000_000_000.0
        if unit == "ms":
            return num / 1_000.0
        if unit == "s":
            return num
        if unit == "m":
            return num * 60.0
        if unit == "h":
            return num * 3600.0

    # Plain number → Sekunden gesamt
    return float(low)


def parse_interactive_fps_dur_total(
    user_raw: str,
) -> Tuple[str, Optional[str], Optional[float], Optional[float]]:
    """
    Rückgabe: (mode, fps_rational, dur_per_frame_sec, total_sec)
      mode ∈ {"fps","dur","total"}

    Regeln (Priorität):
      1) "<zahl>(fps|fp|f)"     → FPS (CFR)
      2) "<zahl><einheit>"      → Zeit/Bild (DUR)  [Einheit ∈ ns, ms, s, m, h]
      3) "hh:mm:ss" / "mm:ss"   → Gesamtdauer (TOTAL)
      4) reine Zahl             → Gesamtdauer (TOTAL)
      5) Bruch (z.B. 24000/1001)→ FPS (CFR)
      6) Fallback               → 25 fps
    """
    s = str(user_raw or "").strip()
    if not s:
        return ("fps", "25", None, None)

    # Normalisieren (Komma → Punkt)
    low = s.lower().replace(",", ".")

    # 1) „fps“-Suffix: z.B. "25fps", "23.976f", "24000/1001 fps"
    if _FPS_TAG_RE.search(low):
        num = _FPS_TAG_RE.sub("", low).strip()
        try:
            _ = safe_parse_rational(num)
            return ("fps", num, None, None)
        except Exception:
            return ("fps", "25", None, None)

    # 2) explizite Einheit → Dauer pro Frame (dur)
    m = _DUR_WITH_UNIT_RE.match(low)
    if m:
        num_str, unit = m.groups()
        try:
            val = float(num_str)
            factor = {"ns": 1e-9, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}[
                unit.lower()
            ]
            return ("dur", None, max(0.0, val * factor), None)
        except Exception:
            # Fallback bei Parsing-Fehler: Standard-FPS
            return ("fps", "25", None, None)

    # 3) „hh:mm:ss“ / „mm:ss“ → Gesamtdauer
    if ":" in low:
        try:
            total = float(he.to_seconds(low))
            return ("total", None, None, max(0.0, total))
        except Exception:
            pass  # weiter prüfen

    # 4) reine Zahl → Gesamtdauer (Sekunden)
    if _NUM_RE.match(low):
        try:
            return ("total", None, None, max(0.0, float(low)))
        except Exception:
            pass

    # 5) Rational ohne Suffix (z.B. "24000/1001") → FPS
    if "/" in low:
        try:
            _ = safe_parse_rational(low)
            return ("fps", low, None, None)
        except Exception:
            pass

    # 6) letzte Chance: als FPS interpretieren, sonst 25
    try:
        _ = safe_parse_rational(low)
        return ("fps", low, None, None)
    except Exception:
        return ("fps", "25", None, None)


def pretty_hms(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds - (h * 3600 + m * 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:06.3f}".rstrip("0").rstrip(".")
    return f"{m:d}:{s:06.3f}".rstrip("0").rstrip(".")
