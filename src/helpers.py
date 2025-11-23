#!/usr/bin/env python3
# helpers.py
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
    cast,
    overload,
)

import consoleOutput as co
import definitions as defin
import fileSystem as fs
import mem_guard as mg

# local modules
from i18n import _
from loghandler import print_log

StrPath = Union[str, os.PathLike, Path]

SCRIPT_DIR = Path(__file__).resolve().parent
ANSI_RE = getattr(defin, "ANSI_REGEX", re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]"))

# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────


def str2bool(v: Union[str, bool, None]) -> Optional[bool]:
    """Wandelt typische Bool-Strings in True/False; sonst None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None  # unbekannt


def parse_time(t: str) -> timedelta:
    """Parst eine Zeitangabe wie HH:MM:SS, MM:SS oder SS zu timedelta."""
    parts = t.strip().split(":")
    ints = [int(p) for p in parts if p.isdigit()]
    if len(ints) == 3:
        return timedelta(hours=ints[0], minutes=ints[1], seconds=ints[2])
    if len(ints) == 2:
        return timedelta(minutes=ints[0], seconds=ints[1])
    if len(ints) == 1:
        return timedelta(seconds=ints[0])
    raise ValueError("Ungültiges Zeitformat")


def _probe_stream_json(path: str, entries: str) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                entries,
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def _probe_fmt_json(path: str, entries: str) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", path],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def probe_wh_fmt_fps(path: str) -> Tuple[int, int, Optional[str], float]:
    """width, height, pix_fmt, fps (float, 0.0 wenn unbekannt)"""
    data = _probe_stream_json(path, "stream=width,height,pix_fmt,avg_frame_rate")
    st = (data.get("streams") or [{}])[0]
    w = int(st.get("width") or 0) or 1
    h = int(st.get("height") or 0) or 1
    pf = st.get("pix_fmt") or None
    r = st.get("avg_frame_rate") or "0/0"
    num, den = r.split("/") if "/" in r else (r, "1")
    fps = float(num) / float(den) if float(den) != 0 else 0.0
    return w, h, pf, fps


def probe_duration(path: str) -> float:
    """Sekunden (float) – robust: erst Format-, dann Streamdauer."""
    d = 0.0
    try:
        data = _probe_fmt_json(path, "format=duration")
        d = float((data.get("format") or {}).get("duration") or 0.0)
    except Exception:
        d = 0.0
    if d > 0.0:
        return d
    try:
        data = _probe_stream_json(path, "stream=duration")
        st = (data.get("streams") or [{}])[0]
        d = float(st.get("duration") or 0.0)
    except Exception:
        d = 0.0
    return max(0.0, d)


def probe_has_audio(path: str) -> bool:
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(p.stdout.strip())
    except Exception:
        return False


def probe_wh(path: str) -> Tuple[int, int]:
    w, h, _, _ = probe_wh_fmt_fps(path)
    return w, h


# ──────────────────────────────────────────────────────────────────────────────
# Zeit-Parsing (flexibel): h, m, s, ms, ns + HH:MM:SS / MM:SS / SS
# Beispiele: "90", "2.5s", "1500ms", "1h", "1h30m", "2m 200ms", "-3.5", "00:01:02.500"
# ──────────────────────────────────────────────────────────────────────────────
def flex_time_seconds(val: Optional[str]) -> float:
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0

    s = s.replace(",", ".")  # Dezimalkommas erlauben
    # Fast-Path: kolon-basierte Zeit → helpers nutzt das schon robust
    if ":" in s:
        try:
            return to_seconds(s)
        except Exception:
            pass

    # Komposit-Parser für 1h30m200ms etc.
    sign = -1.0 if s.startswith("-") else 1.0
    core = s[1:].strip() if s[:1] in "+-" else s

    total = 0.0
    pos = 0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(h|ms|m|s|ns)?", core, flags=re.I):
        num = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "h":
            mult = 3600.0
        elif unit == "m":
            mult = 60.0
        elif unit == "s":
            mult = 1.0
        elif unit == "ms":
            mult = 0.001
        elif unit == "ns":
            mult = 1e-9
        else:
            mult = 1.0
        total += num * mult
        pos = m.end()

    # Wenn nur Zahlen/Einheiten konsumiert wurden → ok
    if total > 0.0 and core[pos:].strip() == "":
        return sign * total

    # Fallback: nackte Zahl als Sekunden
    try:
        return sign * float(core)
    except Exception:
        return 0.0


def seconds_to_time(t: timedelta) -> str:
    total = int(t.total_seconds())
    return f"{total // 3600:02}:{(total % 3600) // 60:02}:{total % 60:02}"


def format_time(seconds: float) -> str:
    """Formatiert Sekunden zu h:mm:ss oder m:ss (ohne führende 00:)."""
    s_int = int(round(seconds))
    h, rem = divmod(s_int, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}" if h > 0 else f"{m}:{s:02}"


def get_duration_seconds(path: Path) -> float:
    """Ermittelt die Dauer via ffprobe; zwecks Robustheit min. 2 Sekunden Fallback."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ]
        )
        dur = float(out.decode().strip())
        if dur > 2.0:
            return dur
        raise RuntimeError("Duration < 2s")
    except Exception:
        return 2.0


def _to_seconds_via_existing(t: str) -> float:
    """Nutzt parse_time; akzeptiert timedelta oder numerische Rückgaben."""
    td = parse_time(t)
    return td.total_seconds()


def parse_time_relaxed_seconds(t: str) -> float:
    """
    Toleranter Parser nur für UI/CLI-Eingaben in trim:
      - akzeptiert '+1:00', 'd:1:00', 'dur:1:00', 'duration:1:00', 'end:2:30', '@2:30'
      - Whitespace tolerant, ',' als Dezimaltrenner
      - 'HH:MM:SS' | 'MM:SS' | 'SS' mit optionalen Dezimal-Sekunden
    """
    if t is None:
        raise ValueError("missing time")

    s = t.strip()
    if not s:
        raise ValueError("empty time")

    low = s.lower().lstrip()
    for pref in ("d:", "dur:", "duration:", "end:", "@"):
        if low.startswith(pref):
            s = s[len(pref) :].strip()
            break
    if s.startswith("+"):
        s = s[1:].strip()

    s = s.replace(",", ".")

    try:
        return _to_seconds_via_existing(s)
    except Exception:
        pass

    parts = [p.strip() for p in s.split(":")]
    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
        elif len(parts) == 2:
            h = 0
            m = int(parts[0])
            sec = float(parts[1])
        elif len(parts) == 1:
            h = 0
            m = 0
            sec = float(parts[0])
        else:
            raise ValueError
    except Exception:
        raise ValueError("invalid time format")

    if h < 0 or m < 0 or sec < 0:
        raise ValueError("negative time not allowed")

    return h * 3600 + m * 60 + sec


def to_seconds(value: Any) -> float:
    """
    Robust: akzeptiert str ('HH:MM:SS(.ms)' / 'MM:SS' / 'SS(.ms)'),
    float/int, datetime.timedelta oder Objekte mit total_seconds()
    → float Sekunden. Führendes +/- gilt für die gesamte Zeit.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, timedelta):
        return value.total_seconds()

    ts_attr = getattr(value, "total_seconds", None)
    if callable(ts_attr):
        ts_callable = cast(Callable[[], Union[SupportsFloat, int, float, str]], ts_attr)
        return float(ts_callable())

    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        # führendes Vorzeichen abspalten und merken
        sign = -1.0 if s.startswith("-") else 1.0
        s_core = s[1:].lstrip() if s[:1] in "+-" else s

        try:
            parsed = parse_time(s_core)
            return sign * to_seconds(parsed)
        except Exception:
            pass

        if ":" in s_core:
            parts = [p for p in s_core.split(":")]
            parts_f = [float(p) for p in parts]
            if len(parts_f) == 3:
                h, m, sec = parts_f
                return sign * (h * 3600 + m * 60 + sec)
            if len(parts_f) == 2:
                m, sec = parts_f
                return sign * (m * 60 + sec)

        # keine Kolon-Zeit → float übernimmt das Vorzeichen selbst
        return float(s)

    # Letzter Versuch: über String-Repräsentation konvertieren
    return float(str(value))


def _parse_ts_to_sec(ts: str) -> Optional[float]:
    """Parst 'HH:MM:SS.mmm' oder 'MM:SS.mmm' (SRT ',' oder VTT '.') → Sekunden."""
    ts = ts.strip().replace(",", ".")
    m = re.match(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?$", ts)
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    ms = int((m.group(4) or "0").ljust(3, "0"))
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def probe_subtitle_duration(path: str) -> float:
    """
    Bestimmt die Endzeit externer Untertitel (SRT/VTT) robust.
    Fallback auf ffprobe, falls nichts gefunden wird.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            last_end = 0.0
            for line in f:
                if "-->" not in line:
                    continue
                # Nimm das Ende rechts von '-->'
                m = re.search(r"-->\s*([0-9:.,]+)", line)
                if not m:
                    continue
                end_s = _parse_ts_to_sec(m.group(1))
                if end_s is not None and end_s > last_end:
                    last_end = end_s
            if last_end > 0.0:
                return last_end
    except Exception:
        pass
    # Fallback
    d = probe_duration(path)
    return max(0.0, d)


def percent_int_or_none(x: Any) -> Optional[int]:
    """Wandelt diverse Eingaben in einen ganzzahligen %-Wert (0–100) um.
    Akzeptiert int/float/str inkl. '55%', '0,8'; liefert None bei Unklarheit."""
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            v = float(x)
        else:
            s = str(x).strip().replace(",", ".").rstrip("%")
            v = float(s)
        v = max(0.0, min(100.0, v))
        return int(round(v))
    except Exception:
        return None


def time_or_percent_to_seconds(raw: str, total_sec: float) -> float:
    """
    Interpretiert *raw* als Zeit ODER Prozent:
      - 'HH:MM:SS', 'MM:SS', 'SS(.ms)' → Sekunden (he.parse_time_relaxed_seconds)
      - 'N%', 'Np', 'NP', 'N P' → Prozent von total_sec (0..100)
    """
    if raw is None:
        raise ValueError("missing time")

    s = raw.strip()
    if not s:
        raise ValueError("empty time")

    tail = s[-1:]
    if tail in ("%", "p", "P"):
        # 'p'/'P' als Prozent behandeln → in '%' normalisieren
        if tail in ("p", "P"):
            s = s[:-1] + "%"

        pct = percent_int_or_none(s)
        if pct is None:
            raise ValueError(
                _("invalid_percent").format(val=raw)
                if hasattr(_, "__call__")
                else f"invalid percent: {raw}"
            )
        return (float(pct) / 100.0) * float(total_sec)

    # sonst normale Zeitangabe
    return parse_time_relaxed_seconds(s)


def yesno(b: bool | None) -> str:
    if b is None:
        return "?"
    return _("yes") if bool(b) else _("no")


def _parse_rational(s: str) -> Tuple[int, int, float]:
    try:
        if s and "/" in s:
            num, den = s.split("/", 1)
            num_i = max(0, int(num.strip()))
            den_i = max(1, int(den.strip()))
            return num_i, den_i, float(Fraction(num_i, den_i))
        f = float(s)
        if f > 0:
            fr = Fraction(f).limit_denominator(100000)
            return fr.numerator, fr.denominator, float(fr)
    except Exception:
        pass
    return (0, 1, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Mini-Helper: Auflösung eines Videos auslesen (ffprobe)
# ──────────────────────────────────────────────────────────────────────────────
def probe_resolution(path: StrPath, *, apply_rotation: bool = True) -> Tuple[int, int]:
    """
    Liefert (width, height) für Videos *und Bilder*.
    - Videos: ffprobe v:0, optional Rotation (DisplayMatrix / TAG:rotate).
    - Bilder: ffprobe-Streams mit width/height; falls das fehlschlägt, optional Pillow-Fallback
      (inkl. EXIF-Orientation 90/270 → w/h tauschen).
    Fehler → (0, 0).
    """
    p = str(Path(path))

    def _apply_stream_rotation(w: int, h: int, s: dict) -> Tuple[int, int]:
        if not apply_rotation:
            return (w, h)

        rot = 0
        # 1) side_data_list → displaymatrix
        try:
            for sd in s.get("side_data_list") or []:
                if sd.get(
                    "side_data_type", ""
                ).lower() == "displaymatrix" and isinstance(
                    sd.get("rotation"), (int, float)
                ):
                    rot = int(sd["rotation"]) % 360
                    break
        except Exception:
            pass

        # 2) tags.rotate (manche Container schreiben Rotation dort)
        if rot == 0:
            try:
                tags = s.get("tags") or {}
                r = tags.get("rotate")
                if isinstance(r, str) and r.strip().lstrip("+-").isdigit():
                    rot = int(r) % 360
            except Exception:
                pass

        if rot in (90, 270):
            return (h, w)
        return (w, h)

    # --- 1) Standard: Videostream v:0 ---
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,side_data_list,tags",
                "-of",
                "json",
                p,
            ],
            text=True,
        )
        data = json.loads(out)
        streams = data.get("streams") or []
        if streams:
            s = streams[0]
            w = int(s.get("width") or 0)
            h = int(s.get("height") or 0)
            if w > 0 and h > 0:
                w, h = _apply_stream_rotation(w, h, s)
                return (max(0, w), max(0, h))
    except Exception:
        pass

    # --- 2) Fallback: irgendein Stream mit width/height (z. B. bei Bildformaten) ---
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,side_data_list,tags",
                "-of",
                "json",
                p,
            ],
            text=True,
        )
        data = json.loads(out)
        streams = data.get("streams") or []

        # Bevorzugt 'video'-Streams, sonst erster mit width/height
        cand = None
        for s in streams:
            w = int(s.get("width") or 0)
            h = int(s.get("height") or 0)
            if w > 0 and h > 0 and (s.get("codec_type") or "").lower() == "video":
                cand = s
                break
        if cand is None:
            for s in streams:
                w = int(s.get("width") or 0)
                h = int(s.get("height") or 0)
                if w > 0 and h > 0:
                    cand = s
                    break

        if cand:
            w = int(cand.get("width") or 0)
            h = int(cand.get("height") or 0)
            w, h = _apply_stream_rotation(w, h, cand)
            if w > 0 and h > 0:
                return (w, h)
    except Exception:
        pass

    # --- 3) Letzter Fallback: reine Bilddateien via Pillow (optional) ---
    try:
        ext = Path(p).suffix.lower()
        img_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".jpe",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
            ".gif",
            ".avif",
            ".heic",
            ".heif",
            ".exr",
            ".ppm",
            ".pgm",
            ".pbm",
            ".pnm",
            ".ico",
            ".dds",
        }
        if ext in img_exts:
            try:
                from PIL import ExifTags, Image  # type: ignore

                with Image.open(p) as im:
                    w, h = im.size

                    if apply_rotation:
                        # EXIF-Orientation auswerten (nur 90/270 relevant für w/h-Tausch)
                        orientation = None
                        try:
                            exif = getattr(im, "_getexif", lambda: None)() or {}
                            if exif:
                                # Tag-ID für "Orientation" finden
                                ori_key = None
                                for k, v in ExifTags.TAGS.items():
                                    if v == "Orientation":
                                        ori_key = k
                                        break
                                if ori_key is not None:
                                    orientation = exif.get(ori_key)
                        except Exception:
                            orientation = None

                        # 6/8 sind 90°/270° gedrehte Varianten; 5/7 beinhalten ebenfalls 90°/270° (mit Spiegelung)
                        if orientation in (5, 6, 7, 8):
                            w, h = h, w
                    return (max(0, int(w)), max(0, int(h)))
            except Exception:
                pass
    except Exception:
        pass

    # Nichts gefunden
    return (0, 0)


def ffprobe_video_meta(video_path: Path) -> Dict[str, Any]:
    exe = shutil.which("ffprobe") or "ffprobe"
    try:
        cmd = [
            exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,time_base",
            "-of",
            "json",
            str(video_path),
        ]
        out = (
            mg.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ).stdout
            or "{}"
        )
        data = json.loads(out)
        st = (data.get("streams") or [{}])[0]
        r = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
        n, d, fpsf = _parse_rational(r)
        nb = int(st.get("nb_frames") or 0)
        dur = float(st.get("duration") or 0.0)
        tb = st.get("time_base") or "1/1000"
        print_log(
            f"[ffprobe_meta] fps={n}/{d}≈{fpsf:.3f} nb={nb} dur={dur:.3f}s tb={tb}"
        )
        return {
            "fps_num": n,
            "fps_den": d,
            "fps": fpsf,
            "nb_frames": nb,
            "duration": dur,
            "time_base": tb,
        }
    except Exception as e:
        co.print_warning(f"[ffprobe_meta] failed: {e}")
        return {
            "fps_num": 0,
            "fps_den": 1,
            "fps": 0.0,
            "nb_frames": 0,
            "duration": 0.0,
            "time_base": "1/1000",
        }


@overload
def sec_to_time_str(x: timedelta) -> str: ...
@overload
def sec_to_time_str(x: Union[float, int]) -> str: ...
def sec_to_time_str(x: Union[float, int, timedelta]) -> str:
    """Nimmt Sekunden (int/float) ODER timedelta und gibt 'HH:MM:SS' zurück."""
    if isinstance(x, timedelta):
        td = x
    else:
        td = timedelta(seconds=float(x))
    return seconds_to_time(td)


def escape_drawtext(text: str) -> str:
    """Escape : und ' für ffmpeg drawtext."""
    return text.replace(":", r"\:").replace("'", r"\'")


def strip_trailing_units(s: str) -> str:
    """
    Entfernt beliebige hintenangestellte Einheiten-/Text-Suffixe.
    - Zeitstrings (enthalten ':'): behält nur Ziffern, ':' sowie optional '.'/',' im letzten Teil.
      Beispiele: "09:43", "1:02:03", "01:23s", "00:10:05.5fps" -> "09:43", "1:02:03", "01:23", "00:10:05.5"
    - Reine Zahlen: gibt nur den Zahlenteil zurück; ',' wird zu '.' normalisiert.
      Beispiele: "30fps", "59.94 FPS", "37,5%", "12s" -> "30", "59.94", "37.5", "12"

    Rückgabe: bereinigter String ("" falls keine Zahl/Zeit erkennbar).
    """
    raw = (s or "").strip()
    if not raw:
        return ""

    # Fall A: Zeitstring – enthält ':'
    if ":" in raw:
        # nimm von Anfang an nur erlaubte Zeichen: Ziffern, :, ., , und Spaces
        # (Stoppt am ersten verbotenen Zeichen → schneidet endständige Suffixe ab)
        m = re.match(r"^\s*([0-9:\s]+(?:[.,]\d+)?)", raw)
        if m:
            token = m.group(1)
        else:
            # Fallback: filtere auf erlaubte Zeichen
            token = "".join(ch for ch in raw if ch.isdigit() or ch in ":., ")

        # Spaces innerhalb entfernen
        token = re.sub(r"\s+", "", token)
        # Dezimal-Komma optional auf Punkt normalisieren (hilfreich für spätere Parsings)
        token = token.replace(",", ".")
        return token

    # Fall B: Zahl mit möglichem Suffix – capture nur die führende Zahl
    #   ^\s*([+-]?\d+(?:[.,]\d+)?)
    m = re.match(r"^\s*([+-]?\d+(?:[.,]\d+)?)", raw)
    if m:
        return m.group(1).replace(",", ".")

    # Fallback: suche irgendwo eine Zahl (zur Not mittendrin)
    m2 = re.search(r"([+-]?\d+(?:[.,]\d+)?)", raw)
    if m2:
        return m2.group(1).replace(",", ".")

    return ""


def is_valid_time(
    s: str,
    not_time_suffix: Optional[str] = None,
    bare_means_time: bool = True,
) -> Optional[bool]:
    """
    Prüft, ob *s* eine gültige Zeit ist ODER – falls angegeben – eine gültige Zahl mit
    'not_time_suffix'. Zusätzlich sind Sekunden mit 's'/'S' (z. B. '12s', '12.5S') reserviert
    und gelten stets als *Zeit*.

    Rückgabe:
      - True  → gültig (Zeit ODER Zahl mit 'not_time_suffix')
      - False → ungültig
      - None  → spezielle Abbruch-Eingabe "0" (kompatibel zu bestehendem UI-Flow)

    Parameter:
      - not_time_suffix: z. B. "%", "fps", "frames" ...; None/"" → deaktiviert
      - bare_means_time: True → nackte Zahl ist Zeit (Sekunden), False → nackte Zahl ist NICHT Zeit
    """

    s = s.strip()
    if not s:
        return False
    if s == "0":
        return None

    import re

    # 1) "12s" / "12.5S" → Sekunden als Zeit (reserviertes Suffix)
    if re.fullmatch(r"^[+-]?\d+(?:[.,]\d+)?\s*[sS]$", s):
        try:
            float(s[:-1].strip().replace(",", "."))
            return True
        except ValueError:
            return False

    # 2) Zeit-Formate mit ':'  → HH:MM:SS(.ms) | MM:SS(.ms)
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            hh, mm, ss = parts
            if not (hh.isdigit() and mm.isdigit()):
                return False
            try:
                mm_i = int(mm)
                ss_f = float(ss.replace(",", "."))
            except ValueError:
                return False
            return (0 <= mm_i < 60) and (0 <= ss_f < 60)
        if len(parts) == 2:
            mm, ss = parts
            if not mm.isdigit():
                return False
            try:
                mm_i = int(mm)
                ss_f = float(ss.replace(",", "."))
            except ValueError:
                return False
            return (0 <= mm_i < 60) and (0 <= ss_f < 60)
        return False

    # 3) Zahl + frei wählbarer Nicht-Zeit-Suffix? (whitespace tolerant, case-insensitive)
    if not_time_suffix and not_time_suffix.strip():
        import re

        pat = rf"^\s*([+-]?\d+(?:[.,]\d+)?)\s*{re.escape(not_time_suffix.strip())}\s*$"
        m = re.fullmatch(pat, s, flags=re.IGNORECASE)
        if m:
            try:
                float(m.group(1).replace(",", "."))
                return True  # gültige "Nicht-Zeit" (Suffix vorhanden)
            except ValueError:
                return False

    # 4) Nackte Zahl → je nach bare_means_time Zeit (True) oder nicht (False)
    try:
        float(s.replace(",", "."))
        return True if bare_means_time else False
    except ValueError:
        return False


def is_valid_time_or_percent(s: str) -> Optional[bool]:
    """
    Abwärtskompatibler Wrapper:
      - akzeptiert Zeiten wie früher
      - akzeptiert 'N%' NUR, wenn 0 <= N <= 100 (wie bisher)
      - '12s' / '12.5S' gelten als Zeit (Sekunden)
      - nackte Zahl gilt als Zeit (Sekunden)
      - '0' → None
    """
    import re

    s0 = s.strip()
    if not s0:
        return False
    if s0 == "0":
        return None

    # Spezielle Prozent-Logik mit Range-Check (0..100) – exakt wie bisher
    m = re.fullmatch(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*%\s*$", s0)
    if m:
        try:
            val = float(m.group(1).replace(",", "."))
            return 0.0 <= val <= 100.0
        except ValueError:
            return False

    # Sonst: Zeit-Validierung (Suffix-Validierung hier deaktiviert)
    return is_valid_time(s0, not_time_suffix=None, bare_means_time=True)


def clamp_seek_time(t: float, duration_s: Optional[float]) -> float:
    """Clampt die Zielzeit in [0, duration-ε], damit ffmpeg sicher noch ein Frame hat."""
    if t < 0:
        t = 0.0
    if duration_s is not None:
        eps = 0.2
        t = min(t, max(duration_s - eps, 0.0))
    return t


def probe_duration_sec(path: Path) -> Optional[float]:
    """Ermittelt die Videolänge in Sekunden via ffprobe (float) oder None."""
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        val_str: str = res.stdout.strip()
        return float(val_str) if val_str else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Framerate helpers
# ──────────────────────────────────────────────────────────────────────────────


def _frac_to_float(fr: Any) -> Optional[float]:
    """'24000/1001' -> 23.976..., '0/0'/'N/A'/None -> None. Akzeptiert beliebige Eingaben robust."""
    if fr is None:
        return None
    s = str(fr).strip()
    if not s or s.upper() in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(s))
    except Exception:
        try:
            return float(s)
        except Exception:
            return None


def _plausible_fps(f: Optional[float]) -> Optional[float]:
    """Filtert Timebase-/Quatschwerte (z. B. 90000) heraus. Akzeptiert nur 1..240 fps."""
    if f is None:
        return None
    return f if (1.0 <= f <= 240.0) else None


def probe_src_fps(path: Path, ffprobe_bin: str = "ffprobe") -> Optional[float]:
    """
    Liefert eine plausible durchschnittliche FPS als float.
    Nutzt avg_frame_rate (Fallback: r_frame_rate) und filtert Timebase-Werte.
    """
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        data_any: Any = json.loads(subprocess.check_output(cmd, text=True))
        data: Dict[str, Any] = cast(
            Dict[str, Any], data_any if isinstance(data_any, dict) else {}
        )
        streams_any = data.get("streams", [])
        streams: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], streams_any if isinstance(streams_any, list) else []
        )
        st: Dict[str, Any] = streams[0] if streams else {}
        fps = _plausible_fps(_frac_to_float(st.get("avg_frame_rate")))
        if fps is None:
            fps = _plausible_fps(_frac_to_float(st.get("r_frame_rate")))
        return fps
    except Exception:
        return None


def format_fps(
    rate: Any,
    decimals: int = 2,
    decimal_sep: str = ",",
    strip_trailing_zeros: bool = False,
    fallback: str = "?",
) -> str:
    """
    Formatiert ein ffprobe-Framerate-Feld (z.B. '30000/1001', '25/1', '23.976')
    als String mit festen Dezimalstellen (default: 2) und wählbarem Dezimaltrennzeichen.
    """
    val: Optional[float] = None
    try:
        if rate is None:
            val = None
        elif isinstance(rate, (int, float)):
            val = float(rate)
        elif isinstance(rate, str):
            s = rate.strip()
            if not s or s.upper() in {"N/A", "NA", "INF", "NAN"}:
                val = None
            else:
                val = float(Fraction(s))
        else:
            # safest: go through string to avoid typing issues
            val = float(Fraction(str(rate)))
    except Exception:
        val = None

    if val is None:
        return fallback

    s_form = f"{val:.{decimals}f}"
    if decimal_sep != ".":
        s_form = s_form.replace(".", decimal_sep)

    if strip_trailing_zeros and decimals > 0:
        if decimal_sep == ",":
            s_form = s_form.rstrip("0").rstrip(",")
        else:
            s_form = s_form.rstrip("0").rstrip(".")
    return s_form


def _file_stream_fps_values(
    path: Union[str, Path], ffprobe_bin: str = "ffprobe"
) -> List[float]:
    """
    Liefert plausible FPS-Werte aller Videostreams einer Datei.
    Bevorzugt avg_frame_rate, fällt auf r_frame_rate zurück.
    Ignoriert time_base/tbn (z. B. 1/90000).
    """
    p = str(Path(path))
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=index,codec_type,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        p,
    ]
    try:
        data_any: Any = json.loads(subprocess.check_output(cmd, text=True))
    except Exception:
        return []

    data: Dict[str, Any] = cast(
        Dict[str, Any], data_any if isinstance(data_any, dict) else {}
    )
    streams_any = data.get("streams", [])
    streams: List[Dict[str, Any]] = cast(
        List[Dict[str, Any]], streams_any if isinstance(streams_any, list) else []
    )

    out: List[float] = []
    for st in streams:
        if st.get("codec_type") != "video":
            continue
        fps = _plausible_fps(_frac_to_float(st.get("avg_frame_rate")))
        if fps is None:
            fps = _plausible_fps(_frac_to_float(st.get("r_frame_rate")))
        if fps is not None:
            out.append(fps)
    return out


@overload
def get_framerate_range(
    files: Union[str, Path, Iterable[Union[str, Path]]],
    ffprobe_bin: str = "ffprobe",
    mode: Literal["file_max", "file_min", "per_stream"] = "file_max",
    *,
    return_paths: Literal[True],
) -> Tuple[Optional[float], Optional[Path], Optional[float], Optional[Path]]: ...
@overload
def get_framerate_range(
    files: Union[str, Path, Iterable[Union[str, Path]]],
    ffprobe_bin: str = "ffprobe",
    mode: Literal["file_max", "file_min", "per_stream"] = "file_max",
    *,
    return_paths: Literal[False] = False,
) -> Tuple[Optional[float], Optional[float]]: ...


def get_framerate_range(
    files: Union[str, Path, Iterable[Union[str, Path]]],
    ffprobe_bin: str = "ffprobe",
    mode: Literal["file_max", "file_min", "per_stream"] = "file_max",
    *,
    return_paths: bool = False,
) -> Union[
    Tuple[Optional[float], Optional[float]],
    Tuple[Optional[float], Optional[Path], Optional[float], Optional[Path]],
]:
    """
    Ermittelt MIN und MAX Framerate über 1..N Dateien mithilfe _file_stream_fps_values().
    """
    if isinstance(files, (str, Path)):
        file_list = [Path(files)]
    else:
        file_list = [Path(f) for f in files]

    global_min: Optional[float] = None
    global_max: Optional[float] = None
    min_path: Optional[Path] = None
    max_path: Optional[Path] = None

    if mode not in {"file_max", "file_min", "per_stream"}:
        mode = "file_max"

    for p in file_list:
        fps_vals = _file_stream_fps_values(p, ffprobe_bin=ffprobe_bin)
        if not fps_vals:
            continue

        if mode == "per_stream":
            for v in fps_vals:
                if (global_min is None) or (v < global_min):
                    global_min, min_path = v, p
                if (global_max is None) or (v > global_max):
                    global_max, max_path = v, p
        else:
            val = (max if mode == "file_max" else min)(fps_vals)
            if (global_min is None) or (val < global_min):
                global_min, min_path = val, p
            if (global_max is None) or (val > global_max):
                global_max, max_path = val, p

    if not return_paths:
        return global_min, global_max
    return global_min, min_path, global_max, max_path


def map_quality_to_crf(quality_percent: int) -> float:
    """100% -> CRF 18 (beste Qualität), 0% -> CRF 30 (max. Kompression)."""
    crf = 30 - (quality_percent / 100.0 * 12.0)
    return round(crf, 1)


def prepare_inputs(
    args: Any,
    *extension_groups: Sequence[str],
    exit_on_error: bool = True,
    files_required: Optional[bool] = None,
    start_signal_words: Optional[Iterable[str]] = None,
) -> Tuple[bool, list[str]]:
    """
    Rückgabe: (batch_mode, files)
    - Erkennt Ordner als Eingaben (über prepare_files).
    - Batch-Mode erst bestimmen, NACHDEM _files_source gesetzt wurde.
    - Bei leerem Ergebnis: alter Fehlerfluss bleibt gleich.
    """
    # 0) args ggf. instanziieren
    if isinstance(args, type):
        args = args()

    # 1) Signalwörter (optional)
    tokens = [re.sub(r"^[\-]+", "", t).lower() for t in sys.argv[1:]]
    sigs = [w.lower() for w in start_signal_words] if start_signal_words else []
    signal_hit = any(t in sigs for t in tokens) if sigs else False

    # 2) Dateien benötigt?
    requires_files = bool(
        getattr(args, "requires_files", True)
        if files_required is None
        else files_required
    )

    # 3) Wenn keine Dateien benötigt werden → fertig
    if not requires_files:
        files_attr = getattr(args, "files", [])
        if isinstance(files_attr, list) and files_attr:
            return signal_hit, files_attr
        return signal_hit, []

    # 4) Extensions (Fallback: VIDEO_EXTENSIONS)
    if not extension_groups:
        extension_groups = (
            getattr(
                defin, "VIDEO_EXTENSIONS", (".mp4", ".mkv", ".mov", ".avi", ".mpg")
            ),
        )

    # 5) Dateien einsammeln (inkl. Ordner-Expansion)
    files = fs.prepare_files(args, *extension_groups)

    # 6) Batch-Mode jetzt korrekt bestimmen
    batch_mode = (getattr(args, "_files_source", None) == "cli") or signal_hit

    # 6b) Erkennen, ob überhaupt Flags übergeben wurden (z. B. -f, --codec, ...)
    # cli_flags_used = any(
    #    isinstance(a, str) and a.startswith("-") for a in sys.argv[1:]
    # )

    # 8) Fehlerbehandlung wie bisher, aber "passed_no_file" nur,
    #    wenn Flags übergeben wurden und keine Dateien angegeben sind
    if not files:
        if exit_on_error:
            # provided = getattr(args, "files", None)
            # if cli_flags_used and not provided:
            #    co.print_error(_("passed_no_file"))
            sys.exit(1)
        return batch_mode, []

    return batch_mode, files


# ──────────────────────────────────────────────────────────────────────────────
# Real-ESRGAN / torch helpers (keine Typwarnungen, defensiv)
# ──────────────────────────────────────────────────────────────────────────────


_MAX_PY_OK = (3, 11)  # Real-ESRGAN wheels not yet built for 3.12+


def run_py(venv_python: Path, code: str) -> bool:
    return subprocess.run([str(venv_python), "-c", code]).returncode == 0


def run_py_out(
    venv_python: Path, code: str, *, env: Optional[Dict[str, str]] = None
) -> str:
    """
    Führt Python-Code in einem (virtuellen) Interpreter aus und gibt stdout zurück.
    - NEU: Optionales 'env' wird 1:1 an subprocess.run übergeben.
           Fallback ist eine Kopie der aktuellen Umgebung (keine „leere“ CUDA-Maske).
    """
    run_env = env if env is not None else os.environ.copy()
    res = subprocess.run(
        [str(venv_python), "-c", code],
        capture_output=True,
        text=True,
        errors="replace",
        env=run_env,
    )
    return res.stdout or ""


def pip(venv_python: Path, *args: str) -> None:
    subprocess.run(
        [str(venv_python), "-m", "pip", *args],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_torch_packages(venv_python: Path, cuda_suffix: str = "cu118") -> None:
    """
    Verify torch + torchvision imports. Install compatible versions if missing.
    For Python 3.12+, official wheels are not available → raise RuntimeError.
    """
    ver_out = run_py_out(
        venv_python, "import sys,json;print(json.dumps(sys.version_info[:3]))"
    ).strip()
    try:
        major, minor, _patch = [int(x) for x in ver_out.strip("[]").split(",")]
    except Exception:
        major, minor = 3, 11  # assume safe default

    if (major, minor) > _MAX_PY_OK:
        raise RuntimeError(
            f"Real-ESRGAN wheels are not available for Python {major}.{minor}. "
            "Please recreate your venv with Python 3.11 or lower."
        )

    if run_py(
        venv_python,
        "from torchvision.transforms.functional import rgb_to_grayscale; import torch",
    ):
        return  # already ok

    print("[upscale] Installing Real-ESRGAN-compatible torch/torchvision …")
    torch_ver = f"1.13.1+{cuda_suffix}"
    tv_ver = f"0.14.1+{cuda_suffix}"

    pip(venv_python, "uninstall", "-y", "torch", "torchvision")
    pip(
        venv_python,
        "install",
        f"torch=={torch_ver}",
        f"torchvision=={tv_ver}",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    )

    if not run_py(
        venv_python, "from torchvision.transforms.functional import rgb_to_grayscale"
    ):
        raise RuntimeError(
            "Torchvision still incompatible after installation – aborting upscaling."
        )


def detect_frame_count(path: Path, python_exec: Path, fallback_fps: str = "25") -> int:
    """Ermittelt die Frameanzahl (mehrstufiger Fallback)."""
    out = (
        run_py_out(
            python_exec,
            f"import ffmpeghelper; print(ffmpeghelper.detect_frame_count({path!r}))",
        )
        or ""
    )
    try:
        n = int(out.strip())
        if n > 0:
            return n
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        n = int(res.stdout.strip())
        if n > 0:
            return n
    except Exception:
        pass

    try:
        cmd_dur = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        dur = float(
            subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip()
        )
        fps_val = float(
            run_py_out(
                python_exec,
                f"import ffmpeghelper; print(ffmpeghelper.detect_fps({path!r}))",
            )
            or fallback_fps
        )
        return int(dur * fps_val)
    except Exception:
        return 0


def calculate_total_frames(files) -> int:
    """
    Summiert die Frame-Anzahlen über alle übergebenen Inputs.
    - Unterstützt: einzelne Datei, Liste/Tuple/Set von Dateien, Ordner mit frame_*.{bmp,png,jpg}
    - Nutzt he.detect_frame_count (mit ffmpeghelper-FPS) + ffprobe-Fallback.
    """
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    def _to_iter(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    def _ffprobe_count_frames(src: Path) -> int:
        exe = shutil.which("ffprobe") or "ffprobe"
        # 1) nb_read_frames (dekodiert)
        try:
            cmd = [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nw=1:nk=1",
                str(src),
            ]
            out = (
                subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                ).stdout
                or ""
            )
            val = out.strip()
            if val.isdigit():
                return int(val)
            # manchmal "N/A"
        except Exception:
            pass
        # 2) nb_frames (Container-Metadaten)
        try:
            cmd = [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nw=1:nk=1",
                str(src),
            ]
            out = (
                subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                ).stdout
                or ""
            )
            val = out.strip()
            if val.isdigit():
                return int(val)
        except Exception:
            pass
        # 3) count_packets (Annäherung)
        try:
            cmd = [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "default=nw=1:nk=1",
                str(src),
            ]
            out = (
                subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                ).stdout
                or ""
            )
            val = out.strip()
            if val.isdigit():
                return int(val)
        except Exception:
            pass
        return 0

    total = 0
    venv_python = Path(sys.executable)

    for item in _to_iter(files):
        p = Path(item)
        if not p.exists():
            try:
                co.print_warning(f"Input nicht gefunden: {p}")
            except Exception:
                pass
            continue

        # Ordner mit bereits extrahierten Frames?
        if p.is_dir():
            cnt = (
                len(list(p.glob("frame_*.bmp")))
                + len(list(p.glob("frame_*.png")))
                + len(list(p.glob("frame_*.jpg")))
            )
            total += max(0, int(cnt))
            continue

        # Videodatei: zuerst FPS bestimmen (optional), dann detect_frame_count
        fps_str = None
        try:
            fps_str = (
                run_py_out(
                    venv_python,
                    f"import ffmpeghelper;print(ffmpeghelper.detect_fps({repr(str(p))}))",
                )
                or None
            )
        except Exception:
            fps_str = None

        try:
            fc = detect_frame_count(p, venv_python, fallback_fps=(fps_str or "25"))
        except Exception:
            fc = -1

        if not isinstance(fc, int) or fc <= 0:
            # ffprobe-Fallback
            fc = _ffprobe_count_frames(p)

        if fc is None or fc < 0:
            fc = 0

        total += int(fc)

    return int(total)


# ──────────────────────────────────────────────────────────────────────────────
# Duration helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_duration(file_path: Union[str, Path]) -> Optional[float]:
    """
    Ermittelt die Dauer (Sekunden) über ffmpeg -i stderr-Parse.
    Rückgabe None bei Fehlern/ffmpeg fehlt.
    """
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(file_path)],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except FileNotFoundError:
        co.print_error(_("ffmpeg_missing_install"))
    return None


def total_sec(paths: List[Path], pause_len: float = 0.0, pause_count: int = 0) -> float:
    """Summe der Dauern aller Dateien (+ pauses)."""
    total = 0.0
    for p in paths:
        total += get_duration(p) or 0.0
    total += pause_len * pause_count
    return total


# ──────────────────────────────────────────────────────────────────────────────
# UI/console helpers
# ──────────────────────────────────────────────────────────────────────────────


def gradient_colour(progress: float) -> Tuple[int, int, int]:
    """Return RGB tuple for a purple→blue→green gradient at *progress* (0–1)."""
    p = max(0.0, min(progress, 1.0))
    if p < 0.5:
        t = p * 2
        r = int(255 * (1 - t))
        g = 0
        b = 255
    else:
        t = (p - 0.5) * 2
        r = 0
        g = int(255 * t)
        b = int(255 * (1 - t))
    return r, g, b


def format_color_params(params: Mapping[str, Any]) -> str:
    """
    Gibt alle Farbkorrektur-Parameter ≠ 50 % als formatierten String zurück,
    in stabiler Reihenfolge.
    """
    ordered = [
        ("brightness", _("brightness")),
        ("tint", _("tint")),
        ("warmth", _("warmth")),
        ("contrast", _("contrast")),
        ("saturation", _("saturation")),
    ]
    parts: List[str] = []
    for key, label in ordered:
        v = params.get(key)
        if isinstance(v, (int, float)) and int(round(v)) != 50:
            parts.append(f"{label}={int(round(v))}%")
    return ", ".join(parts)


def guess_lang_from_name(stem: str) -> Optional[str]:
    # Treffer wie ".de", "_eng", "-german", "(GER)", "[DEU]" etc.
    m = re.search(
        r"(?:^|[._\-\[\(])([A-Za-z]{2,5}|german|english|french|spanish|italian|portuguese|russian)(?:$|[._\-\]\)])",
        stem,
        re.I,
    )
    if not m:
        return None
    key = m.group(1).lower()
    return defin.LANG_ISO3.get(key)


def ffprobe_tags(path: str) -> dict:
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream_tags:format_tags",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def count_audio_streams(video_path: str) -> int:
    """Anzahl vorhandener Audio-Streams im (Video-)Container."""
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return len([ln for ln in (p.stdout or "").splitlines() if ln.strip()])
    except Exception:
        return 0


def sanitize_title(s: str, max_len: int = 200) -> str:
    """Entfernt Steuerzeichen, normalisiert Unicode, kollabiert Whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    # Steuerzeichen & BOMs entfernen
    s = "".join(ch for ch in s if ch.isprintable() and ch not in "\u200b\ufeff\r\n\t")
    # Whitespace normalisieren
    s = re.sub(r"\s+", " ", s).strip()
    # Optionale „unsaubere“ Zeichen raus (wenn du ganz streng sein willst)
    s = re.sub(r"[\"`´]+", "", s)  # Anführungszeichen o. Ä. optional entfernen
    # Länge begrenzen (einige UIs schneiden sonst hässlich ab)
    return s[:max_len]
