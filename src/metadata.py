#!/usr/bin/env python3
# metadata.py
import json
import re
import shutil
import subprocess
from dataclasses import field, make_dataclass
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Type, cast

import consoleOutput as co
import definitions as defin
import fileSystem as fs
import helpers as he
import metadata_support as ms
import userInteraction as ui
import video_thumbnail as vt

# local modules
from i18n import _, tr

# --- Virtuelle, nur lesbare Query-Tags (nicht setz-/löschbar) ---------------
VIRTUAL_QUERY_TAGS: List[str] = list(getattr(defin, "VIRTUAL_META_INFO", {}).keys())


def _build_MetadataArgs_dataclass() -> Type[Any]:
    """Erzeugt dynamisch eine Args-Dataclass inkl. Felder aus defin.*-Tabellen."""
    fields: List[Tuple[str, Any, Any]] = [
        ("files", List[str], field(default_factory=list)),
        ("list_tags", bool, False),
        ("list_tags_json", bool, False),
        ("list_tagsname", bool, False),
        # --tag kann mehrfach vorkommen: action="append" -> List[str]
        ("tag", Optional[List[str]], None),
        ("all", bool, None),
    ]
    for key in getattr(defin, "EDITABLE_META_KEYS", []):
        fields.append((key, Optional[str], None))  # --{key}
        fields.append((f"list_tag_{key}", bool, False))  # --list-tag-{key}
        fields.append((f"set_tag_{key}", Optional[str], None))  # --set-tag-{key}
        fields.append((f"delete_tag_{key}", bool, False))  # --delete-tag-{key}
    for pkey in getattr(defin, "PROTECTED_META_KEYS", []):
        fields.append((f"list_tag_{pkey}", bool, False))
    for vkey in VIRTUAL_QUERY_TAGS:
        fields.append((f"list_tag_{vkey}", bool, False))
    return make_dataclass("MetadataArgs", fields)


# WICHTIG: wirkliche Dataclass-Klasse mit Namen "MetadataArgs" erstellen
MetadataArgs = _build_MetadataArgs_dataclass()


def _probe_stream_info(path: Path) -> dict[str, Any]:
    """Liest Basis-Streaminfos via ffprobe (nur v:0) und gibt das JSON-Objekt zurück."""
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,pix_fmt,"
            "sample_aspect_ratio,display_aspect_ratio,"
            "color_primaries,color_transfer,color_space,color_range,chroma_location",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        info: dict[str, Any] = json.loads(r.stdout or "{}")
        streams: List[dict[str, Any]] = cast(
            List[dict[str, Any]], info.get("streams") or []
        )
        stream: dict[str, Any] = streams[0] if streams else {}
    except Exception:
        stream = {}
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "sample_aspect_ratio": stream.get("sample_aspect_ratio"),
        "display_aspect_ratio": stream.get("display_aspect_ratio"),
        # NEW: color signalling
        "color_primaries": stream.get("color_primaries"),
        "color_transfer": stream.get("color_transfer"),
        "color_space": stream.get("color_space"),
        "color_range": stream.get("color_range"),
        "chroma_location": stream.get("chroma_location"),
    }


def _editable_keys() -> Set[str]:
    return set(getattr(defin, "EDITABLE_META_KEYS", []))


def _protected_keys() -> Set[str]:
    return set(getattr(defin, "PROTECTED_META_KEYS", []))


def _compose_tag_views(
    path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Liefert (raw_tags, canon_editable, display_tags).

    - raw_tags: direkte ffprobe-Format-Tags (lowercase), container-spezifisch.
    - canon_editable: *kanonische* editable Tags über metadata_support (mit AVI/Unicode-Fallbacks).
    - display_tags: für die UI – Protected/Other aus raw, Editable aus canon_editable.
    """
    raw_tags = _read_format_tags(path)
    canon_edit = ms.read_editable_metadata(path)  # << zentrale Lesefunktion
    display = dict(raw_tags)
    for k, v in canon_edit.items():
        display[k] = v
    return raw_tags, canon_edit, display


def _split_tag_groups_display(
    raw_tags: dict[str, str], canon_edit: dict[str, str]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    pset, eset = _protected_keys(), _editable_keys()
    protected = {k: v for k, v in raw_tags.items() if k in pset}
    editable = {k: v for k, v in canon_edit.items() if k in eset}
    others = {k: v for k, v in raw_tags.items() if k not in pset and k not in eset}
    return protected, editable, others


def _virtual_tag_value(path: Path, key: str) -> str:
    key = key.lower()
    if key == "container":
        return path.suffix.lstrip(".").lower() or "?"
    if key == "duration":
        _, _, dur = _probe_format_info(path)
        return _format_hms(dur)
    si: dict[str, Any] = _probe_stream_info(path)
    if key == "resolution":
        w = str(si.get("width") or "?")
        h = str(si.get("height") or "?")
        return f"{w}x{h}"
    if key == "codec":
        return str(si.get("codec_name") or "?")
    if key == "fps":
        return he.format_fps(str(si.get("r_frame_rate") or "0/1"))
    if key in {"pixel_format", "pix_fmt", "pixelformat"}:
        return str(si.get("pix_fmt") or "?")

    # NEW: color signalling + chapters
    if key == "primaries":
        return _friendly_color_value("primaries", si.get("color_primaries"))
    if key == "trc":
        return _friendly_color_value("trc", si.get("color_transfer"))
    if key == "matrix":
        return _friendly_color_value("matrix", si.get("color_space"))
    if key == "range":
        return _friendly_color_value("range", si.get("color_range"))
    if key == "chapters":
        try:
            return str(len(_probe_chapters(path)))
        except Exception:
            return "0"

    # --- virtuals printed in UI ---
    if key == "filename":
        return path.stem
    if key == "file_size":
        try:
            return _fmt_bytes(path.stat().st_size)
        except Exception:
            return "?"
    if key == "container_long":
        fmt_name, fmt_long, _ = _probe_format_info(path)
        return str(fmt_long or fmt_name or "")
    if key == "display_resolution":
        si = _probe_stream_info(path)
        enc_w = cast(Optional[int], si.get("width"))
        enc_h = cast(Optional[int], si.get("height"))
        sar = si.get("sample_aspect_ratio") or "1:1"
        dw = _display_width_from_sar(enc_w, str(sar))
        return f"{dw}x{enc_h}" if (dw and enc_h) else "?"
    if key == "sar":
        return str(_probe_stream_info(path).get("sample_aspect_ratio") or "1:1")
    if key == "dar":
        si = _probe_stream_info(path)
        enc_w = si.get("width")
        enc_h = si.get("height")
        return str(si.get("display_aspect_ratio") or f"{enc_w}:{enc_h}")
    if key == "thumbnail":
        try:
            return he.yesno(vt.check_thumbnail(path, silent=True))
        except Exception:
            return "?"
    if key == "alpha_channel":
        try:
            si = _probe_stream_info(path)
            return he.yesno(_pixfmt_supports_alpha(str(si.get("pix_fmt") or "")))
        except Exception:
            return "?"
    if key == "transparency":
        try:
            _, has_trans = _detect_alpha_content(path, max_frames=200)
            return he.yesno(has_trans)
        except Exception:
            return "?"
    if key in {
        "audio_codec",
        "audio_bitrate",
        "audio_channels",
        "audio_sample_rate",
        "audio_language",
    }:
        a = _audio_summary(path)
        return a.get(
            {
                "audio_codec": "codec",
                "audio_bitrate": "bitrate",
                "audio_channels": "channels",
                "audio_sample_rate": "sample_rate",
                "audio_language": "language",
            }[key],
            "",
        )
    if key == "audio_streams":
        try:
            return str(len(_probe_audio_streams(path)))
        except Exception:
            return "0"
    if key == "subtitle_streams":
        try:
            return str(len(_probe_subtitle_streams(path)))
        except Exception:
            return "0"

    return ""


def _tag_display_name(key: str, show_code: bool = True) -> str:
    meta = getattr(defin, "META_TAGS", {}).get(key)
    if meta and "name" in meta:
        try:
            disp = tr(meta["name"])
        except Exception:
            name = meta["name"]
            disp = (
                (name.get("de") or name.get("en") or key)
                if isinstance(name, dict)
                else str(name)
            )
    else:
        vmeta = getattr(defin, "VIRTUAL_META_INFO", {}).get(key, {})
        if vmeta and "name" in vmeta:
            try:
                disp = tr(vmeta["name"])
            except Exception:
                name = vmeta["name"]
                disp = (
                    (name.get("de") or name.get("en") or key)
                    if isinstance(name, dict)
                    else str(name)
                )
        else:
            disp = key
    if show_code and disp != key:
        return f"{disp} ({key})"
    return disp


def _sort_by_localized_name(keys: List[str]) -> List[str]:
    return sorted(keys, key=lambda k: _tag_display_name(k, show_code=False).lower())


def _probe_format_info(
    path: Path,
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=format_name,format_long_name,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    fmt: dict[str, Any] = {}
    try:
        fmt = cast(dict[str, Any], json.loads(r.stdout or "{}").get("format", {}) or {})
    except json.JSONDecodeError:
        pass
    fmt_name = cast(Optional[str], fmt.get("format_name"))
    fmt_long = cast(Optional[str], fmt.get("format_long_name"))
    dur_raw = fmt.get("duration")
    try:
        duration: Optional[float] = float(dur_raw) if dur_raw is not None else None
    except Exception:
        duration = None
    return fmt_name, fmt_long, duration


def _probe_audio_streams(path: Path) -> List[dict[str, Any]]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            (
                "stream=index,codec_name,bit_rate,channels,channel_layout,sample_rate,disposition"
                ":stream_tags=language,title,BPS,BPS-eng,BPS_ENG,BPS-NOMINAL,BPS-NOMINAL-eng"
            ),
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(r.stdout or "{}")
        return cast(List[dict[str, Any]], data.get("streams") or [])
    except Exception:
        return []


def _pick_default_audio(streams: List[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for s in streams:
        disp = cast(dict, s.get("disposition") or {})
        if (disp.get("default") == 1) or (disp.get("dub") == 1):
            return s
    return streams[0] if streams else None


def _extract_bitrate_int(s: dict[str, Any]) -> Optional[int]:
    br = s.get("bit_rate")
    if not br:
        tags = cast(dict, s.get("tags") or {})
        for k in ("BPS-eng", "BPS_ENG", "BPS", "BPS-NOMINAL", "BPS-NOMINAL-eng"):
            if tags.get(k):
                br = tags.get(k)
                break
    try:
        return int(str(br))
    except Exception:
        return None


def _fmt_sr(sr: Any) -> str:
    try:
        n = int(str(sr))
        return f"{n / 1000:.1f} kHz" if n >= 1000 else f"{n} Hz"
    except Exception:
        return "—"


def _audio_summary(path: Path) -> dict[str, str]:
    streams = _probe_audio_streams(path)
    if not streams:
        return {
            "codec": "—",
            "bitrate": "—",
            "channels": "—",
            "sample_rate": "—",
            "language": "",
            "language_code": "",
            "count": "0",
        }
    s = _pick_default_audio(streams) or streams[0]
    codec = str(s.get("codec_name") or "—")
    br = _extract_bitrate_int(s)
    bitrate = _fmt_kbps(br) if br else "—"
    ch = s.get("channels")
    layout = s.get("channel_layout")
    channels = (
        f"{ch} ({layout})"
        if ch and layout
        else str(ch) if ch else str(layout) if layout else "—"
    )
    sr = _fmt_sr(s.get("sample_rate"))
    tags = cast(dict, s.get("tags") or {})
    lang_code = str(tags.get("language") or "").strip()
    lang_disp = _friendly_language_display(lang_code) if lang_code else ""
    return {
        "codec": codec,
        "bitrate": bitrate,
        "channels": channels,
        "sample_rate": sr,
        "language": lang_disp,
        "language_code": lang_code,
        "count": str(len(streams)),
    }


# --- Farbsignalisierung / Color signalling -----------------------------------
def _friendly_color_value(kind: str, val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    if not v:
        return "—"
    if kind == "primaries":
        mapping = {
            "bt709": "BT.709",
            "bt470bg": "BT.601 PAL (BT.470BG)",
            "smpte170m": "BT.601 NTSC (SMPTE 170M)",
            "bt2020": "BT.2020",
            "smpte432": "DCI-P3 D65 (SMPTE 432)",
            "smpte431": "DCI-P3 (Theater)",
            "jedec-p22": "JEDEC P22",
        }
    elif kind == "trc":
        mapping = {
            "bt709": "BT.709 / BT.1886",
            "iec61966-2-1": "sRGB",
            "gamma22": "Gamma 2.2",
            "gamma28": "Gamma 2.8",
            "smpte2084": "PQ (ST 2084)",
            "arib-std-b67": "HLG (ARIB B67)",
            "linear": "Linear",
            "log": "Log",
            "log_sqrt": "LogSqrt",
        }
    elif kind == "matrix":
        mapping = {
            "bt709": "BT.709",
            "smpte170m": "BT.601 NTSC",
            "bt470bg": "BT.601 PAL",
            "bt2020nc": "BT.2020 non-const",
            "bt2020c": "BT.2020 const-lumin.",
            "ycgco": "YCgCo",
            "rgb": "RGB",
        }
    elif kind == "range":
        mapping = {
            "tv": "Limited (TV/Video)",
            "mpeg": "Limited (MPEG)",
            "pc": "Full (PC)",
            "jpeg": "Full (JPEG)",
        }
    else:
        mapping = {}

    return mapping.get(v, v)


def _print_color_info_from_streaminfo(si: dict[str, Any]) -> None:
    prim = _friendly_color_value("primaries", si.get("color_primaries"))
    trc = _friendly_color_value("trc", si.get("color_transfer"))
    mat = _friendly_color_value("matrix", si.get("color_space"))
    rng = _friendly_color_value("range", si.get("color_range"))
    co.print_value_info("   " + _("color_primaries_label"), prim)
    co.print_value_info("   " + _("color_trc_label"), trc)
    co.print_value_info("   " + _("color_matrix_label"), mat)
    co.print_value_info("   " + _("color_range_label"), rng)


# --- Sprache / Language (freundliche Anzeige) --------------------------------
def _friendly_language_name(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""

    # 1) Bevorzugt deine robuste Heuristik (nutzt Codes, Aliasse, Mischformen)
    try:
        name = he.guess_lang_from_name(s)
        if isinstance(name, str) and name.strip() and name.strip().lower() != s.lower():
            return name.strip()
    except Exception:
        pass

    # 2) Definitions-Maps verwenden (statt lokalem 'm'-Dict)
    try:
        key = re.split(r"[,;|/_\-\s]+", s.lower())[
            0
        ]  # erster Token, z.B. "pt-br" → "pt-br"
        lang_iso3_map = getattr(defin, "LANG_ISO3", {}) or {}
        lang_display = getattr(defin, "LANG_DISPLAY", {}) or {}

        # Alias/Code → ISO3 (direkt, oder Fallback auf die ersten 2 Zeichen wie "en-us" → "en")
        iso = lang_iso3_map.get(key)
        if not iso and len(key) > 2:
            iso = lang_iso3_map.get(key[:2])

        # Falls schon ein ISO3-Code übergeben wurde, der in LANG_DISPLAY existiert
        if not iso and key in lang_display:
            iso = key

        if iso:
            disp = lang_display.get(iso)
            if isinstance(disp, str) and disp.strip():
                return disp.strip()
    except Exception:
        pass

    # 3) Fallback: unverändert zurückgeben
    return s


def _friendly_language_display(code_or_label: str) -> str:
    if not code_or_label:
        return ""
    name = _friendly_language_name(code_or_label)
    return (
        f"{name} ({code_or_label})"
        if name and name.lower() != code_or_label.lower()
        else code_or_label
    )


# --- Kapitel / Chapters -------------------------------------------------------
def _probe_chapters(path: Path) -> List[dict[str, Any]]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(r.stdout or "{}")
        chapters = cast(List[dict[str, Any]], data.get("chapters") or [])
    except Exception:
        chapters = []
    out: List[dict[str, Any]] = []
    for i, ch in enumerate(chapters, 1):
        try:
            start = float(ch.get("start_time") or 0.0)
        except Exception:
            start = 0.0
        try:
            end = float(ch.get("end_time") or 0.0)
        except Exception:
            end = 0.0
        tags = cast(dict, ch.get("tags") or {})
        title = str(tags.get("title") or tags.get("TITLE") or "").strip()
        out.append({"index": i, "title": title, "start": start, "end": end})
    return out


def _print_chapters_count_or_list(path: Path, show_all: bool) -> None:
    try:
        chs = _probe_chapters(path)
        co.print_value_info("   " + _("chapters_label"), str(len(chs)))
        if show_all and chs:
            for ch in chs:
                title = ch["title"] or _("chapter_fallback_title").format(
                    index=f"{ch['index']:02d}"
                )
                span = f"{_format_hms(ch['start'])} – {_format_hms(ch['end'])}"
                co.print_value_info(f"      {ch['index']:02d}. {title}", span)
    except Exception:
        pass


def _print_audio_info(path: Path) -> None:
    try:
        a = _audio_summary(path)
        co.print_value_info("   " + _("audio_codec_label"), a["codec"])
        co.print_value_info("   " + _("audio_bitrate_label"), a["bitrate"])
        co.print_value_info("   " + _("audio_channels_label"), a["channels"])
        co.print_value_info("   " + _("sample_rate_label"), a["sample_rate"])
        if a["language"]:
            co.print_value_info("   " + _("language_label"), a["language"])
    except Exception:
        co.print_value_info("   " + _("audio_codec_label"), "—")
        co.print_value_info("   " + _("audio_bitrate_label"), "—")


def _probe_subtitle_streams(path: Path) -> List[dict[str, Any]]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name,disposition:stream_tags=language,title",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(r.stdout or "{}")
        return cast(List[dict[str, Any]], data.get("streams") or [])
    except Exception:
        return []


def _print_audio_streams_list(path: Path) -> None:
    streams = _probe_audio_streams(path)
    if not streams:
        return
    for s in streams:
        idx = s.get("index")
        codec = s.get("codec_name") or "?"
        tags = cast(dict, s.get("tags") or {})
        lang_code = (tags.get("language") or "").strip()
        lang = _friendly_language_display(lang_code) if lang_code else ""
        title = (tags.get("title") or "").strip()
        disp = cast(dict, s.get("disposition") or {})
        flags = [k for k, v in disp.items() if v == 1]
        meta = ", ".join(flags) if flags else ""
        line = f"{codec}"
        if lang:
            line += f", {lang}"
        if title:
            line += f" — {title}"
        if meta:
            line += f" [{meta}]"
        co.print_value_info(f"      a:{idx}", line)


def _print_subtitle_streams_list(path: Path) -> None:
    streams = _probe_subtitle_streams(path)
    if not streams:
        return
    for s in streams:
        idx = s.get("index")
        codec = s.get("codec_name") or "?"
        tags = cast(dict, s.get("tags") or {})
        lang_code = (tags.get("language") or "").strip()
        lang = _friendly_language_display(lang_code) if lang_code else ""
        title = (tags.get("title") or "").strip()
        disp = cast(dict, s.get("disposition") or {})
        flags = [k for k, v in disp.items() if v == 1]
        meta = ", ".join(flags) if flags else ""
        line = f"{codec}"
        if lang:
            line += f", {lang}"
        if title:
            line += f" — {title}"
        if meta:
            line += f" [{meta}]"
        co.print_value_info(f"      s:{idx}", line)


# --- Container-Fähigkeiten ----------------------------------------------------
_AUDIO_CONTAINER_EXTS = {
    "mp4",
    "m4v",
    "mov",
    "mkv",
    "webm",
    "avi",
    "mxf",
    "ts",
    "m2ts",
    "flv",
}
_SUBTITLE_CONTAINER_EXTS = {"mkv", "mp4", "m4v", "mov", "webm"}


def _container_supports_audio(container_ext: str) -> bool:
    return (container_ext or "").lower() in _AUDIO_CONTAINER_EXTS


def _container_supports_subs(container_ext: str) -> bool:
    return (container_ext or "").lower() in _SUBTITLE_CONTAINER_EXTS


def _format_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    s = int(round(seconds))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02}:{mm:02}:{ss:02}"


def _read_format_tags(path: Path) -> dict[str, str]:
    """
    Roh-Lesen für Anzeige/Protected:
    ffprobe + (bei AVI) ExifTool-Ergänzung für Felder wie IARL.
    """
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        fmt = json.loads(r.stdout or "{}").get("format", {}) or {}
        tags = dict(
            (str(k).lower(), str(v)) for k, v in (fmt.get("tags", {}) or {}).items()
        )
    except Exception:
        tags = {}
    if path.suffix.lower() == ".avi" and shutil.which("exiftool"):
        rx = subprocess.run(
            ["exiftool", "-j", "-n", "-charset", "filename=utf8", str(path)],
            capture_output=True,
            text=True,
        )
        if rx.returncode == 0:
            try:
                arr = json.loads(rx.stdout or "[]")
                if isinstance(arr, list) and arr:
                    meta = arr[0]
                    val = meta.get("ArchivalLocation")
                    if val:
                        tags.setdefault("iarl", str(val))
            except Exception:
                pass
    return tags


# --- kleine Helfer -----------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    try:
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        size = float(n)
        i = 0
        while size >= 1024.0 and i < len(units) - 1:
            size /= 1024.0
            i += 1
        return f"{size:.0f} {units[i]}" if size >= 100 else f"{size:.1f} {units[i]}"
    except Exception:
        return "?"


def _fmt_kbps(bps_val: Any) -> str:
    try:
        bps = int(bps_val)
        kbps = bps / 1000.0
        return f"{int(round(kbps))} kb/s" if kbps >= 100 else f"{kbps:.1f} kb/s"
    except Exception:
        return "?"


def _parse_ratio_safe(s: Optional[str]) -> Tuple[int, int]:
    try:
        if not s or ":" not in s:
            return (1, 1)
        a, b = s.split(":", 1)
        na = int(a) if int(a) > 0 else 1
        nb = int(b) if int(b) > 0 else 1
        return (na, nb)
    except Exception:
        return (1, 1)


def _display_width_from_sar(enc_w: Optional[int], sar: Optional[str]) -> Optional[int]:
    if not enc_w or enc_w <= 0:
        return None
    n, d = _parse_ratio_safe(sar)
    if n == 1 and d == 1:
        return None
    try:
        disp_w = int(round(enc_w * (n / d)))
        return max(1, disp_w)
    except Exception:
        return None


# --- Alpha / Transparenz -----------------------------------------------------
def _pixfmt_supports_alpha(pxfmt: Optional[str]) -> bool:
    if not pxfmt:
        return False
    p = str(pxfmt).lower()
    import re

    if re.match(r"^(?:argb|abgr|rgba|bgra)(?:\d+)?(?:le|be)?$", p):  # RGBA-Familie
        return True
    if p.startswith("gbrap"):  # planar GBR mit Alpha
        return True
    if p.startswith("yuva"):  # YUVA-Varianten
        return True
    if re.match(r"^ya\d+(?:le|be)?$", p):  # Luma+Alpha
        return True
    if p.startswith("ayuv"):
        return True
    return False


def _detect_alpha_content(
    path: Path, max_frames: int = 200
) -> tuple[bool, Optional[bool]]:
    """
    Bestimmt, ob ein Clip einen Alphakanal hat und ob dort transparente Pixel vorkommen.
    Mehrstufiger Ansatz:
      1) ffprobe: liefert das Pixel-Format → frühes Nein, falls kein Alpha unterstützt wird.
      2) ffmpeg+signalstats: schnelle Textauswertung von YMIN (Alpha-Plane) für einige Frames.
      3) Falls unklar: verkleinerte Rohframes lesen und per Byte-Scan auf <255 prüfen.
    Rückgabe: (has_alpha_channel, has_transparency_or_None)
    """
    try:
        si = _probe_stream_info(path)
        pxfmt = str(si.get("pix_fmt") or "").lower()
        has_alpha = _pixfmt_supports_alpha(pxfmt)
        if not has_alpha:
            return False, False

        import re

        _YMIN_RE = re.compile(
            r"(?:YMIN|lavfi\.signalstats\.YMIN)\s*[:=]\s*(\d+(?:\.\d+)?)", re.I
        )

        def _scan_text(txt: str) -> tuple[bool, Optional[bool]]:
            if not txt:
                return True, None
            found_any = False
            for m in _YMIN_RE.finditer(txt):
                found_any = True
                try:
                    v = float(m.group(1))
                    if v < 255.0:
                        return True, True
                except Exception:
                    pass
            return (True, False if found_any else None)

        cmdA = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-filter_complex",
            "format=rgba,alphaextract,signalstats",
            "-frames:v",
            str(max_frames),
            "-f",
            "null",
            "-",
        ]
        rA = subprocess.run(cmdA, capture_output=True, text=True)
        has_alpha_ch, has_trans = _scan_text((rA.stderr or "") + (rA.stdout or ""))
        if has_trans is not None:
            return has_alpha_ch, has_trans

        cmdB = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-filter_complex",
            "format=rgba,alphaextract,signalstats,"
            "metadata=mode=print:key=lavfi.signalstats.YMIN",
            "-frames:v",
            str(max_frames),
            "-f",
            "null",
            "-",
        ]
        rB = subprocess.run(cmdB, capture_output=True, text=True)
        has_alpha_ch, has_trans = _scan_text((rB.stderr or "") + (rB.stdout or ""))
        if has_trans is not None:
            return has_alpha_ch, has_trans

        frames_to_scan = max(1, min(int(max_frames), 60))
        cmdC = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "format=rgba,alphaextract,scale=64:36",
            "-frames:v",
            str(frames_to_scan),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        rC = subprocess.run(cmdC, capture_output=True)
        alpha_bytes = rC.stdout or b""
        if alpha_bytes:
            return (True, True) if any(b < 255 for b in alpha_bytes) else (True, False)
        return True, None
    except Exception:
        return False, None


def _print_all_tags(
    path: Path, _ignored_tags: dict[str, str], show_all: bool = False
) -> None:
    co.print_value_info("   " + _("filename"), path.stem, "gold")
    try:
        fsize = path.stat().st_size
        co.print_value_info("   " + _("file_size"), _fmt_bytes(fsize), "gold")
    except Exception:
        co.print_value_info("   " + _("file_size"), "?", "gold")

    try:
        fmt_name, fmt_long, duration = _probe_format_info(path)
    except Exception:
        fmt_name, fmt_long, duration = None, None, None
    container = path.suffix.lstrip(".").lower()
    long_disp = (fmt_long or fmt_name or "") or ""
    extra = f" ({long_disp})" if long_disp else ""
    co.print_value_info("   " + _("container"), f"{container}{extra}")
    co.print_value_info("   " + _("duration"), _format_hms(duration))
    try:
        has_thumb = vt.check_thumbnail(path, silent=True)
    except Exception:
        has_thumb = False
    co.print_value_info("   " + _("thumbnail_label"), he.yesno(has_thumb))

    try:
        si = _probe_stream_info(path)
        enc_w = si.get("width")
        enc_h = si.get("height")
        sar = si.get("sample_aspect_ratio") or "1:1"
        dar = si.get("display_aspect_ratio") or f"{enc_w}:{enc_h}"
        fpsr = str(si.get("r_frame_rate") or "0/1")
        codec = si.get("codec_name", "?")
        pxfmt = si.get("pix_fmt", "?")
        co.print_value_info(
            "   " + _("resolution_encoded_label"),
            f"{enc_w if enc_w else '?'}x{enc_h if enc_h else '?'}",
        )
        disp_w = _display_width_from_sar(cast(Optional[int], enc_w), sar)
        if disp_w is not None and enc_h:
            co.print_value_info(
                "   " + _("display_resolution_label"),
                f"{disp_w}x{enc_h}  (SAR {sar}, DAR {dar})",
            )
        co.print_value_info("   " + _("codec_label"), codec or "?")
        co.print_value_info("   " + _("fps_label"), he.format_fps(fpsr))
        co.print_value_info("   " + _("pixel_format_label"), pxfmt or "?")

        # NEW: Farbsignalisierung
        _print_color_info_from_streaminfo(si)

    except Exception:
        co.print_warning(_("no_video_stream"))

    try:
        has_alpha, has_trans = _detect_alpha_content(path, max_frames=200)
        co.print_value_info(
            "   " + _("alpha_channel_label"), _("yes") if has_alpha else _("no")
        )
        if has_alpha:
            co.print_value_info(
                "   " + _("transparency_content_limited_label"),
                (
                    _("yes")
                    if has_trans is True
                    else _("no") if has_trans is False else "?"
                ),
            )
    except Exception:
        co.print_value_info("   " + _("alpha_channel_label"), "?")
        co.print_value_info("   " + _("transparency_content_label"), "?")

    _print_audio_info(path)
    # Audio + Untertitel: erst Anzahl, Listen nur bei --all
    try:
        container = path.suffix.lstrip(".").lower()
        if _container_supports_audio(container):
            a_streams = _probe_audio_streams(path)
            co.print_value_info("   " + _("audio_streams_label"), str(len(a_streams)))
            if show_all:
                _print_audio_streams_list(path)
        if _container_supports_subs(container):
            s_streams = _probe_subtitle_streams(path)
            co.print_value_info(
                "   " + _("subtitle_streams_label"), str(len(s_streams))
            )
            if show_all:
                _print_subtitle_streams_list(path)
    except Exception:
        pass

    # Kapitel: nur Anzahl; Liste nur bei --all
    _print_chapters_count_or_list(path, show_all=show_all)

    raw_tags, canon_edit, _display = _compose_tag_views(path)
    protected_tags, editable_tags, other_tags = _split_tag_groups_display(
        raw_tags, canon_edit
    )

    def _print_tag_block(title: str, kv: dict[str, str], val_color: str) -> None:
        if not kv:
            return
        for k in _sort_by_localized_name(list(kv.keys())):
            label = _tag_display_name(k, show_code=False)
            co.print_value_info(f"   {label}", kv[k], val_color)

    _print_tag_block(_("protected_metadata_tags"), protected_tags, "rose")
    _print_tag_block(_("editable_metadata_tags"), editable_tags, "light_green")
    other_title = _("other_metadata_tags")
    _print_tag_block(other_title, other_tags, "khaki")

    if not (protected_tags or editable_tags or other_tags):
        co.print_info(_("no_metadata_found"))


def _parse_generic_tag_args(
    tag_args: Optional[List[str]],
) -> tuple[Set[str], dict[str, str]]:
    get_keys: Set[str] = set()
    set_map: dict[str, str] = {}
    if not tag_args:
        return get_keys, set_map
    for item in tag_args:
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip().lower()
            set_map[k] = v.strip()
        else:
            get_keys.add(item.strip().lower())
    return get_keys, set_map


def metadata(args: Any) -> None:
    if isinstance(args, type):
        args = args()

    def _bool_attr(*names: str, default: bool = False) -> bool:
        for n in names:
            v = getattr(args, n, None)
            if isinstance(v, bool):
                return v
            if v is not None:
                return bool(v)
        return default

    # Flags
    list_all = _bool_attr("list_tags", default=False)
    list_json = _bool_attr("list_tags_json", default=False)
    list_names = _bool_attr("list_tagnames", "list_tagsname", default=False)
    show_all = _bool_attr("all", default=False)

    # Generisches --tag
    get_keys_tag, set_map_tag = _parse_generic_tag_args(getattr(args, "tag", None))
    print_keys: Set[str] = set(get_keys_tag)
    set_map: dict[str, str] = dict(set_map_tag)
    delete_keys: Set[str] = set()

    # EDITABLE-Keys spezifisch
    for k in getattr(defin, "EDITABLE_META_KEYS", []):
        if getattr(args, f"list_tag_{k}", False):
            print_keys.add(k)
        if getattr(args, f"delete_tag_{k}", False):
            delete_keys.add(k)
        v_alias = getattr(args, f"set_tag_{k}", None)
        if v_alias is not None:
            set_map[k] = v_alias
        v_main = getattr(args, k, None)
        if v_main is not None:
            if v_main == "__READ__":
                print_keys.add(k)
            else:
                set_map[k] = v_main

    # PROTECTED nur lesen
    for pk in getattr(defin, "PROTECTED_META_KEYS", []):
        if getattr(args, f"list_tag_{pk}", False):
            print_keys.add(pk)

    # Virtuelle nur lesen
    for vk in VIRTUAL_QUERY_TAGS:
        if getattr(args, f"list_tag_{vk}", False):
            print_keys.add(vk)

    # Thumb-Operationen
    has_thumb_ops = (
        _bool_attr("delete_thumbnail")
        or _bool_attr("show_thumbnail")
        or (getattr(args, "set_thumbnail", None) is not None)
    )

    cli_files_provided = bool(getattr(args, "files", None))
    any_param_except_names = (
        list_all
        or list_json
        or bool(print_keys)
        or bool(set_map)
        or bool(delete_keys)
        or has_thumb_ops
        or bool(getattr(args, "tag", None))
    )
    BATCH_MODE = (
        any_param_except_names
        or list_names
        or (getattr(args, "_files_source", None) == "cli")
    )

    if any_param_except_names and not cli_files_provided:
        co.print_error(_("passed_no_file"))
        return

    # Nur Namen-Listing (ohne Dateien)
    if list_names and not any_param_except_names:
        editable_keys = list(getattr(defin, "EDITABLE_META_KEYS", []))
        protected_keys = list(getattr(defin, "PROTECTED_META_KEYS", []))
        virtual_keys = list(VIRTUAL_QUERY_TAGS)

        def disp_name(key: str) -> str:
            if key in VIRTUAL_QUERY_TAGS:
                vm = getattr(defin, "VIRTUAL_META_INFO", {}).get(key, {})
                return tr(vm.get("name")) if vm else key
            meta = getattr(defin, "META_TAGS", {}).get(key, {})
            return tr(meta.get("name")) if meta else key

        all_keys = editable_keys + protected_keys + virtual_keys
        key_w = max((co.visible_width(k) for k in all_keys), default=0) + 1
        name_w = max((co.visible_width(disp_name(k)) for k in all_keys), default=0) + 1
        indent = "   "

        co.print_headline(
            "      =========  " + _("editable_metadata_tags") + "  =========\n",
            "bright_green",
        )
        for key in sorted(editable_keys):
            meta = getattr(defin, "META_TAGS", {}).get(key, {})
            name = tr(meta.get("name")) if meta else key
            desc = tr(meta.get("description")) if meta else ""
            key_pad = " " * max(0, key_w - co.visible_width(key) - 1)
            name_pad = " " * max(0, name_w - co.visible_width(name) - 1)
            line = f"{indent}{key}{key_pad} - {name}{name_pad}: "
            co.print_multi_line((line, "light_green", "bild"), (desc, "khaki"))

        if protected_keys:
            print()
            co.print_headline(
                "      =========  " + _("protected_metadata_tags") + "  =========\n",
                "salmon",
            )
            for key in sorted(protected_keys):
                meta = getattr(defin, "META_TAGS", {}).get(key, {})
                name = tr(meta.get("name")) if meta else key
                desc = tr(meta.get("description")) if meta else ""
                key_pad = " " * max(0, key_w - co.visible_width(key) - 1)
                name_pad = " " * max(0, name_w - co.visible_width(name) - 1)
                line = f"{indent}{key}{key_pad} - {name}{name_pad}: "
                co.print_multi_line((line, "rose", "bild"), (desc, "khaki"))

        print()
        co.print_headline(
            "      =========  " + _("virtual_readonly_header") + "  =========\n", "cyan"
        )
        for key in sorted(virtual_keys):
            vm = getattr(defin, "VIRTUAL_META_INFO", {}).get(key, {})
            name = tr(vm.get("name")) if vm else key
            desc = tr(vm.get("description")) if vm else ""
            key_pad = " " * max(0, key_w - co.visible_width(key) - 1)
            name_pad = " " * max(0, name_w - co.visible_width(name) - 1)
            line = f"{indent}{key}{key_pad} - {name}{name_pad}: "
            co.print_multi_line((line, "cyan", "bild"), (desc, "khaki"))
        return

    # Dateien einsammeln
    files_any = fs.prepare_files(args, defin.VIDEO_EXTENSIONS)
    files: List[str] = [str(x) for x in (files_any or [])]
    if not files:
        if list_names and not any_param_except_names:
            pass
        else:
            return

    if BATCH_MODE:
        no_other_ops = (
            not list_all
            and not list_json
            and not list_names
            and not print_keys
            and not set_map
            and not delete_keys
            and not has_thumb_ops
        )
        if no_other_ops:
            list_all = True

    raw_read_only = (
        not list_all
        and not list_json
        and not list_names
        and bool(print_keys)
        and not set_map
        and not delete_keys
    )

    if not BATCH_MODE and not raw_read_only:
        co.print_start(_("metadata_method"))

    json_out: List[dict[str, Any]] = []

    # ====== pro Datei ======
    for file in files:
        path = Path(file)
        raw_tags, canon_edit, display_tags = _compose_tag_views(path)

        # RAW-READ-ONLY: nur Werte, keine Banner
        if raw_read_only:
            for key in sorted(print_keys):
                if key in VIRTUAL_QUERY_TAGS:
                    val = _virtual_tag_value(path, key)
                elif key in _editable_keys():
                    val = canon_edit.get(key, "")
                else:
                    val = raw_tags.get(key, "")
                print("" if val is None else str(val))
            continue

        # Kurz-Thumb-Aktionen
        if getattr(args, "delete_thumbnail", False):
            vt.delete_thumbnail(path, BATCH_MODE=True)
            continue
        val_thumb = getattr(args, "set_thumbnail", None)
        if val_thumb is not None:
            vt.set_thumbnail(path, value=val_thumb, BATCH_MODE=True)
            continue
        if getattr(args, "show_thumbnail", False):
            vt.show_thumbnail(path)
            continue

        should_write = False

        # ========== BATCH ==========
        if BATCH_MODE:
            if list_all:
                _print_all_tags(path, raw_tags, show_all=show_all)

            if print_keys:
                for k in sorted(print_keys):
                    if k in VIRTUAL_QUERY_TAGS:
                        co.print_value_info(
                            f"   {k}", _virtual_tag_value(path, k), "cyan"
                        )
                    elif k in _protected_keys():
                        co.print_value_info(f"   {k}", raw_tags.get(k, ""), "rose")
                    elif k in _editable_keys():
                        co.print_value_info(
                            f"   {k}", canon_edit.get(k, ""), "light_green"
                        )
                    else:
                        co.print_info(f"   {k}: " + _("not_set"))

            if list_json:
                # probes (einmalig pro Datei)
                try:
                    size_bytes = path.stat().st_size
                except Exception:
                    size_bytes = None
                fmt_name, fmt_long, duration = _probe_format_info(path)
                si = _probe_stream_info(path)
                enc_w = si.get("width")
                enc_h = si.get("height")
                sar = si.get("sample_aspect_ratio") or "1:1"
                dar = si.get("display_aspect_ratio") or (
                    f"{enc_w}:{enc_h}" if enc_w and enc_h else None
                )
                fpsr = str(si.get("r_frame_rate") or "0/1")
                codec = si.get("codec_name") or "?"
                pxfmt = si.get("pix_fmt") or "?"
                disp_w = _display_width_from_sar(cast(Optional[int], enc_w), str(sar))
                try:
                    has_thumb = vt.check_thumbnail(path, True)
                except Exception:
                    has_thumb = False
                try:
                    has_alpha, has_trans = _detect_alpha_content(path, max_frames=200)
                except Exception:
                    has_alpha, has_trans = (None, None)

                a_streams = _probe_audio_streams(path)
                s_streams = _probe_subtitle_streams(path)
                ch_list = _probe_chapters(path)

                def _mk_audio_json(s: dict[str, Any]) -> dict[str, Any]:
                    tags = cast(dict, s.get("tags") or {})
                    disp = cast(dict, s.get("disposition") or {})
                    return {
                        "index": s.get("index"),
                        "codec": s.get("codec_name"),
                        "bit_rate": _extract_bitrate_int(s),
                        "channels": s.get("channels"),
                        "channel_layout": s.get("channel_layout"),
                        "sample_rate": s.get("sample_rate"),
                        "language": tags.get("language"),
                        "title": tags.get("title"),
                        "disposition": {k: int(v) for k, v in disp.items() if v},
                    }

                def _mk_sub_json(s: dict[str, Any]) -> dict[str, Any]:
                    tags = cast(dict, s.get("tags") or {})
                    disp = cast(dict, s.get("disposition") or {})
                    return {
                        "index": s.get("index"),
                        "codec": s.get("codec_name"),
                        "language": tags.get("language"),
                        "title": tags.get("title"),
                        "disposition": {
                            k: int(v) for k, v in (disp or {}).items() if v
                        },
                    }

                # Virtuelle Tags bequem als Map (alle bekannten)
                virtual_map = {}
                for vk in VIRTUAL_QUERY_TAGS:
                    try:
                        virtual_map[vk] = _virtual_tag_value(path, vk)
                    except Exception:
                        virtual_map[vk] = ""

                entry = {
                    "file": str(path),
                    "file_name": path.stem,
                    "container": path.suffix.lstrip(".").lower(),
                    "container_long_name": fmt_long or fmt_name,
                    "size_bytes": size_bytes,
                    "size_human": (
                        _fmt_bytes(size_bytes) if size_bytes is not None else None
                    ),
                    "thumbnail": bool(has_thumb),
                    "duration_seconds": duration,
                    "duration_hms": _format_hms(duration),
                    "video": {
                        "width": enc_w,
                        "height": enc_h,
                        "sar": sar,
                        "dar": dar,
                        "display_width": disp_w,
                        "display_resolution": (
                            f"{disp_w}x{enc_h}" if (disp_w and enc_h) else None
                        ),
                        "codec": codec,
                        "fps": he.format_fps(fpsr),
                        "pix_fmt": pxfmt,
                    },
                    "color": {
                        "primaries": _friendly_color_value(
                            "primaries", si.get("color_primaries")
                        ),
                        "trc": _friendly_color_value("trc", si.get("color_transfer")),
                        "matrix": _friendly_color_value(
                            "matrix", si.get("color_space")
                        ),
                        "range": _friendly_color_value("range", si.get("color_range")),
                        "raw": {
                            "color_primaries": si.get("color_primaries"),
                            "color_transfer": si.get("color_transfer"),
                            "color_space": si.get("color_space"),
                            "color_range": si.get("color_range"),
                        },
                    },
                    "alpha": {
                        "has_alpha_channel": (
                            True
                            if has_alpha is True
                            else False if has_alpha is False else None
                        ),
                        "has_transparency_content": (
                            True
                            if has_trans is True
                            else False if has_trans is False else None
                        ),
                    },
                    "audio": {
                        "count": len(a_streams),
                        "summary": _audio_summary(path),  # wie Textausgabe
                        "streams": (
                            [_mk_audio_json(s) for s in a_streams] if show_all else None
                        ),
                    },
                    "subtitles": {
                        "count": len(s_streams),
                        "streams": (
                            [_mk_sub_json(s) for s in s_streams] if show_all else None
                        ),
                    },
                    "chapters": {
                        "count": len(ch_list),
                        "list": ch_list if show_all else None,
                    },
                    "metadata": {
                        "editable": canon_edit,  # kanonisch (schreibbar)
                        "raw": raw_tags,  # roh (container)
                    },
                    "virtual": virtual_map,  # alle virtuellen Tags als Map
                }

                json_out.append(entry)

            only_list = (
                list_all or list_json or list_names or bool(print_keys)
            ) and not (bool(set_map) or bool(delete_keys))
            if only_list:
                continue

            # Mutationen NUR auf kanonischen Editable-Keys
            before = dict(canon_edit)
            for k in list(delete_keys):
                if k in _editable_keys() and k in canon_edit:
                    canon_edit.pop(k, None)
            for k, v in list(set_map.items()):
                if k in _editable_keys():
                    if v is None or v == "":
                        canon_edit.pop(k, None)
                    else:
                        canon_edit[k] = v
            should_write = canon_edit != before

        # ========== INTERAKTIV ==========
        else:
            co.print_value_info("   " + _("filename"), path.stem, "gold")
            try:
                fsize = path.stat().st_size
                co.print_value_info("   " + _("file_size"), _fmt_bytes(fsize), "gold")
            except Exception:
                co.print_value_info("   " + _("file_size"), "?", "gold")

            try:
                co.print_headline(
                    "\n   ====== " + _("video_info_header") + " ======", "bright_blue"
                )
                si = _probe_stream_info(path)
                enc_w = si.get("width")
                enc_h = si.get("height")
                sar = si.get("sample_aspect_ratio") or "1:1"
                dar = si.get("display_aspect_ratio") or f"{enc_w}:{enc_h}"
                fpsr = str(si.get("r_frame_rate") or "0/1")
                codec = si.get("codec_name", "?")
                pxfmt = si.get("pix_fmt", "?")
                co.print_value_info(
                    "   " + _("resolution_encoded_label"),
                    f"{enc_w if enc_w else '?'}x{enc_h if enc_h else '?'}",
                )
                disp_w = _display_width_from_sar(cast(Optional[int], enc_w), sar)
                if disp_w is not None and enc_h:
                    co.print_value_info(
                        "   " + _("display_resolution_label"),
                        f"{disp_w}x{enc_h}  (SAR {sar}, DAR {dar})",
                    )
                fmt_name, fmt_long, duration = _probe_format_info(path)
                container = path.suffix.lstrip(".").lower()
                co.print_value_info("   " + _("duration"), _format_hms(duration))
                long_disp = (fmt_long or fmt_name or "") or ""
                co.print_value_info(
                    "   " + _("container"),
                    f"{container}{' (' + long_disp + ')' if long_disp else ''}",
                )
                co.print_value_info("   " + _("codec_label"), codec or "?")
                co.print_value_info("   " + _("fps_label"), he.format_fps(fpsr))
                co.print_value_info("   " + _("pixel_format_label"), pxfmt or "?")
                _print_color_info_from_streaminfo(si)
            except Exception:
                co.print_warning(_("no_video_stream"))

            try:
                has_alpha, has_trans = _detect_alpha_content(path, max_frames=200)
                co.print_value_info(
                    "   " + _("alpha_channel_label"), _("yes") if has_alpha else _("no")
                )
                if has_alpha:
                    if has_trans is True:
                        co.print_value_info(
                            "   " + _("transparency_content_limited_label"), _("yes")
                        )
                    elif has_trans is False:
                        co.print_value_info(
                            "   " + _("transparency_content_limited_label"), _("no")
                        )
                    else:
                        co.print_value_info(
                            "   " + _("transparency_content_label"), "?"
                        )
            except Exception:
                co.print_value_info("   " + _("alpha_channel_label"), "?")
                co.print_value_info("   " + _("transparency_content_label"), "?")

            _print_audio_info(path)
            try:
                container = path.suffix.lstrip(".").lower()
                if _container_supports_audio(container):
                    a_streams = _probe_audio_streams(path)
                    co.print_value_info(
                        "   " + _("audio_streams_label"), str(len(a_streams))
                    )
                    if show_all:
                        _print_audio_streams_list(path)
                if _container_supports_subs(container):
                    s_streams = _probe_subtitle_streams(path)
                    co.print_value_info(
                        "   " + _("subtitle_streams_label"), str(len(s_streams))
                    )
                    if show_all:
                        _print_subtitle_streams_list(path)
                _print_chapters_count_or_list(path, show_all=show_all)
            except Exception:
                pass

            protected_tags, editable_tags, other_tags = _split_tag_groups_display(
                raw_tags, canon_edit
            )
            if protected_tags:
                print(" ")
                co.print_headline(
                    "   ====== " + _("protected_metadata_tags") + " ======", "salmon"
                )
                for k in _sort_by_localized_name(list(protected_tags.keys())):
                    co.print_value_info(
                        f"   {_tag_display_name(k, show_code=False)}",
                        protected_tags[k],
                        "rose",
                    )
            if editable_tags:
                print(" ")
                co.print_headline(
                    "   ====== " + _("editable_metadata_tags") + " ======",
                    "bright_green",
                )
                for k in _sort_by_localized_name(list(editable_tags.keys())):
                    co.print_value_info(
                        f"   {_tag_display_name(k, show_code=False)}",
                        editable_tags[k],
                        "light_green",
                    )
            if other_tags:
                print(" ")
                other_title = _("other_metadata_tags")
                co.print_headline("   ====== " + other_title + " ======", "cyan")
                for k in _sort_by_localized_name(list(other_tags.keys())):
                    co.print_value_info(
                        f"   {_tag_display_name(k, show_code=False)}",
                        other_tags[k],
                        "khaki",
                    )
            elif not (protected_tags or editable_tags):
                co.print_info(_("no_metadata_found"))

            if not ui.ask_yes_no(_("edit_metadata_prompt"), True, back_option=False):
                co.print_info(_("quit_no_changes"))
                continue

            should_write = False
            while True:
                ekeys = list(_editable_keys())
                set_keys = [k for k in ekeys if k in canon_edit]
                unset_keys = [k for k in ekeys if k not in canon_edit]

                set_options = []
                for k in set_keys:
                    tname = tr(getattr(defin, "META_TAGS", {}).get(k, {}).get("name"))
                    value = canon_edit.get(k, "")
                    set_options.append((k, tname, value))

                unset_options = []
                for k in unset_keys:
                    tname = tr(getattr(defin, "META_TAGS", {}).get(k, {}).get("name"))
                    tdesc = tr(
                        getattr(defin, "META_TAGS", {}).get(k, {}).get("description")
                    )
                    unset_options.append((k, tname, tdesc))

                default_index = 0
                has_thumbnail = vt.check_thumbnail(path, True)
                action, key = ui.ask_user_grouped_tags(
                    prompt=_("select_tag_number"),
                    set_options=set_options,
                    unset_options=unset_options,
                    delete_thumb=has_thumbnail,
                    default_index=default_index,
                    back_button_text=_("exit_and_save") if should_write else _("exit"),
                )

                if action == "thumb":
                    vt.set_thumbnail(str(path))
                    continue
                if action == "del_thumb":
                    vt.delete_thumbnail(str(path), BATCH_MODE=BATCH_MODE)
                    continue
                if action == "exit":
                    break

                if action in {"edit", "add"} and key:
                    name = tr(getattr(defin, "META_TAGS", {}).get(key, {}).get("name"))
                    new_val = input(
                        co.return_promt(
                            "\n   " + _("enter_new_value_for").format(name=name)
                        )
                    ).strip()
                    if action == "edit":
                        if new_val:
                            if canon_edit.get(key) != new_val:
                                canon_edit[key] = new_val
                                should_write = True
                        else:
                            if key in canon_edit:
                                del canon_edit[key]
                                should_write = True
                                co.print_success((_("tag_deleted").format(key=name)))
                    else:  # add
                        if new_val:
                            canon_edit[key] = new_val
                            should_write = True
                    continue

        if not should_write:
            continue

        # ======= Gemeinsamer Schreibpfad: Kanonische Editable nach Datei =======
        if BATCH_MODE and not (set_map or delete_keys):
            continue

        if (BATCH_MODE and (set_map or delete_keys)) or (not BATCH_MODE):
            try:
                # WICHTIG: Nutzung der robusten Schreib-API (AVI-Sanity + XMP handled inside)
                ms.write_editable_metadata(path, path, canon_edit)  # in-place
                ok, missing = ms.verify_metadata_written(path, canon_edit)
                print()
                if ok:
                    co.print_success(
                        _("metadata_sucessfully_updated").format(filename=path.name)
                    )
                else:
                    # Sollte praktisch nicht auftreten, da ms.* bei fehlender Direktschreibbarkeit abbricht
                    co.print_warning(
                        _("metadata_not_fully_written").format(
                            filename=path.name, missing=", ".join(sorted(missing))
                        )
                    )
                print()
            except Exception as e:
                co.print_error(
                    _("metadata_write_failed").format(filename=path.name, error=e)
                )

    if BATCH_MODE and list_json:
        print(json.dumps(json_out, indent=2, ensure_ascii=False))
        return
