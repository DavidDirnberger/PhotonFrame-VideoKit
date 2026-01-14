#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import audio_support as aud
import consoleOutput as co
import definitions as defin
import extract as ex
import fileSystem as fs
import helpers as he
import i18n as _i18n
import info_helpers as ih
import process_wrappers as pw
import userInteraction as ui
import video_pixfmt as vp
import video_thumbnail as vt
import VideoEncodersCodecs as vec
import VideoEncodersCodecs_support as vcs

# local modules
from ffmpeg_perf import autotune_final_cmd

_RES_CUSTOM_RE = re.compile(r"^\s*custom\s*(?:=|:|\s)\s*(\d+)\s*[:x]\s*(\d+)\s*$", re.I)
_RES_PLAIN_RE = re.compile(r"^\s*(\d+)\s*[:x]\s*(\d+)\s*$")
_FPS_RE = re.compile(r"^\s*(?:([0-9]+(?:\.[0-9]+)?)|([0-9]+)\s*/\s*([0-9]+))\s*$")

# ---- Typing helpers ----------------------------------------------------------


class _Translator(Protocol):
    def __call__(self, value: object | None = ...) -> str: ...


_: _Translator = cast(_Translator, _i18n._)
tr: _Translator = cast(_Translator, _i18n.tr)


def _list_str_factory() -> List[str]:
    """Explizit getypte Factory für List[str], damit Pylance kein list[Unknown] annimmt."""
    return []


def _safe_int(val: Any) -> Optional[int]:
    try:
        i = int(str(val))
        return i
    except Exception:
        return None


def _aac_fallback_extras(input_path: Optional[Path]) -> List[str]:
    """
    Liefert konservative AAC-Defaults, wenn aus Preset/Container keine Extras kommen.
    Ziel: sinnvolle Bitrate nach Kanalzahl, um Artefakte zu vermeiden.
    """
    ch: Optional[int] = None
    if input_path:
        try:
            _codec, ch_raw = aud._probe_audio_codec(input_path)
            ch = _safe_int(ch_raw)
        except Exception:
            ch = None
    ch = ch or 2
    if ch <= 2:
        bps = "192k"
    elif ch <= 6:
        bps = "448k"
    else:
        bps = "384k"
    return ["-b:a", bps, "-ac", str(ch)]


def _bitrate_to_kbps(val: str) -> Optional[float]:
    try:
        s = str(val).strip().lower()
        if s.endswith("k"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 1000.0
        return float(s)
    except Exception:
        return None


def _normalize_video_bitrate_choice(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    kbps = _bitrate_to_kbps(s)
    if kbps is None or kbps <= 0:
        return None
    return f"{int(round(kbps))}k"


def _normalize_audio_channel_choice(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s or s in {"keep", "original", "auto", "default", "same", "copy"}:
        return None
    if s in {"stereo", "2.0"}:
        return 2
    if s in {"mono", "1.0"}:
        return 1
    m = re.fullmatch(r"(\d+)\s*[\.:]\s*(\d+)", s)
    if m:
        try:
            base = int(m.group(1))
            ext = int(m.group(2))
        except Exception:
            return None
        if base > 0:
            if ext == 0:
                return base
            if ext == 1:
                return base + 1
        return None
    m = re.fullmatch(r"(\d+)(?:\s*ch)?", s)
    if m:
        ch = int(m.group(1))
        return ch if ch > 0 else None
    return None


def _normalize_subtitle_format(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip().lower().lstrip(".")
    if s in getattr(defin, "SUBTITLE_FORMATS", {}):
        return s
    return None


def _apply_video_bitrate_cap(cmd: List[str], cap: str) -> None:
    kbps = _bitrate_to_kbps(cap)
    if kbps is None or kbps <= 0:
        return
    maxrate = f"{int(round(kbps))}k"
    bufsize = f"{int(round(kbps * 2))}k"
    vec._strip_options(cmd, {"-b:v", "-vb", "-maxrate", "-minrate", "-bufsize"})
    vec._insert_before_output(cmd, ["-maxrate", maxrate, "-bufsize", bufsize])


def _norm_aac_extras(extras: List[str], input_path: Optional[Path]) -> List[str]:
    """
    Normalisiert AAC-Extras auf moderate Bitraten für Kompatibilität (z. B. Jellyfin).
    """
    ch: Optional[int] = None
    if input_path:
        try:
            _codec, ch_raw = aud._probe_audio_codec(input_path)
            ch = _safe_int(ch_raw)
        except Exception:
            ch = None
    ch = ch or 2

    def rec_bitrate() -> str:
        if ch <= 2:
            return "192k"
        if ch <= 6:
            return "320k"
        return "384k"

    out = list(extras)
    # Bitrate clamp/set
    if "-b:a" in out:
        try:
            i = out.index("-b:a")
            if i + 1 < len(out):
                kb = _bitrate_to_kbps(out[i + 1])
                want = _bitrate_to_kbps(rec_bitrate())
                if kb is None or (want is not None and kb > want):
                    out[i + 1] = rec_bitrate()
        except Exception:
            pass
    else:
        out += ["-b:a", rec_bitrate()]

    # Kanalzahl beilegen, falls nicht vorhanden
    if "-ac" not in out and ch:
        out += ["-ac", str(ch)]

    return out


def _find_video_encoder(cmd: List[str]) -> Optional[str]:
    for flag in ("-c:v", "-codec:v"):
        if flag in cmd:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                return str(cmd[idx + 1]).strip()
    return None


def _fallback_to_sw_encoder(cmd: List[str]) -> Optional[List[str]]:
    """Best-effort: wandelt einen HW-Encode-Command in SW (z. B. h264_vaapi → libx264)."""
    enc = _find_video_encoder(cmd)
    if not enc:
        return None
    core, hw = vcs._encoder_core_hw(enc)
    if hw == "sw":
        return None

    fb = defin.CODEC_FALLBACK_POLICY.get(core, [])
    next_sw: Optional[str] = None
    try:
        cur_idx = [e.lower() for e in fb].index(enc.lower())
        for cand in fb[cur_idx + 1 :]:
            _, hw_c = vcs._encoder_core_hw(cand)
            if hw_c == "sw":
                next_sw = cand
                break
    except ValueError:
        for cand in fb:
            _, hw_c = vcs._encoder_core_hw(cand)
            if hw_c == "sw":
                next_sw = cand
                break
    if not next_sw:
        return None

    new_cmd = list(cmd)
    for flag in ("-c:v", "-codec:v"):
        if flag in new_cmd:
            i = new_cmd.index(flag)
            if i + 1 < len(new_cmd):
                new_cmd[i + 1] = next_sw
            break

    # HW-Optionen entfernen
    try:
        vcs._strip_hw_device_options(new_cmd)
    except Exception:
        pass

    # HW-Filter säubern
    if "-vf" in new_cmd:
        i_vf = new_cmd.index("-vf")
        vf_chain = str(new_cmd[i_vf + 1])
        cleaned: List[str] = []
        for part in vf_chain.split(","):
            p = part.strip()
            if not p or p in {"hwupload", "format=nv12"}:
                continue
            if "scale_vaapi" in p:
                p = p.replace("scale_vaapi", "scale")
            cleaned.append(p)
        if cleaned:
            new_cmd[i_vf + 1] = ",".join(cleaned)
        else:
            del new_cmd[i_vf : i_vf + 2]

    try:
        vec.ensure_pre_output_order(new_cmd)
    except Exception:
        pass

    return new_cmd


def _verify_and_embed_thumbnail(
    *,
    output: Path,
    preserved_cover: Optional[Path],
    had_thumb: bool,
) -> None:
    """Stellt sicher, dass ein vorhandenes Thumbnail im Ziel eingebettet ist."""
    if not had_thumb:
        return
    if not preserved_cover or not preserved_cover.exists():
        return

    def _embed() -> None:
        vt.set_thumbnail(output, value=str(preserved_cover), BATCH_MODE=True)

    _embed()
    ok = False
    try:
        ok = vt.check_thumbnail(output, silent=True)
    except Exception:
        ok = False
    if ok:
        return

    # Cleanup + retry
    try:
        vt.delete_thumbnail(output, BATCH_MODE=True)
    except Exception:
        pass
    _embed()


@dataclass
class ConvertArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    preset: Optional[str] = "casual"
    format: Optional[str] = None
    resolution: Optional[str] = "original"
    resolution_scale: Optional[str] = None
    framerate: Optional[str] = "original"
    codec: Optional[str] = None
    pixelformat: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_channel: Optional[str] = None
    video_bitrate: Optional[str] = None
    keep_subtitles: bool = False
    subtitle_format: Optional[str] = None
    output: Optional[str] = None


@dataclass
class _HdrInfo:
    meta: Dict[str, Optional[str]]
    src_w: Optional[int]
    src_h: Optional[int]
    src_pix_fmt: Optional[str]
    is_hdr: bool


# --- kleine, aber wichtige Typ-Helfer ----------------------------------------


def _to_str_list(v: object) -> List[str]:
    """
    Normalisiert beliebige File-Argumente strikt zu List[str].
    Vermeidet Unknown in Comprehensions.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(p) for p in cast(Sequence[object], v) if p is not None]
    return [str(v)]


def _prepare_inputs_typed(args: Any) -> Tuple[bool, List[str]]:
    """
    Typisierter Wrapper um he.prepare_inputs:
    - BATCH_MODE als bool
    - files garantiert als List[str]
    """
    bm_raw, fl_any = he.prepare_inputs(args)
    files_list: List[str] = _to_str_list(fl_any)
    return bool(bm_raw), files_list


def _merge_with_convert_defaults(args: Any) -> ConvertArgs:
    cfg = ConvertArgs()

    # files sauber zu List[str] normalisieren (ohne Unknown in Comprehension)
    if hasattr(args, "files"):
        cfg.files = _to_str_list(getattr(args, "files"))

    # restliche (stringartige) Felder setzen – leere Listen ignorieren
    def _set_str_opt(name: str) -> None:
        if hasattr(args, name):
            val = getattr(args, name)
            if val is None:
                return
            if isinstance(val, list) and val == []:
                return
            setattr(cfg, name, str(val))

    for n in (
        "preset",
        "format",
        "resolution",
        "framerate",
        "codec",
        "pixelformat",
        "audio_codec",
        "audio_channel",
        "video_bitrate",
        "subtitle_format",
        "output",
    ):
        _set_str_opt(n)

    if hasattr(args, "keep_subtitles"):
        cfg.keep_subtitles = bool(getattr(args, "keep_subtitles"))

    cfg.resolution, cfg.resolution_scale = _parse_resolution_arg(cfg.resolution)

    return cfg


def _parse_resolution_arg(val: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Interpretiert '--resolution custom=WxH', 'custom:WxH', 'custom WxH' oder nur 'WxH'
    → ('custom', 'W:H'). Andernfalls → (val or 'original', None).
    """
    if not val:
        return ("original", None)
    s = str(val)
    m = _RES_CUSTOM_RE.match(s)
    if m:
        w, h = m.group(1), m.group(2)
        return ("custom", f"{w}:{h}")
    m2 = _RES_PLAIN_RE.match(s)
    if m2:
        w, h = m2.group(1), m2.group(2)
        return ("custom", f"{w}:{h}")
    return (s, None)


def _parse_framerate_arg(val: Optional[str]) -> Optional[str]:
    """
    Akzeptiert:
      - ganze/float Zahl: '20', '23.976'
      - rational: '24000/1001', '30000/1001'
    Rückgabe: normalisierte Zeichenkette für ffmpeg ('num/den' oder Zahl) oder None bei Ungültigkeit.
    """
    if not val:
        return None
    s = str(val).strip()
    if s.lower() == "original":
        return None
    m = _FPS_RE.match(s)
    if not m:
        return None
    # Fall 1: Dezimalzahl
    if m.group(1):
        try:
            f = float(m.group(1))
            if 1.0 <= f <= 240.0:
                # Schlank formatieren (ohne überflüssige .0)
                txt = f"{f}".rstrip("0").rstrip(".")
                return txt if txt else None
        except Exception:
            return None
        return None
    # Fall 2: Rational num/den
    try:
        num = int(m.group(2))
        den = int(m.group(3))
        if den <= 0:
            return None
        valf = num / den
        if 1.0 <= valf <= 240.0:
            return f"{num}/{den}"
    except Exception:
        return None
    return None


# --- HDR → SDR helpers -------------------------------------------------------
_HDR_AUTO_PRESETS = {"web", "messenger360p", "messenger720p"}

# --- Audio codec helpers -----------------------------------------------------
_AUDIO_CODEC_ALIASES: Dict[str, str] = {
    "libmp3lame": "mp3",
    "libopus": "opus",
    "libvorbis": "vorbis",
    "pcm": "pcm_s16le",
    "copy": "copy",
    "keep": "copy",
    "original": "copy",
    "source": "copy",
    "passthrough": "copy",
}

_AUDIO_CODEC_ENCODERS: Dict[str, str] = {
    "aac": "aac",
    "mp3": "libmp3lame",
    "opus": "libopus",
    "vorbis": "libvorbis",
    "ac3": "ac3",
    "eac3": "eac3",
    "mp2": "mp2",
    "flac": "flac",
    "alac": "alac",
    "pcm_s16le": "pcm_s16le",
    "pcm_s24le": "pcm_s24le",
}

_AUDIO_CODECS_BY_CONTAINER: Dict[str, Tuple[str, ...]] = {
    "mp4": ("aac", "mp3", "ac3", "alac"),
    "m4v": ("aac", "mp3", "ac3", "alac"),
    "mov": ("aac", "alac", "pcm_s16le", "pcm_s24le"),
    "mkv": ("aac", "opus", "vorbis", "mp3", "ac3", "flac", "pcm_s16le"),
    "webm": ("opus", "vorbis"),
    "avi": ("ac3", "mp3", "pcm_s16le"),
    "mpeg": ("mp2", "ac3", "mp3"),
    "mpg": ("mp2", "ac3", "mp3"),
}

_AUDIO_DEFAULT_BY_CONTAINER: Dict[str, str] = {
    "mp4": "aac",
    "m4v": "aac",
    "mov": "aac",
    "mkv": "aac",
    "webm": "opus",
    "avi": "ac3",
    "mpeg": "mp2",
    "mpg": "mp2",
}

_AUDIO_EXTRA_KEYS = {"-b:a", "-ac", "-ar", "-mapping_family"}
_AUDIO_NO_BITRATE = {"pcm_s16le", "pcm_s24le", "alac", "flac"}

# --- Pixel-Format helpers ----------------------------------------------------
_PIX_FMT_LIST_BASE: Tuple[str, ...] = (
    "yuv420p",
    "yuv420p10le",
    "yuv422p",
    "yuv422p10le",
    "yuv444p",
    "yuv444p10le",
    "yuva420p",
    "yuva444p10le",
    "rgba",
)

_PIX_FMT_BY_CONTAINER_LIMITED: Dict[str, Tuple[str, ...]] = {
    # 4:2:0 only containers (plus 10-bit variant)
    "mp4": ("yuv420p", "yuv420p10le"),
    "m4v": ("yuv420p", "yuv420p10le"),
    "webm": ("yuv420p",),
    "mpeg": ("yuv420p",),
    "mpg": ("yuv420p",),
}


def _normalize_pix_fmt_choice(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip().lower()
    if not v or v in {"auto", "default", "best", "keep", "source"}:
        return None
    return v


def _pix_fmt_options_for(
    container: Optional[str], src_pix_fmt: Optional[str], has_alpha: bool
) -> List[str]:
    c = _normalize_container_name(container)
    base = list(_PIX_FMT_BY_CONTAINER_LIMITED.get(c, _PIX_FMT_LIST_BASE))
    if not has_alpha:
        base = [pf for pf in base if not pf.startswith(("yuva", "rgba", "bgra"))]
    # Quelle/nähere Varianten priorisiert vorne einsortieren
    src_pf = (src_pix_fmt or "").lower()
    if src_pf and src_pf in base:
        base = [src_pf] + [b for b in base if b != src_pf]
    seen: set[str] = set()
    out: List[str] = []
    for pf in base:
        if pf in seen:
            continue
        seen.add(pf)
        out.append(pf)
    return out


def _pix_fmt_has_alpha(pf: Optional[str]) -> bool:
    p = (pf or "").lower()
    if not p:
        return False
    if p.startswith(("yuva", "gbrap", "rgba", "bgra", "argb", "abgr")):
        return True
    if p in {"ya8", "ya16be", "ya16le"}:
        return True
    return False


def _strip_pix_fmt_args(cmd: List[str]) -> None:
    vec._strip_options(cmd, {"-pix_fmt"})
    # entferne format=... aus -vf / -filter:v
    for fl in ("-vf", "-filter:v"):
        if fl in cmd:
            i = cmd.index(fl)
            if i + 1 < len(cmd) and isinstance(cmd[i + 1], str):
                vf = str(cmd[i + 1])
                parts = [
                    p.strip()
                    for p in vf.split(",")
                    if p.strip() and not p.strip().lower().startswith("format=")
                ]
                cmd[i + 1] = ",".join(parts)


def _apply_pix_fmt_override(cmd: List[str], pix_fmt: str) -> None:
    if not pix_fmt:
        return
    _strip_pix_fmt_args(cmd)
    vec._insert_before_output(cmd, ["-pix_fmt", pix_fmt])


def _normalize_container_name(container: Optional[str]) -> str:
    c = (container or "").strip().lower()
    return {"mpg": "mpeg", "m4v": "mp4", "matroska": "mkv"}.get(c, c)


def _has_subtitle_streams(path: Path) -> bool:
    try:
        return bool(vec._probe_subtitle_streams(path))
    except Exception:
        return False


def _should_offer_subtitle_export(
    files: List[str], format_choice: Optional[str]
) -> bool:
    if not files:
        return False
    if not format_choice or format_choice == "keep":
        return False
    tgt = _normalize_container_name(format_choice)
    if tgt not in {"mp4", "m4v"}:
        return False
    for f in files:
        cont = _normalize_container_name(vec.detect_container_from_path(Path(f)))
        if cont == "mkv" and _has_subtitle_streams(Path(f)):
            return True
    return False


def _container_drops_subtitles(container: Optional[str]) -> bool:
    c = _normalize_container_name(container)
    return c in {"mp4", "m4v"}


def _normalize_audio_codec_name(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    v = str(val).strip().lower()
    return _AUDIO_CODEC_ALIASES.get(v, v)


def _normalize_audio_codec_choice(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = _normalize_audio_codec_name(val)
    if not v or v in {"auto", "default", "same", "as-is", "asis"}:
        return None
    return v


def _audio_codecs_for_container(container: Optional[str]) -> List[str]:
    c = _normalize_container_name(container)
    opts = _AUDIO_CODECS_BY_CONTAINER.get(c)
    if opts:
        return list(opts)
    return ["aac", "mp3", "opus", "flac"]


def _default_audio_codec_for_container(container: Optional[str]) -> str:
    c = _normalize_container_name(container)
    return _AUDIO_DEFAULT_BY_CONTAINER.get(c, "aac")


def _audio_encoder_for_choice(choice: str) -> str:
    return _AUDIO_CODEC_ENCODERS.get(choice, choice)


def _audio_codec_allowed(container: Optional[str], codec: Optional[str]) -> bool:
    c = _normalize_container_name(container)
    k = _normalize_audio_codec_name(codec)
    if not k:
        return True
    allowed = _AUDIO_CODECS_BY_CONTAINER.get(c)
    if not allowed:
        return True
    if k in allowed:
        return True
    if c in {"mkv", "matroska"}:
        return True
    return False


def _extract_audio_extras(base_args: Sequence[str], codec_choice: str) -> List[str]:
    extras: List[str] = []
    i = 0
    while i < len(base_args):
        key = str(base_args[i])
        if key in _AUDIO_EXTRA_KEYS and i + 1 < len(base_args):
            val = str(base_args[i + 1])
            if key == "-mapping_family" and codec_choice != "opus":
                i += 2
                continue
            if key == "-b:a" and codec_choice in _AUDIO_NO_BITRATE:
                i += 2
                continue
            extras += [key, val]
            i += 2
            continue
        i += 1
    return extras


def _strip_audio_args(cmd: List[str]) -> None:
    vec._strip_options(
        cmd,
        {
            "-c:a",
            "-acodec",
            "-b:a",
            "-ab",
            "-ac",
            "-ar",
            "-q:a",
            "-af",
            "-filter:a",
            "-mapping_family",
        },
    )


def _cap_audio_bitrate_for_preset(
    extras: List[str], preset_name: Optional[str]
) -> List[str]:
    """Begrenzt Bitraten konservativ je nach Preset."""
    max_kbps_by_preset = {
        "ultra": 320.0,
        "studio": 320.0,
        "casual": 256.0,
        "fast": 192.0,
        "web": 192.0,
        "messenger360p": 128.0,
        "messenger720p": 160.0,
    }
    max_kbps = max_kbps_by_preset.get((preset_name or "").lower(), 256.0)
    out = list(extras)
    if "-b:a" in out:
        try:
            i = out.index("-b:a")
            if i + 1 < len(out):
                kb = _bitrate_to_kbps(out[i + 1])
                if kb is None or kb > max_kbps:
                    out[i + 1] = f"{int(max_kbps)}k"
        except Exception:
            pass
    else:
        out += ["-b:a", f"{int(max_kbps)}k"]
    return out


def _apply_audio_codec_override(
    cmd: List[str],
    *,
    audio_choice: str,
    target_container: str,
    preset_name: Optional[str],
    input_path: Path,
    force_channels: Optional[int] = None,
) -> None:
    cont = _normalize_container_name(target_container)
    if audio_choice == "copy":
        _strip_audio_args(cmd)
        vec._insert_before_output(cmd, ["-c:a", "copy"])
        return

    base_args = vec.build_audio_args(cont, preset_name, input_path)
    if "-c:a" in base_args:
        i = base_args.index("-c:a")
        if i + 1 < len(base_args) and str(base_args[i + 1]) == "copy":
            base_args = vec.build_audio_args(cont, preset_name, None)

    extras = _extract_audio_extras(base_args, audio_choice)
    if audio_choice.startswith("aac"):
        if not extras:
            extras = _aac_fallback_extras(input_path)
        extras = _norm_aac_extras(extras, input_path)
    # Downmix erzwingen, falls gewünscht
    if force_channels and force_channels > 0:
        if "-ac" in extras:
            try:
                idx_ac = extras.index("-ac")
                if idx_ac + 1 < len(extras):
                    extras[idx_ac + 1] = str(int(force_channels))
                else:
                    extras += ["-ac", str(int(force_channels))]
            except Exception:
                extras += ["-ac", str(int(force_channels))]
        else:
            extras += ["-ac", str(int(force_channels))]
    # Bitrate-Preset-Limit
    extras = _cap_audio_bitrate_for_preset(extras, preset_name)
    _strip_audio_args(cmd)
    vec._insert_before_output(
        cmd, ["-c:a", _audio_encoder_for_choice(audio_choice), *extras]
    )


@lru_cache(maxsize=1)
def _hdr_tonemap_filters_available() -> bool:
    return vec.has_filter("zscale") and vec.has_filter("tonemap")


def _probe_hdr_info(path: Path) -> _HdrInfo:
    src_w, src_h, src_pf = vec.ffprobe_geometry(path)
    meta = vp.probe_color_metadata(path)
    return _HdrInfo(
        meta=meta,
        src_w=src_w,
        src_h=src_h,
        src_pix_fmt=src_pf,
        is_hdr=vp.is_hdr_signal(meta, src_pf),
    )


def _get_hdr_info(path: Path, cache: Dict[str, _HdrInfo]) -> _HdrInfo:
    key = str(path)
    info = cache.get(key)
    if info is None:
        info = _probe_hdr_info(path)
        cache[key] = info
    return info


def _detect_any_hdr(files: List[str], cache: Dict[str, _HdrInfo]) -> bool:
    for f in files:
        p = Path(f).resolve()
        if not p.exists():
            continue
        if _get_hdr_info(p, cache).is_hdr:
            return True
    return False


def _norm_hdr_primaries(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    if v in {"bt2020", "bt709", "smpte170m", "smpte240m", "bt470bg"}:
        return v
    return "bt2020"


def _norm_hdr_transfer(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    if v == "pq":
        return "smpte2084"
    if v == "hlg":
        return "arib-std-b67"
    if v in {
        "smpte2084",
        "arib-std-b67",
        "bt709",
        "smpte170m",
        "smpte240m",
        "iec61966-2-1",
        "iec61966-2-4",
        "bt2020-10",
        "bt2020-12",
    }:
        return v
    return "smpte2084"


def _norm_hdr_matrix(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    if v in {"bt2020nc", "bt2020ncl"}:
        return "bt2020nc"
    if v in {"bt2020c", "bt2020cl"}:
        return "bt2020c"
    if v in {"bt709", "smpte170m", "smpte240m", "bt470bg", "fcc", "ycgco"}:
        return v
    return "bt2020nc"


def _norm_hdr_range(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    if v in {"pc", "full", "jpeg"}:
        return "full"
    return "limited"


def _build_hdr_to_sdr_chain(meta: Dict[str, Optional[str]]) -> Optional[str]:
    if not _hdr_tonemap_filters_available():
        return None
    prim = _norm_hdr_primaries(meta.get("color_primaries"))
    trc = _norm_hdr_transfer(meta.get("color_trc"))
    mat = _norm_hdr_matrix(meta.get("colorspace"))
    rng = _norm_hdr_range(meta.get("color_range"))
    return (
        "zscale=transfer=linear:npl=100:"
        f"primariesin={prim}:transferin={trc}:matrixin={mat}:rangein={rng},"
        "format=gbrpf32le,"
        "tonemap=tonemap=hable:desat=0,"
        "zscale=transfer=bt709:primaries=bt709:matrix=bt709:range=limited"
    )


def _set_vf_chain(cmd: List[str], vf: str) -> None:
    if "-vf" in cmd:
        i = cmd.index("-vf")
        if i + 1 < len(cmd):
            cmd[i + 1] = vf
        else:
            cmd.insert(i + 1, vf)
        return
    insert_at = max(len(cmd) - 1, 0)
    cmd[insert_at:insert_at] = ["-vf", vf]


def _strip_color_metadata_args(cmd: List[str]) -> None:
    keys = {"-color_primaries", "-color_trc", "-colorspace", "-color_range"}
    i = 0
    while i < len(cmd):
        if cmd[i] in keys:
            del cmd[i : i + 2]
            continue
        i += 1


def _strip_x264_color_params(cmd: List[str]) -> None:
    if "-x264-params" not in cmd:
        return
    i = cmd.index("-x264-params")
    if i + 1 >= len(cmd):
        return
    raw = str(cmd[i + 1])
    parts = [
        p
        for p in raw.split(":")
        if p
        and not p.startswith(("colorprim=", "transfer=", "colormatrix=", "fullrange="))
    ]
    if parts:
        cmd[i + 1] = ":".join(parts)
    else:
        del cmd[i : i + 2]


def _hdr_param_value(
    hdr_found: bool, hdr_to_sdr: bool, mode: Optional[str]
) -> Optional[Any]:
    if not hdr_found:
        return None
    if hdr_to_sdr:
        if mode == "auto":
            return {"de": "Automatisch (BT.709)", "en": "Auto (BT.709)"}
        return {"de": "Ja (BT.709)", "en": "Yes (BT.709)"}
    return {"de": "Nein", "en": "No"}


# --- Lossless Cache -----------------------------------------------------------
_LOSSLESS_CACHE: Dict[str, Dict[str, Dict[str, List[str]]]] = {}


class _PlanAdapter:
    """Adapter, damit {final_cmd_without_output: [...]} erwartet werden kann."""

    def __init__(self, tokens: Sequence[Any]):
        self.final_cmd_without_output: List[str] = [str(t) for t in tokens]


T = TypeVar("T")


def _first_of_iterable(seq: Iterable[T]) -> Optional[T]:
    """
    Gibt das erste Element der Iteration zurück oder None, wenn leer.
    (Kein `for x in seq` -> verhindert 'Type of x is unknown')
    """
    it = iter(seq)
    try:
        return next(it)
    except StopIteration:
        return None


def _normalize_encoder_candidate(val: Any) -> Any:
    """
    Normalisiert beliebige Encoder-Kandidaten (Listen/Tuples/Dicts/Objekte).
    """
    v: Any = val
    while isinstance(v, (list, tuple)):
        if not v:
            return None
        nxt: Optional[Any] = _first_of_iterable(cast(Iterable[Any], v))
        if nxt is None:
            return None
        v = nxt

    if isinstance(v, dict):
        picked: Optional[Any] = None
        for _k, _vv in cast(Mapping[Any, Any], v).items():
            norm = _normalize_encoder_candidate(_vv)
            if norm is not None:
                picked = norm
                break
        if picked is None:
            return None
        v = picked

    if hasattr(v, "final_cmd_without_output"):
        return v

    if isinstance(v, (list, tuple)):
        return _PlanAdapter(cast(Sequence[Any], v))

    return v


def _lossless_cache_key(files: List[str], preset: str) -> str:
    first = str(Path(files[0]).resolve()) if files else ""
    try:
        mtime = int(Path(files[0]).stat().st_mtime)
    except Exception:
        mtime = 0
    return f"{preset}|{first}|{len(files)}|{mtime}"


def _compute_and_cache_lossless_for_all_containers(
    files: List[str],
    format_keys: List[str],
    preset: str = "lossless",
) -> Dict[str, Dict[str, List[str]]]:
    key = _lossless_cache_key(files, preset)
    cached = _LOSSLESS_CACHE.get(key)
    if cached is not None:
        return cached

    if not files:
        _LOSSLESS_CACHE[key] = {}
        _LOSSLESS_CACHE["_last"] = {}
        return {}

    container_to_codecmap: Dict[str, OrderedDict[str, Any]] = OrderedDict()
    for fk in format_keys:
        try:
            emap = cast(
                Dict[str, OrderedDict[str, Any]],
                vec.prepare_encoder_maps(
                    files,
                    fk,
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                ),
            )
        except Exception:
            emap = {}

        for cont, cmap in (emap or {}).items():
            if cont not in container_to_codecmap:
                container_to_codecmap[cont] = OrderedDict(cmap)
            else:
                for ck, v in cmap.items():
                    container_to_codecmap[cont].setdefault(ck, v)

    if not container_to_codecmap:
        try:
            emap_keep = cast(
                Dict[str, OrderedDict[str, Any]],
                vec.prepare_encoder_maps(
                    files,
                    "keep",
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                ),
            )
            if emap_keep:
                container_to_codecmap.update(emap_keep)
        except Exception as e:
            co.print_error(
                _("prepare_encoder_maps_fallback_keep_failed").format(error=str(e))
            )

    if not container_to_codecmap:
        _LOSSLESS_CACHE[key] = {}
        _LOSSLESS_CACHE["_last"] = {}
        return {}

    squashed: Dict[str, OrderedDict[str, Any]] = OrderedDict()
    for cont, cmap in container_to_codecmap.items():
        out: OrderedDict[str, Any] = OrderedDict()
        for ck, raw in cmap.items():
            cand = _normalize_encoder_candidate(raw)
            if cand is None:
                continue
            if isinstance(cand, (list, tuple)):
                cand = _PlanAdapter(cand)
            out[ck] = cand
        squashed[cont] = out

    try:
        matrix: Dict[str, Dict[str, List[str]]] = vec.find_lossless_combinations(
            input_path=Path(files[0]),
            container_to_codecmap=squashed,
            req_scale=None,
            user_fps_rational=None,
            preset_max_fps=None,
        )
    except Exception:
        matrix = {}

    ALIASES = {"m4v": "mp4"}
    for alias, target in ALIASES.items():
        if alias in format_keys and alias not in matrix and target in matrix:
            matrix[alias] = matrix[target]

    _LOSSLESS_CACHE[key] = matrix
    _LOSSLESS_CACHE["_last"] = matrix
    return matrix


# --- Tabellen-Param-Building --------------------------------------------------
def _build_params_for_table_interactive(
    *,
    files: List[str],
    preset_choice: Optional[str],
    format_choice: Optional[str],
    resolution_key: Optional[str],
    resolution_scale: Optional[str],
    codec_choice_by_container: Mapping[str, str],
    presets: Mapping[str, Mapping[str, Any]],
    res_defs: Mapping[str, Mapping[str, Any]],
    user_fps_rational: Optional[str] = None,
    target_pix_fmt: Optional[str] = None,
    profile_hint: Optional[str] = None,
    hdr_to_sdr: Optional[Any] = None,
    audio_codec_choice: Optional[str] = None,
    audio_channels: Optional[int] = None,
    video_bitrate_cap: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _target_container() -> Optional[str]:
        if format_choice != "keep":
            return (format_choice or "").lower() if format_choice else None
        return vec.detect_container_from_path(Path(files[0])) if files else None

    def _resolution_display(res_key: Optional[str], scale: Optional[str]) -> Any:
        if res_key == "custom":
            return {
                "de": f"Benutzerdefiniert ({scale or '—'})",
                "en": f"Custom ({scale or '—'})",
            }
        if res_key in ("original", None):
            return {"de": "Original", "en": "Original"}
        name = (res_defs.get(res_key or "", {}) or {}).get(
            "name", res_key or "original"
        )
        return name

    def _fr_label(preset_name: Optional[str], user_cap: Optional[str]) -> Any:
        if user_cap:
            return {
                "de": f"Begrenzen auf {user_cap} fps",
                "en": f"Cap at {user_cap} fps",
            }
        max_fps = (presets.get(preset_name or "", {}) or {}).get("max_fps")
        if max_fps:
            return f"≤{max_fps}"
        return {"de": "Original", "en": "Original"}

    def _summarize_codec_choice_for_table(
        fmt_choice: Optional[str],
        files_list: List[str],
        by_container: Mapping[str, str],
    ) -> str:
        def _detect(p: str) -> Optional[str]:
            return vec.detect_container_from_path(Path(p))

        if fmt_choice != "keep":
            cont = (fmt_choice or "").lower()
            ck = by_container.get(cont, "copy")
            return ck
        seen: List[str] = []
        parts: List[str] = []
        for f in files_list:
            c = _detect(f)
            if c and c not in seen:
                seen.append(c)
                cc = cast(str, c)
                parts.append(f"{cc}:{by_container.get(cc, 'copy')}")
        return ", ".join(parts) if parts else "copy"

    def _humanize_audio(container: Optional[str]) -> str:
        c = (container or "").lower()
        if c in ("mp4", "mov", "m4v"):
            return "aac 128k stereo"
        if c == "webm":
            return "opus 112k stereo"
        return "copy"

    def _audio_display(
        container: Optional[str], choice: Optional[str]
    ) -> Optional[Any]:
        if choice is None:
            return _humanize_audio(container) if container else None
        if choice == "copy":
            return {"de": "Original (copy)", "en": "Original (copy)"}
        return choice

    def _audio_channels_display(ch: Optional[int]) -> Optional[Any]:
        if not ch:
            return None
        if ch == 1:
            return {"de": "Mono (1.0)", "en": "Mono (1.0)"}
        if ch == 2:
            return {"de": "Stereo (2.0)", "en": "Stereo (2.0)"}
        return str(ch)

    tgt_container = _target_container()

    preset_key = preset_choice or ""
    format_key = format_choice or ""

    preset_val = (presets.get(preset_key, {}) or {}).get("name", preset_choice)
    format_val = (defin.CONVERT_FORMAT_DESCRIPTIONS.get(format_key, {}) or {}).get(
        "name", format_choice
    )

    if resolution_key == "custom":
        scale_val = resolution_scale
    elif resolution_key in ("original", None):
        scale_val = None
    else:
        scale_val = (res_defs.get(resolution_key or "", {}) or {}).get("scale")

    resolution_val = _resolution_display(resolution_key, scale_val)
    video_codec_val = _summarize_codec_choice_for_table(
        format_choice, files, codec_choice_by_container
    )
    framerate_val = _fr_label(preset_choice, user_fps_rational)
    audio_codec_val = _audio_display(tgt_container, audio_codec_choice)
    audio_channels_val = _audio_channels_display(audio_channels)
    faststart_on = bool(
        (presets.get(preset_key, {}) or {}).get("faststart") and tgt_container == "mp4"
    )

    params: Dict[str, Any] = {
        "files": [str(Path(f)) for f in files],
        "preset": preset_val,
        "format": format_val,
        "video_codec": video_codec_val,
        "resolution": resolution_val,
        "framerate": framerate_val,
    }
    if scale_val:
        params["scale"] = scale_val
    if audio_codec_val:
        params["audio_codec"] = audio_codec_val
    if audio_channels_val:
        params["audio_channels"] = audio_channels_val
    if video_bitrate_cap:
        params["video_bitrate"] = f"cap {video_bitrate_cap}"
    if faststart_on:
        params["faststart"] = {"de": "an", "en": "on"}
    if target_pix_fmt:
        params["target_pix_fmt"] = target_pix_fmt
    if profile_hint:
        params["profile"] = profile_hint
    if hdr_to_sdr is not None:
        params["hdr_to_sdr"] = hdr_to_sdr

    labels: Dict[str, Any] = {
        "preset": {"de": "Voreinstellung", "en": "Preset"},
        "format": {"de": "Format", "en": "Format"},
        "video_codec": {"de": "Video-Codec", "en": "Video codec"},
        "resolution": {"de": "Zielauflösung", "en": "Target resolution"},
        "scale": {"de": "Skalierung", "en": "Scale"},
        "framerate": {"de": "Bildrate", "en": "Framerate"},
        "audio_codec": {"de": "Audio-Codec", "en": "Audio codec"},
        "audio_channels": {"de": "Audio-Kanaele", "en": "Audio channels"},
        "video_bitrate": {"de": "Video-Bitrate (Cap)", "en": "Video bitrate (cap)"},
        "faststart": {"de": "Faststart (MP4)", "en": "Faststart (MP4)"},
        "target_pix_fmt": {"de": "Pixelformat", "en": "Pixel format"},
        "profile": {"de": "Profil", "en": "Profile"},
        "hdr_to_sdr": {"de": "HDR → SDR (BT.709)", "en": "HDR → SDR (BT.709)"},
    }
    return params, labels


def _plan_to_cmd(
    plan: Any,
    *,
    input_path: Path,
    output: Path,
    target_container: str,
    preset_name: str,
) -> List[str]:
    """
    Erzeugt eine komplette ffmpeg-CLI aus Plan-Objekt oder Token-Fragment.
    """
    if isinstance(plan, (list, tuple)):
        tokens: List[str] = [str(t) for t in cast(Sequence[Any], plan)]
        if tokens and "ffmpeg" in tokens[0]:
            return tokens + [str(output)]

        base = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-stats_period",
            "0.5",
            "-i",
            str(input_path),
        ]
        try:
            base += vec.build_stream_mapping_args(target_container)
        except Exception:
            base += ["-map", "0:v:0", "-map", "0:a?"]
        try:
            base += vec.build_audio_args(target_container, preset_name, input_path)
        except Exception:
            base += ["-c:a", "copy"]

        return base + tokens + [str(output)]

    return vec.assemble_ffmpeg_cmd(plan, output)


# --- Lossless-Badges ----------------------------------------------------------
def _badge(color: str) -> str:
    return {"green": "🟢", "blue": "🔵"}.get(color, "")


def _lossless_mark_label(base_label: str, *, copy_ok: bool, strict_ok: bool) -> str:
    badges: List[str] = []
    if copy_ok:
        badges.append(_badge("green"))
    if strict_ok:
        badges.append(_badge("blue"))
    suffix = " ".join(b for b in badges if b)
    label = f"{base_label} {suffix}" if suffix else base_label
    color = "khaki"
    return co.return_promt(label, color=color)


def convert(args: Any) -> None:
    # 0) Inputs (mit voll typisiertem Wrapper) – bewusst ohne Tuple-Unpacking
    _inp: Tuple[bool, List[str]] = _prepare_inputs_typed(args)
    BATCH_MODE: bool = _inp[0]
    files: List[str] = _inp[1]

    co.print_start(_("converting_method"))

    # Einheitliche Typen für beide Pfade
    preset_choice: Optional[str] = None
    format_choice: Optional[str] = None
    resolution: Optional[str] = None
    framerate: str = "original"
    resolution_scale: Optional[str] = None
    user_fps_rational: Optional[str] = None

    encoder_maps_by_container: Dict[str, OrderedDict[str, Any]] = OrderedDict()
    codec_choice_by_container: Dict[str, str] = {}
    hdr_cache: Dict[str, _HdrInfo] = {}
    hdr_to_sdr: bool = False
    hdr_mode: Optional[str] = None
    audio_codec_choice: Optional[str] = None
    pix_fmt_choice: Optional[str] = None
    force_audio_channels: Optional[int] = None
    video_bitrate_choice: Optional[str] = None
    keep_subtitles: bool = False
    subtitle_format: Optional[str] = None
    src_audio_codec: Optional[str] = None
    src_audio_channels: Optional[int] = None
    src_pix_fmt_first: Optional[str] = None

    first_input = Path(files[0]).resolve() if files else None
    if first_input and first_input.exists():
        try:
            a_codec, a_ch = aud._probe_audio_codec(first_input)
            src_audio_codec = a_codec
            src_audio_channels = _safe_int(a_ch)
        except Exception:
            pass
        try:
            src_pix_fmt_first = _get_hdr_info(first_input, hdr_cache).src_pix_fmt
        except Exception:
            src_pix_fmt_first = None

    # 1) Parameter
    if BATCH_MODE:
        cfg = _merge_with_convert_defaults(args)
        preset_choice = cfg.preset or "casual"
        format_choice = cfg.format or "keep"
        resolution = cfg.resolution or "original"
        framerate = cfg.framerate or "original"  # spätere Anzeige/Suffix
        codec_key_def = vec.normalize_codec_key(cfg.codec) or "h264"
        audio_codec_choice = _normalize_audio_codec_choice(cfg.audio_codec)
        force_audio_channels = _normalize_audio_channel_choice(cfg.audio_channel)
        pix_fmt_choice = _normalize_pix_fmt_choice(getattr(cfg, "pixelformat", None))
        video_bitrate_choice = _normalize_video_bitrate_choice(cfg.video_bitrate)
        keep_subtitles = bool(getattr(cfg, "keep_subtitles", False))
        subtitle_format = _normalize_subtitle_format(cfg.subtitle_format) or "srt"
        if (
            keep_subtitles
            and cfg.subtitle_format
            and not _normalize_subtitle_format(cfg.subtitle_format)
        ):
            co.print_warning(
                _("invalid_sub_format").format(fmt=str(cfg.subtitle_format))
            )
        fps_norm = _parse_framerate_arg(framerate)
        user_fps_rational = fps_norm

        if force_audio_channels and audio_codec_choice in (None, "copy"):
            if format_choice != "keep":
                audio_cont = format_choice
            else:
                audio_cont = (
                    vec.detect_container_from_path(Path(files[0])) if files else None
                )
            audio_codec_choice = _default_audio_codec_for_container(audio_cont)

        # container/codec aus deiner CLI
        warnings, errors, suggestions = vec.plan_warnings_and_errors_for_choice(
            input_path=Path(files[0]),
            target_container=format_choice,  # z.B. "mp4", "mkv", ...
            codec_key=codec_key_def,  # z.B. "h264", "prores", ...
            convert_format_descriptions=defin.CONVERT_FORMAT_DESCRIPTIONS,
            encoder_name=None,  # optional, sonst aus Codec abgeleitet
        )

        if errors:
            for e in errors:
                co.print_error(e)
            return

        for w in warnings:
            co.print_warning(w)  # oder deine i18n-Ausgabe

        # Optional: Alternativen anzeigen (Container, Codec)
        if suggestions:
            pretty = ", ".join([f"{c}+{k}" for (c, k) in suggestions[:6]])
            co.print_info(_("alpha_alternative_suggestions").format(list=pretty))

        if framerate != "original" and user_fps_rational is None:
            co.print_warning(_("invalid_framerate_arg").format(value=str(framerate)))
            framerate = "original"

        resolution_scale = cfg.resolution_scale

        if resolution == "custom" and resolution_scale:
            try:
                min_src_w, min_src_h = vec.probe_min_source_geometry(cfg.files or files)
                ok, msg = vec.validate_custom_scale_leq(
                    min_src_w, min_src_h, resolution_scale, require_even=True
                )
                if not ok:
                    co.print_warning(msg)
                    resolution = "original"
                    resolution_scale = None
            except Exception as e:
                co.print_warning(
                    _("custom_resolution_validation_failed").format(error=str(e))
                )
                resolution = "original"

        encoder_maps_by_container = cast(
            Dict[str, OrderedDict[str, Any]],
            vec.prepare_encoder_maps(
                files,
                format_choice,
                defin.CONVERT_FORMAT_DESCRIPTIONS,
                prefer_hw=True,
                ffmpeg_bin="ffmpeg",
            ),
        )
        for c, enc_map in encoder_maps_by_container.items():
            if codec_key_def == "copy":
                codec_choice_by_container[c] = "copy"
            elif codec_key_def in enc_map or vec.container_allows_codec(
                c, codec_key_def
            ):
                # User-Wunsch respektieren, auch wenn prepare_encoder_maps ihn nicht gelistet hat
                codec_choice_by_container[c] = codec_key_def
            elif "h264" in enc_map:
                codec_choice_by_container[c] = "h264"
            else:
                codec_choice_by_container[c] = next(iter(enc_map.keys()))

        hdr_found = _detect_any_hdr(files, hdr_cache)
        if hdr_found:
            if (preset_choice or "") in _HDR_AUTO_PRESETS:
                hdr_to_sdr = True
                hdr_mode = "auto"
            else:
                hdr_to_sdr = bool(getattr(args, "hdr_to_sdr", False))
                hdr_mode = "manual" if hdr_to_sdr else "off"

        if hdr_to_sdr and not _hdr_tonemap_filters_available():
            co.print_warning(_("hdr_to_sdr_filters_missing"))
            hdr_to_sdr = False
            hdr_mode = "off"

        hdr_param = _hdr_param_value(hdr_found, hdr_to_sdr, hdr_mode)

        # resolution_scale = None

        params, labels = _build_params_for_table_interactive(
            files=files,
            preset_choice=preset_choice,
            format_choice=format_choice,
            resolution_key=resolution,
            resolution_scale=resolution_scale,
            codec_choice_by_container=codec_choice_by_container,
            presets=defin.CONVERT_PRESET,
            res_defs=defin.RESOLUTIONS,
            user_fps_rational=user_fps_rational,
            hdr_to_sdr=hdr_param,
            target_pix_fmt=pix_fmt_choice,
            audio_codec_choice=audio_codec_choice,
            audio_channels=force_audio_channels,
            video_bitrate_cap=video_bitrate_choice,
        )
    else:
        while True:
            # Preset
            preset_keys = list(defin.CONVERT_PRESET)
            preset_descriptions = [
                str(tr(defin.CONVERT_PRESET[k].get("description") or ""))
                for k in preset_keys
            ]
            preset_labels = [
                str(defin.CONVERT_PRESET[k].get("name", k)) for k in preset_keys
            ]
            preset_choice = ui.ask_user(
                _("choose_quality_preset"),
                preset_keys,
                preset_descriptions,
                3,
                preset_labels,
                back_button=False,
            )
            if preset_choice is None:
                continue

            # Infoblöcke
            info = ih.general_info_for_preset(preset_choice)
            print()
            co.print_headline(str(tr(info["headline"])), "orchid")
            for block in info["blocks"]:
                co.print_line(
                    str(tr(block["title"])), color="bright_orange", style="bold"
                )
                names = [
                    str(b.get("name", ""))
                    for b in block.get("bullets", [])
                    if isinstance(b, dict)
                ]
                description = [
                    (
                        str(tr(b.get("description")))
                        if b.get("description") is not None
                        else ""
                    )
                    for b in block.get("bullets", [])
                    if isinstance(b, dict)
                ]
                co.print_bullet_list(names, description)
            print()

            # Format-Auswahl (inkl. Lossless-Badges)
            format_keys = list(defin.CONVERT_FORMAT_DESCRIPTIONS)
            format_descriptions = [
                str(tr(defin.CONVERT_FORMAT_DESCRIPTIONS[k].get("description") or ""))
                for k in format_keys
            ]

            lossless_matrix: Dict[str, Dict[str, List[str]]] = {}

            if preset_choice == "lossless":
                print()
                co.print_text(str(tr(ih.LOSSLESS_DISCLAIMER)))
                print()
                co.print_line("    " + _badge("green") + " " + _("bitstream_copy"))
                co.print_line("    " + _badge("blue") + " " + _("strict_copy"))
                print()
                co.print_line("    " + _("analysing_lossless"), color="gold")

                lossless_all = _compute_and_cache_lossless_for_all_containers(
                    files, format_keys, preset_choice
                )

                format_labels: List[str] = []
                for fk in format_keys:
                    base_label = str(
                        tr(defin.CONVERT_FORMAT_DESCRIPTIONS[fk].get("name", fk))
                    )
                    if fk == "keep":
                        has_copy = any(
                            bool((lossless_all.get(c, {}).get("copy")))
                            for c in (lossless_all or {})
                        )
                        has_strict = any(
                            bool((lossless_all.get(c, {}).get("strict")))
                            for c in (lossless_all or {})
                        )
                    else:
                        mtx = lossless_all.get(fk, {})
                        has_copy = bool(mtx.get("copy"))
                        has_strict = bool(mtx.get("strict"))
                    format_labels.append(
                        _lossless_mark_label(
                            base_label, copy_ok=has_copy, strict_ok=has_strict
                        )
                    )
                co.delete_last_n_lines(2)
            else:
                format_labels = [
                    str(tr(defin.CONVERT_FORMAT_DESCRIPTIONS[fk].get("name", fk)))
                    for fk in format_keys
                ]

            format_choice = ui.ask_user(
                _("choose_format"), format_keys, format_descriptions, 0, format_labels
            )
            if format_choice is None:
                continue

            # Encoder-Map
            encoder_maps_by_container = cast(
                Dict[str, OrderedDict[str, Any]],
                vec.prepare_encoder_maps(
                    files,
                    format_choice,
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                ),
            )

            # Lossless-Matrix (nur für Labeling/Infos)
            if preset_choice == "lossless":
                _cache_key = _lossless_cache_key(files, preset_choice)
                lossless_matrix = (
                    _LOSSLESS_CACHE.get(_cache_key)
                    or _LOSSLESS_CACHE.get("_last")
                    or {}
                )

            # Codec je Container
            codec_choice_by_container = {}
            ans: Optional[str] = None
            for c in encoder_maps_by_container.keys():
                encoders_codec_keys: List[str] = list(
                    encoder_maps_by_container[c].keys()
                )
                codec_descriptions = [
                    str(tr(defin.VIDEO_CODECS[e].get("description") or ""))
                    for e in encoders_codec_keys
                ]
                if preset_choice == "lossless":
                    codec_labels: List[str] = []
                    mtx = lossless_matrix.get(c, {"copy": [], "strict": []})
                    copy_set = set(mtx.get("copy", []))
                    strict_set = set(mtx.get("strict", []))
                    for e in encoders_codec_keys:
                        base = str(defin.VIDEO_CODECS[e].get("name", e))
                        copy_ok = e in copy_set
                        lbl = _lossless_mark_label(
                            base,
                            copy_ok=copy_ok,
                            strict_ok=False if copy_ok else e in strict_set,
                        )
                        codec_labels.append(lbl)
                else:
                    codec_labels = [
                        str(defin.VIDEO_CODECS[e].get("name", e))
                        for e in encoders_codec_keys
                    ]

                ans = ui.ask_user(
                    _("choose_codec").format(format=c),
                    encoders_codec_keys,
                    codec_descriptions,
                    0,
                    codec_labels,
                )
                if ans is None:
                    break
                codec_choice_by_container[c] = ans

            if ans is None:
                continue

            # Audio-Codec
            audio_keep_prompt = (
                _("audio_codec_keep_prompt")
                + f" ({_('codec_label')}: {src_audio_codec or '?'})"
            )
            ans_keep_audio = ui.ask_yes_no(audio_keep_prompt, default=True)
            if ans_keep_audio is None:
                continue
            if ans_keep_audio:
                audio_codec_choice = "copy"
                force_audio_channels = None
            else:
                if format_choice != "keep":
                    audio_cont = format_choice
                else:
                    audio_cont = (
                        vec.detect_container_from_path(Path(files[0]))
                        if files
                        else None
                    )
                audio_opts = _audio_codecs_for_container(audio_cont)
                default_audio = _default_audio_codec_for_container(audio_cont)
                default_idx = (
                    audio_opts.index(default_audio)
                    if default_audio in audio_opts
                    else 0
                )
                # Labels mit aktuellem Codec kennzeichnen
                audio_labels = []
                for opt in audio_opts:
                    if src_audio_codec and opt.lower() == src_audio_codec.lower():
                        audio_labels.append(f"{opt} (aktuell)")
                    else:
                        audio_labels.append(opt)
                audio_choice = ui.ask_user(
                    _("choose_audio_codec"),
                    audio_opts,
                    default=default_idx,
                    back_button=True,
                    display_labels=audio_labels,
                )
                if audio_choice is None:
                    continue
                audio_codec_choice = audio_choice
                force_audio_channels = None
                if (src_audio_channels or 0) > 2:
                    ch = int(src_audio_channels or 0)
                    opts: List[str] = ["keep"] + [str(c) for c in range(ch - 1, 0, -1)]
                    channel_labels: List[str] = []
                    descs: List[str] = []
                    for o in opts:
                        if o == "keep":
                            channel_labels.append(f"Original ({ch} ch)")
                            descs.append("Keine Aenderung")
                        else:
                            c = int(o)
                            if c == 2:
                                channel_labels.append("Stereo (2.0)")
                                descs.append("Kompatibel")
                            elif c == 1:
                                channel_labels.append("Mono (1.0)")
                                descs.append("Maximal kompatibel")
                            else:
                                channel_labels.append(f"{c} Kanäle")
                                descs.append("Downmix")
                    sel = ui.ask_user(
                        _("choose_audio_channels").format(ch=ch),
                        opts,
                        descs,
                        default=0,
                        display_labels=channel_labels,
                        back_button=True,
                    )
                    if sel is None:
                        continue
                    if sel == "keep":
                        force_audio_channels = None
                    else:
                        force_audio_channels = int(sel)

            # Pixelformat
            target_cont_for_pf = (
                format_choice
                if format_choice != "keep"
                else vec.detect_container_from_path(Path(files[0])) if files else None
            )
            src_pf_opt = src_pix_fmt_first or (
                _get_hdr_info(Path(files[0]), hdr_cache).src_pix_fmt if files else None
            )
            has_alpha = _pix_fmt_has_alpha(src_pf_opt)
            pix_prompt = _("pix_fmt_keep_prompt") + f" (aktuell: {src_pf_opt or '?'})"
            ans_keep_pf = ui.ask_yes_no(pix_prompt, default=True)
            if ans_keep_pf is None:
                continue
            if ans_keep_pf:
                pix_fmt_choice = None
            else:
                pf_opts = _pix_fmt_options_for(
                    target_cont_for_pf, src_pf_opt, has_alpha
                )
                default_pf = vp.playback_pix_fmt_for(target_cont_for_pf)
                default_idx_pf = (
                    pf_opts.index(default_pf) if default_pf in pf_opts else 0
                )
                pf_labels = []
                for opt in pf_opts:
                    if src_pf_opt and opt.lower() == src_pf_opt.lower():
                        pf_labels.append(f"{opt} (aktuell)")
                    else:
                        pf_labels.append(opt)
                pf_choice = ui.ask_user(
                    _("choose_pix_fmt"),
                    pf_opts,
                    default=default_idx_pf,
                    back_button=True,
                    display_labels=pf_labels,
                )
                if pf_choice is None:
                    continue
                pix_fmt_choice = pf_choice

            # Video-Bitrate (VBV-Cap)
            ans_keep_vb = ui.ask_yes_no(_("video_bitrate_keep_prompt"), default=True)
            if ans_keep_vb is None:
                continue
            if ans_keep_vb:
                video_bitrate_choice = None
            else:
                vb_opts = ["2M", "4M", "8M", "12M", "custom"]
                vb_desc = [
                    "SD/TV (sparsam)",
                    "HD (ausgewogen)",
                    "Full HD (gute Qualitaet)",
                    "UHD (hoch)",
                    "Benutzerdefiniert",
                ]
                vb_labels = ["2 Mbps", "4 Mbps", "8 Mbps", "12 Mbps", "Custom"]
                vb_choice = ui.ask_user(
                    _("choose_video_bitrate"),
                    vb_opts,
                    vb_desc,
                    default=2,
                    display_labels=vb_labels,
                    back_button=True,
                )
                if vb_choice is None:
                    continue
                if vb_choice == "custom":
                    while True:
                        raw = input(
                            co.return_promt(_("video_bitrate_custom_prompt") + ": ")
                        ).strip()
                        norm = _normalize_video_bitrate_choice(raw)
                        if norm:
                            video_bitrate_choice = norm
                            break
                        co.print_fail(_("invalid_video_bitrate"))
                else:
                    video_bitrate_choice = _normalize_video_bitrate_choice(vb_choice)

            # Untertitel bei MKV → MP4 extern mitführen?
            keep_subtitles = False
            subtitle_format = None
            if _should_offer_subtitle_export(files, format_choice):
                ans_keep_subs = ui.ask_yes_no(_("keep_subtitles_prompt"), default=False)
                if ans_keep_subs is None:
                    continue
                if ans_keep_subs:
                    sub_tbl = getattr(defin, "SUBTITLE_FORMATS", {})
                    subtitle_keys = list(sub_tbl.keys())
                    subtitle_descriptions = [
                        tr((sub_tbl.get(k) or {}).get("description", ""))
                        for k in subtitle_keys
                    ]
                    subtitle_labels = [
                        (sub_tbl.get(k) or {}).get("name", k) for k in subtitle_keys
                    ]
                    sub_choice = ui.ask_user(
                        _("select_subtitle_format"),
                        subtitle_keys,
                        subtitle_descriptions,
                        0,
                        subtitle_labels,
                        back_button=True,
                    )
                    if sub_choice is None:
                        continue
                    keep_subtitles = True
                    subtitle_format = sub_choice

            # Auflösung
            if preset_choice == "lossless":
                resolution = "original"
                resolution_scale = None
            else:
                min_src_w, min_src_h = vec.probe_min_source_geometry(files)
                resolution_keys = list(defin.RESOLUTIONS)
                filtered_keys = vec.filter_resolution_keys_for_source(
                    cast(Mapping[str, Mapping[str, str]], defin.RESOLUTIONS),
                    resolution_keys,
                    min_src_w,
                    min_src_h,
                )
                resolution_descriptions = [
                    str(tr(defin.RESOLUTIONS[k].get("description") or ""))
                    for k in filtered_keys
                ]
                resolution_labels = [
                    str(defin.RESOLUTIONS[k].get("name", k)) for k in filtered_keys
                ]
                res_prompt = _("choose_resolution")
                if min_src_w and min_src_h:
                    res_prompt = f"{res_prompt} ({min_src_w}x{min_src_h})"
                resolution = ui.ask_user(
                    res_prompt,
                    filtered_keys,
                    resolution_descriptions,
                    0,
                    resolution_labels,
                )
                if resolution is None:
                    continue
                resolution_scale = None
                if resolution == "custom":
                    while True:
                        candidate = ui.ask_two_ints(
                            _("enter_custom_resolution")
                        )  # "W:H"
                        ok, msg = vec.validate_custom_scale_leq(
                            min_src_w, min_src_h, candidate, require_even=True
                        )
                        if ok:
                            resolution_scale = candidate
                            break
                        co.print_fail(msg)

            # FPS-Cap
            if preset_choice == "lossless":
                framerate = "original"
                user_fps_rational = None
            else:
                user_fps_rational = None
                min_fps, max_fps = he.get_framerate_range(files)
                max_fps_disp = (
                    f"{max_fps:.2f}" if isinstance(max_fps, (int, float)) else "?"
                )
                preset_max_fps = defin.CONVERT_PRESET[preset_choice].get("max_fps")
                if preset_max_fps is None:
                    ans_cap = ui.ask_yes_no(
                        _("want_framerate_cap").format(max_fps=max_fps_disp)
                    )
                    if ans_cap is None:
                        continue
                    if ans_cap:
                        default_val = (
                            "30"
                            if isinstance(min_fps, (int, float)) and min_fps >= 30
                            else (
                                str(min_fps)
                                if isinstance(min_fps, (int, float))
                                else "30"
                            )
                        )
                        max_val = (
                            float(min_fps)
                            if isinstance(min_fps, (int, float))
                            else 30.0
                        )
                        val = ui.ask_float(
                            _("enter_framerate_cap"),
                            default=default_val,
                            min_value=1,
                            max_value=max_val,
                            return_type="rational",
                        )
                        user_fps_rational = str(val) if val is not None else None

            framerate = "original" if user_fps_rational is None else user_fps_rational

            hdr_to_sdr = False
            hdr_mode = None
            hdr_found = _detect_any_hdr(files, hdr_cache)
            if hdr_found:
                if (preset_choice or "") in _HDR_AUTO_PRESETS:
                    hdr_to_sdr = True
                    hdr_mode = "auto"
                else:
                    ans_hdr = ui.ask_yes_no(_("hdr_to_sdr_prompt"))
                    if ans_hdr is None:
                        continue
                    hdr_to_sdr = bool(ans_hdr)
                    hdr_mode = "manual" if hdr_to_sdr else "off"

            if hdr_to_sdr and not _hdr_tonemap_filters_available():
                co.print_warning(_("hdr_to_sdr_filters_missing"))
                hdr_to_sdr = False
                hdr_mode = "off"

            hdr_param = _hdr_param_value(hdr_found, hdr_to_sdr, hdr_mode)

            params, labels = _build_params_for_table_interactive(
                files=files,
                preset_choice=preset_choice,
                format_choice=format_choice,
                resolution_key=resolution,
                resolution_scale=resolution_scale,
                codec_choice_by_container=codec_choice_by_container,
                presets=defin.CONVERT_PRESET,
                res_defs=defin.RESOLUTIONS,
                user_fps_rational=user_fps_rational,
                hdr_to_sdr=hdr_param,
                target_pix_fmt=pix_fmt_choice,
                audio_codec_choice=audio_codec_choice,
                audio_channels=force_audio_channels,
                video_bitrate_cap=video_bitrate_choice,
            )
            break

    # 2) Status
    co.print_selected_params_table(params, labels=labels)

    # 3) Konvertierung
    for i, file in enumerate(files):
        input_path = Path(file).resolve()
        if not input_path.exists():
            co.print_warning(_("file_not_found").format(file=input_path.name))
            continue

        # Cover der Quelle sichern
        preserved_cover: Optional[Path] = None
        had_thumb: bool = False
        try:
            had_thumb = vt.check_thumbnail(input_path, silent=True)
            preserved_cover = vt.extract_thumbnail(input_path)
        except Exception:
            preserved_cover = None
            had_thumb = False

        target_container = (
            (format_choice or "keep").lower()
            if format_choice != "keep"
            else (vec.detect_container_from_path(input_path) or "mkv")
        )
        if BATCH_MODE and not keep_subtitles:
            in_cont = _normalize_container_name(
                vec.detect_container_from_path(input_path)
            )
            out_cont = _normalize_container_name(target_container)
            if (
                in_cont
                and out_cont
                and in_cont != out_cont
                and _container_drops_subtitles(out_cont)
                and _has_subtitle_streams(input_path)
            ):
                co.print_warning(_("subtitles_will_be_lost"))
        pf_choice_effective = pix_fmt_choice
        cont_norm_pf = _normalize_container_name(target_container)
        allowed_pfs = _PIX_FMT_BY_CONTAINER_LIMITED.get(cont_norm_pf)
        if (
            pf_choice_effective
            and allowed_pfs
            and pf_choice_effective not in allowed_pfs
        ):
            co.print_warning(
                _("pix_fmt_not_allowed").format(
                    pix_fmt=pf_choice_effective,
                    container=target_container,
                    sugg=allowed_pfs[0],
                )
            )
            pf_choice_effective = allowed_pfs[0]

        encoder_map: OrderedDict[str, Any] = encoder_maps_by_container.get(
            target_container, OrderedDict()
        )

        # Quelle
        hdr_info = _get_hdr_info(input_path, hdr_cache)
        src_fps = he.probe_src_fps(input_path)
        src_w, src_h, src_fmt = (
            hdr_info.src_w,
            hdr_info.src_h,
            hdr_info.src_pix_fmt,
        )
        hdr_to_sdr_this = bool(hdr_to_sdr and hdr_info.is_hdr)
        if hdr_to_sdr_this and not _hdr_tonemap_filters_available():
            hdr_to_sdr_this = False

        # Container/Codec prüfen
        chosen_codec_key = codec_choice_by_container.get(target_container, "copy")
        if chosen_codec_key != "copy" and not vec.container_allows_codec(
            target_container, chosen_codec_key
        ):
            sugg = vec.suggest_codec_for_container(target_container)
            co.print_warning(
                _("unsuitable_codec").format(
                    codec=chosen_codec_key, container=target_container, sugg=sugg
                )
            )
            chosen_codec_key = sugg

        if hdr_to_sdr_this and chosen_codec_key == "copy":
            fallback_codec = vec.suggest_codec_for_container(target_container)
            co.print_warning(
                _("hdr_to_sdr_requires_reencode").format(codec=fallback_codec)
            )
            chosen_codec_key = fallback_codec

        if pf_choice_effective and chosen_codec_key == "copy":
            fallback_codec = vec.suggest_codec_for_container(target_container)
            co.print_warning(
                _("pix_fmt_requires_reencode").format(codec=fallback_codec)
            )
            chosen_codec_key = fallback_codec

        preferred: Optional[Any] = (
            encoder_map.get(chosen_codec_key) if chosen_codec_key != "copy" else None
        )

        # Ziel-Skalierung
        if resolution == "custom" and resolution_scale:
            req_scale: Optional[str] = resolution_scale
        elif resolution != "original":
            # FIX: cast(str, resolution) statt cast[str](resolution)
            req_scale = str(defin.RESOLUTIONS[cast(str, resolution)]["scale"])
        else:
            preset_scale = defin.CONVERT_PRESET.get(cast(str, preset_choice), {}).get(
                "scale"
            )
            req_scale = preset_scale if isinstance(preset_scale, str) else None

        # Plan bauen
        plan = vec.build_transcode_plan(
            input_path=input_path,
            target_container=target_container,
            preset_name=cast(str, preset_choice),
            codec_key=chosen_codec_key,
            preferred_encoder=preferred,
            req_scale=req_scale,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            user_fps_rational=user_fps_rational,
            preset_max_fps=defin.CONVERT_PRESET[cast(str, preset_choice)].get(
                "max_fps"
            ),
            allow_stream_copy=not hdr_to_sdr_this and not bool(pf_choice_effective),
            force_sw=hdr_to_sdr_this,
        )

        eff_pix_fmt = pf_choice_effective or vp.infer_target_pix_fmt_from_plan(plan)
        co.print_info(_("target_pix_fmt") + f" => {eff_pix_fmt or '—'}")

        in_ext = input_path.suffix.lstrip(".").lower()
        out_ext = target_container

        is_plan_obj = hasattr(plan, "codec_key")
        real_codec_key = (
            getattr(plan, "codec_key", chosen_codec_key) or chosen_codec_key
        )
        plan_safe_scale = getattr(plan, "safe_scale", None) if is_plan_obj else None

        suffix_parts: List[str] = []
        if real_codec_key != "copy":
            suffix_parts.append(real_codec_key)
        if plan_safe_scale:
            suffix_parts.append(str(plan_safe_scale))
        if framerate != "original":
            suffix_parts.append(f"{framerate}fps")

        suffix = "_converted" + (
            ("_" + "_".join(map(str, suffix_parts))) if suffix_parts else ""
        )
        suffix = suffix.replace(":", "X").replace("/", "_")

        output = fs.build_output_path(
            input_path=input_path,
            output_arg=getattr(args, "output", None),
            default_suffix=suffix,
            idx=i,
            total=len(files),
            target_ext=None if in_ext == out_ext else out_ext,
        )

        final_cmd = _plan_to_cmd(
            plan,
            input_path=input_path,
            output=output,
            target_container=target_container,
            preset_name=cast(str, preset_choice),
        )
        final_cmd = vec.apply_container_codec_quirks(
            final_cmd, target_container, chosen_codec_key
        )

        if hdr_to_sdr_this:
            tone_vf = _build_hdr_to_sdr_chain(hdr_info.meta)
            if tone_vf:
                current_vf = None
                if "-vf" in final_cmd:
                    i_vf = final_cmd.index("-vf")
                    if i_vf + 1 < len(final_cmd):
                        current_vf = str(final_cmd[i_vf + 1])
                new_vf = f"{tone_vf},{current_vf}" if current_vf else str(tone_vf)
                _set_vf_chain(final_cmd, new_vf)
                _strip_color_metadata_args(final_cmd)
                _strip_x264_color_params(final_cmd)
                sdr_meta: Dict[str, Optional[str]] = {
                    "color_primaries": "bt709",
                    "color_trc": "bt709",
                    "colorspace": "bt709",
                    "color_range": "tv",
                }
                final_cmd = vp.apply_color_signaling(
                    final_cmd,
                    container=target_container,
                    codec_key=real_codec_key,
                    meta=sdr_meta,
                    src_pix_fmt=src_fmt,
                )

        if audio_codec_choice:
            audio_choice_eff = _normalize_audio_codec_choice(audio_codec_choice)
            if audio_choice_eff == "copy":
                src_audio_codec, _t = aud._probe_audio_codec(input_path)
                src_norm = _normalize_audio_codec_name(src_audio_codec)
                if src_norm and not _audio_codec_allowed(target_container, src_norm):
                    fallback = _default_audio_codec_for_container(target_container)
                    co.print_warning(
                        _("unsuitable_audio_codec").format(
                            codec=src_norm, container=target_container, sugg=fallback
                        )
                    )
                    audio_choice_eff = None
            elif audio_choice_eff and not _audio_codec_allowed(
                target_container, audio_choice_eff
            ):
                fallback = _default_audio_codec_for_container(target_container)
                co.print_warning(
                    _("unsuitable_audio_codec").format(
                        codec=audio_choice_eff,
                        container=target_container,
                        sugg=fallback,
                    )
                )
                audio_choice_eff = None

            if audio_choice_eff:
                _apply_audio_codec_override(
                    final_cmd,
                    audio_choice=audio_choice_eff,
                    target_container=target_container,
                    preset_name=cast(str, preset_choice),
                    input_path=input_path,
                    force_channels=force_audio_channels,
                )

        if pf_choice_effective:
            pf_eff = _normalize_pix_fmt_choice(pf_choice_effective)
            if pf_eff:
                _apply_pix_fmt_override(final_cmd, pf_eff)

        if video_bitrate_choice:
            if real_codec_key == "copy":
                co.print_warning(_("video_bitrate_requires_reencode"))
            else:
                _apply_video_bitrate_cap(final_cmd, video_bitrate_choice)

        final_cmd = autotune_final_cmd(input_path, final_cmd)

        # Sicherstellen, dass der Output-Pfad am Ende steht (manche Schritte hängen Flags hinten an)
        try:
            out_idx = final_cmd.index(str(output))
            if out_idx != len(final_cmd) - 1:
                final_cmd.pop(out_idx)
                final_cmd.append(str(output))
        except ValueError:
            final_cmd.append(str(output))

        # Re-Order, falls Optionen hinter dem Output gelandet sind
        try:
            vec.ensure_pre_output_order(final_cmd)
        except Exception:
            pass

        co.print_debug("ffmeg_cmd", final_cmd=final_cmd)

        # subprocess.run(final_cmd)
        try:
            pw.run_ffmpeg_with_progress(
                input_path.name,
                final_cmd,
                _("converting_file_progress"),
                _("converting_file_done"),
                output,
                BATCH_MODE=BATCH_MODE,
            )
        except pw.FFmpegFailed as err:
            fallback_cmd = _fallback_to_sw_encoder(final_cmd)
            if fallback_cmd:
                co.print_warning(
                    "Hardware-Encoding fehlgeschlagen – wechsle auf Software-Encoder."
                )
                pw.run_ffmpeg_with_progress(
                    input_path.name,
                    fallback_cmd,
                    _("converting_file_progress"),
                    _("converting_file_done"),
                    output,
                    BATCH_MODE=BATCH_MODE,
                )
            else:
                raise err

        # Cover zurückschreiben
        try:
            if output.exists() and had_thumb:
                _verify_and_embed_thumbnail(
                    output=output,
                    preserved_cover=preserved_cover,
                    had_thumb=had_thumb,
                )
            elif output.exists() and not had_thumb:
                co.print_info(_("no_thumbnail_found"))
        except Exception as e:
            co.print_warning(_("embedding_skipped") + f": {e}")

        try:
            if preserved_cover and preserved_cover.exists():
                preserved_cover.unlink(missing_ok=True)
        except Exception:
            pass

        # Untertitel ggf. extern extrahieren
        if keep_subtitles and subtitle_format:
            if _has_subtitle_streams(input_path):
                try:
                    sub_args = ex.ExtractArgs(
                        files=[str(input_path)],
                        subtitle="",
                        format=subtitle_format,
                        output=str(output.parent),
                    )
                    ex.extract(sub_args)
                except Exception as e:
                    co.print_warning(_("extracting_subtitle_failed") + f": {e}")

    co.print_finished(_("converting_method"))
