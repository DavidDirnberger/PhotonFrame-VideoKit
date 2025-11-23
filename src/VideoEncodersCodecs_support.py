#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VideoEncodersCodecs_support.py
from __future__ import annotations
from pathlib import Path
import re
import json
import subprocess
import glob
import os
import tempfile
from functools import lru_cache
from typing import (
    List,
    Mapping,
    Any,
    Optional,
    Dict,
    Tuple,
    NamedTuple,
    Set,
    cast,
)

import definitions as defin

PresetLike = Mapping[str, Any]


class LosslessChoice(NamedTuple):
    encoder: str
    args: List[str]
    allow_pixfmt_inject: bool
    extra_args: List[str]  # z. B. ['-tag:v','hvc1'] für MP4+HEVC


def _output_index(cmd: List[str]) -> int:
    """
    Bestimme die Einfügeposition 'vor Output'.
    - Wenn das letzte Token plausibel wie ein Ausgabepfad aussieht (kein '-' Prefix,
      hat '.'-Erweiterung oder '/' bzw. '\\'), dann ist es der Output -> index = len(cmd)-1.
    - Werte wie '3M', '160k', '30000/1001' sind KEIN Output.
    - Wenn kein Output erkennbar ist, liefern wir len(cmd) (=> Append-Position).
    """
    if not cmd:
        return 0

    last = str(cmd[-1])
    if not last.startswith("-"):
        # Nicht als Output zählen: nackte Zahlen/Bitraten/Ratios
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?[kKmM]?", last):
            return len(cmd)  # z.B. '3M'
        if re.fullmatch(r"\d+/\d+", last):
            return len(cmd)  # z.B. '30000/1001'
        # Dateipfad-Heuristik
        if ("/" in last) or ("\\" in last):
            return len(cmd) - 1
        if "." in last and not last.startswith("."):
            return len(cmd) - 1

    # Kein echter Output am Ende → wir hängen lieber an
    return len(cmd)


def _ffprobe_geometry(
    path: Path, ffprobe_bin: str = "ffprobe"
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        data = cast(Dict[str, Any], json.loads(out))
        streams = cast(List[Dict[str, Any]], data.get("streams") or [])
        st: Dict[str, Any] = streams[0] if streams else {}
        w = cast(Optional[int], st.get("width"))
        h = cast(Optional[int], st.get("height"))
        pf = cast(Optional[str], st.get("pix_fmt"))
        return w, h, pf
    except Exception:
        return None, None, None


def _insert_before_output(cmd: List[str], tokens: List[str]) -> None:
    idx = _output_index(cmd)
    cmd[idx:idx] = tokens


def _strip_options(ffmpeg_cmd: List[str], names: set[str]) -> None:
    """
    Entfernt Optionen robust – egal ob als Paar (-flag value) oder als alleinstehendes Flag.
    """
    out: List[str] = []
    i = 0
    n = len(ffmpeg_cmd)
    while i < n:
        tok = ffmpeg_cmd[i]
        if tok in names:
            # Wenn es einen Wert gibt und der nicht wieder ein Flag ist, überspringen wir beides.
            if i + 1 < n and not str(ffmpeg_cmd[i + 1]).startswith("-"):
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    ffmpeg_cmd[:] = out


def _set_kv_arg(cmd: List[str], key: str, val: str):
    if key in cmd:
        i = cmd.index(key)
        if i + 1 < len(cmd):
            cmd[i + 1] = val
        else:
            cmd += [val]
    else:
        cmd += [key, val]


def _get_preset_spec(
    presets: Mapping[str, PresetLike], name: str, fallback: str = "casual"
) -> PresetLike:
    if name in presets:
        return presets[name]
    if fallback in presets:
        return presets[fallback]
    raise KeyError(f"Preset '{name}' not found and fallback '{fallback}' missing.")


def _norm_matrix(v: Optional[str]) -> str:
    v = (v or "").lower()
    if v in ("bt709", "bt.709"):
        return "bt709"
    if v in ("smpte170m", "bt470bg"):
        return "smpte170m"
    return "bt709"


def _norm_range(v: Optional[str]) -> str:
    v = (v or "").lower()
    if v in ("pc", "jpeg", "full"):
        return "full"
    return "limited"


def _encoder_family(enc: str) -> str:
    e = (enc or "").lower()
    if e in ("mjpeg", "ljpeg"):
        return "mjpeg"
    if e.startswith("h264_") or e == "libx264":
        return "h264"
    if e.startswith("hevc_") or e == "libx265":
        return "hevc"
    if e.startswith("av1_") or e in ("libsvtav1", "libaom-av1", "rav1e"):
        return "av1"
    if e in ("libvpx-vp9", "vp9_qsv", "vp9_vaapi"):
        return "vp9"
    if e in ("libvpx", "libvpx-v8", "vp8_qsv", "vp8_vaapi"):
        return "vp8"
    if e == "mpeg4":
        return "mpeg4"
    if e.startswith("prores"):
        return "prores"
    if e.startswith("dnx"):
        return "dnxhd"
    if e in ("ffv1", "huffyuv", "utvideo", "rawvideo", "mjpeg", "png"):
        return e
    if e in ("jpeg2000", "libopenjpeg"):
        return "jpeg2000"
    if e in ("qtrle", "hap", "magicyuv", "cfhd", "cineform"):
        return "cineform" if e in ("cfhd", "cineform") else e
    if e in ("theora", "libtheora"):
        return "theora"
    if e in ("mpeg2video", "mpeg1video"):
        return e
    if e.endswith("_qsv"):
        return "qsv"
    if e.endswith("_vaapi"):
        return "vaapi"
    if e.endswith("_amf"):
        return "amf"
    if e.endswith("_videotoolbox"):
        return "vtb"
    return "other"


def _container_allows_codec(container: str, codec_key: str) -> bool:
    c = (container or "").lower()
    k = (codec_key or "").lower()
    if c == "webm":
        return k in ("vp8", "vp9", "av1")
    if c == "mp4":
        return k in ("h264", "hevc", "av1", "mpeg4")
    if c == "avi":
        return k in (
            "mpeg4",
            "mjpeg",
            "huffyuv",
            "utvideo",
            "hap",
            "rawvideo",
            "magicyuv",
            "cineform",
        )
    if c == "flv":
        return k in ("flv", "vp6", "screenvideo")
    if c == "mpeg":
        return k in ("mpeg1video", "mpeg2video")
    if c in ("mkv", "mov"):
        return True
    return True


def _active_encoder_or_guess(cmd: List[str], codec_key: str) -> str:
    try:
        i = cmd.index("-c:v")
        return cmd[i + 1]
    except ValueError:
        return _encoder_for_codec(_normalize_codec_key(codec_key) or codec_key)


@lru_cache(maxsize=512)
def _encoder_supports_pix_fmt(
    encoder: str, pix_fmt: str, ffmpeg_bin: str = "ffmpeg"
) -> bool:
    enc = (encoder or "").strip()
    pf = (pix_fmt or "").strip()
    if not enc or not pf:
        return False
    # Schneller negativer Pfad: Hilfeausgabe parsen (wenn verfügbar)
    try:
        out = subprocess.check_output(
            [ffmpeg_bin, "-hide_banner", "-h", f"encoder={enc}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        for line in out.splitlines():
            if "Supported pixel formats" in line:
                fmts = line.split(":", 1)[1].replace(",", " ").split()
                if pf in fmts:
                    return True
                break
    except Exception:
        pass
    # Definitive Kurzprobe (1 Frame → null), komplett stumm
    try:
        rc = subprocess.call(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=16x16:rate=1:duration=0.04",
                "-frames:v",
                "1",
                "-pix_fmt",
                pf,
                "-c:v",
                enc,
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return rc == 0
    except Exception:
        return False


def _merge_x264_params(cmd: List[str], extra: str) -> None:
    if "-x264-params" in cmd:
        i = cmd.index("-x264-params")
        cmd[i + 1] = cmd[i + 1].rstrip(":") + ":" + extra.lstrip(":")
    else:
        cmd += ["-x264-params", extra]


def _apply_h264_vui_flags_if_needed(
    cmd: List[str], meta: Dict[str, Optional[str]], src_pix_fmt: Optional[str]
) -> List[str]:
    if "-x264-params" in cmd:
        i = cmd.index("-x264-params")
        v = cmd[i + 1]
        if any(
            k in v for k in ("colorprim=", "transfer=", "colormatrix=", "fullrange=")
        ):
            return cmd

    prim = (meta.get("color_primaries") or "").lower()
    trc = (meta.get("color_trc") or "").lower()
    csp = (meta.get("colorspace") or "").lower()
    rng = (meta.get("color_range") or "").lower()
    pf = (src_pix_fmt or "").lower()

    cm = None
    if csp in ("bt709", "smpte170m", "smpte240m", "bt470bg", "bt2020ncl", "bt2020cl"):
        cm = (
            "bt709"
            if csp == "bt709"
            else (
                "smpte170m"
                if csp == "smpte170m"
                else (
                    "smpte240m"
                    if csp == "smpte240m"
                    else (
                        "bt470bg"
                        if csp == "bt470bg"
                        else "bt2020nc" if csp == "bt2020ncl" else "bt2020c"
                    )
                )
            )
        )

    cp = None
    if prim in ("bt709", "smpte170m", "smpte240m", "bt470bg", "bt2020"):
        cp = (
            "bt709"
            if prim == "bt709"
            else (
                "smpte170m"
                if prim == "smpte170m"
                else (
                    "smpte240m"
                    if prim == "smpte240m"
                    else "bt470bg" if prim == "bt470bg" else "bt2020"
                )
            )
        )

    tf = None
    if trc in (
        "bt709",
        "smpte170m",
        "smpte240m",
        "iec61966-2-1",
        "iec61966-2-4",
        "bt2020-10",
        "bt2020-12",
    ):
        tf = (
            "bt709"
            if trc == "bt709"
            else (
                "smpte170m"
                if trc == "smpte170m"
                else (
                    "smpte240m"
                    if trc == "smpte240m"
                    else (
                        "iec61966-2-1"
                        if trc == "iec61966-2-1"
                        else (
                            "iec61966-2-4"
                            if trc == "iec61966-2-4"
                            else "bt2020-10" if trc == "bt2020-10" else "bt2020-12"
                        )
                    )
                )
            )
        )

    is_full = (
        (rng in ("pc", "jpeg"))
        or pf.startswith("yuvj")
        or pf.startswith("rgb")
        or pf == "gbr"
    )
    fr = "fullrange=on" if is_full else "fullrange=off"

    vui_parts = []
    if cp:
        vui_parts.append(f"colorprim={cp}")
    if tf:
        vui_parts.append(f"transfer={tf}")
    if cm:
        vui_parts.append(f"colormatrix={cm}")
    vui_parts.append(fr)

    if vui_parts:
        _merge_x264_params(cmd, ":".join(vui_parts))

    return cmd


def _is_stream_copy(cmd: List[str]) -> bool:
    try:
        i = cmd.index("-c:v")
        return (i + 1 < len(cmd)) and cmd[i + 1].strip().lower() == "copy"
    except ValueError:
        return False


def _cmd_contains_any(cmd: List[str], needles: set[str]) -> bool:
    s = " ".join(map(str, cmd))
    return any(n in s for n in needles)


def _is_explicit_lossless(cmd: List[str]) -> bool:
    if _cmd_contains_any(cmd, {"-qp 0", "lossless=1", "-tune lossless"}):
        return True
    if _cmd_contains_any(
        cmd, {"-c:v ffv1", "-c:v huffyuv", "-c:v utvideo", "-c:v rawvideo"}
    ):
        return True
    return False


def _active_encoder(cmd: List[str], fallback: str) -> str:
    try:
        i = cmd.index("-c:v")
        return cmd[i + 1]
    except ValueError:
        return fallback


def _normalize_codec_key(value: Optional[str]) -> Optional[str]:
    """
    Fix für Pylance: Parameter nicht 'x' nennen und hartes Guarding + klare Typen.
    """
    if value is None:
        return None
    s = value.strip().lower()
    s = re.sub(r"_(nvenc|qsv|amf|vaapi|videotoolbox)$", "", s)

    alias: Dict[str, str] = {
        "libx264": "h264",
        "x264": "h264",
        "avc": "h264",
        "avc1": "h264",
        "libx264rgb": "h264",
        "h264": "h264",
        "libx265": "hevc",
        "x265": "hevc",
        "hevc": "hevc",
        "h265": "hevc",
        "hvc1": "hevc",
        "hev1": "hevc",
        "libaom-av1": "av1",
        "libsvtav1": "av1",
        "rav1e": "av1",
        "av1": "av1",
        "libvpx-vp9": "vp9",
        "vp9": "vp9",
        "libvpx": "vp8",
        "libvpx-v8": "vp8",
        "vp8": "vp8",
        "mpeg4": "mpeg4",
        "mpeg2video": "mpeg2video",
        "mpeg2": "mpeg2video",
        "mpeg1video": "mpeg1video",
        "mpeg1": "mpeg1video",
        "prores_ks": "prores",
        "prores": "prores",
        "dnxhd": "dnxhd",
        "dnxhr": "dnxhd",
        "dnx": "dnxhd",
        "ffv1": "ffv1",
        "huffyuv": "huffyuv",
        "ffvhuff": "huffyuv",
        "utvideo": "utvideo",
        "rawvideo": "rawvideo",
        "magicyuv": "magicyuv",
        "jpeg2000": "jpeg2000",
        "j2k": "jpeg2000",
        "libopenjpeg": "jpeg2000",
        "mjpeg": "mjpeg",
        "png": "png",
        "theora": "theora",
        "libtheora": "theora",
        "qtrle": "qtrle",
        "hap": "hap",
        "cfhd": "cineform",
        "cineform": "cineform",
        "vp6": "vp6",
        "vp6f": "vp6",
        "flv": "flv",
        "flv1": "flv",
        "screenvideo": "screenvideo",
        "flashsv": "screenvideo",
        "flashsv2": "screenvideo",
    }
    return alias.get(s, s)


def _encoder_for_codec(codec_name: Optional[str]) -> str:
    cn = (codec_name or "").strip().lower()
    if not cn:
        return "libx264"

    try:
        for _, enc_list in defin.CODEC_ENCODER_CANDIDATES.items():
            if cn in (e.lower() for e in enc_list):
                return cn
    except Exception:
        pass

    alias_to_key = {
        "avc": "h264",
        "avc1": "h264",
        "x264": "h264",
        "libx264": "h264",
        "libx264rgb": "h264",
        "h264": "h264",
        "hevc": "hevc",
        "h265": "hevc",
        "hev1": "hevc",
        "hvc1": "hevc",
        "x265": "hevc",
        "libx265": "hevc",
        "av1": "av1",
        "libaom-av1": "av1",
        "libsvtav1": "av1",
        "rav1e": "av1",
        "vp9": "vp9",
        "vp90": "vp9",
        "vp09": "vp9",
        "libvpx-vp9": "vp9",
        "vp8": "vp8",
        "vp80": "vp8",
        "libvpx": "vp8",
        "libvpx-v8": "vp8",
        "libvpx-vp8": "vp8",
        "mpeg4": "mpeg4",
        "libxvid": "mpeg4",
        "mpeg2": "mpeg2video",
        "mpeg2video": "mpeg2video",
        "mpeg1": "mpeg1video",
        "mpeg1video": "mpeg1video",
        "prores": "prores",
        "prores_ks": "prores",
        "prores_aw": "prores",
        "dnxhd": "dnxhd",
        "dnxhr": "dnxhd",
        "dnx": "dnxhd",
        "jpeg2000": "jpeg2000",
        "j2k": "jpeg2000",
        "libopenjpeg": "jpeg2000",
        "mjpeg": "mjpeg",
        "png": "png",
        "ffv1": "ffv1",
        "huffyuv": "huffyuv",
        "ffvhuff": "huffyuv",
        "utvideo": "utvideo",
        "rawvideo": "rawvideo",
        "magicyuv": "magicyuv",
        "theora": "theora",
        "libtheora": "theora",
        "qtrle": "qtrle",
        "hap": "hap",
        "cineform": "cineform",
        "cfhd": "cineform",
        "h263": "h263",
        "vp6": "vp6",
        "screenvideo": "screenvideo",
        "flashsv2": "screenvideo",
        "flashsv": "screenvideo",
    }
    key = alias_to_key.get(cn, cn)

    default_encoder_by_key = {
        "h264": "libx264",
        "hevc": "libx265",
        "av1": "libaom-av1",
        "vp9": "libvpx-vp9",
        "vp8": "libvpx",
        "mpeg2video": "mpeg2video",
        "mpeg1video": "mpeg1video",
        "mpeg4": "mpeg4",
        "prores": "prores_ks",
        "dnxhd": "dnxhd",
        "jpeg2000": "jpeg2000",
        "mjpeg": "mjpeg",
        "ffv1": "ffv1",
        "huffyuv": "huffyuv",
        "utvideo": "utvideo",
        "theora": "libtheora",
        "qtrle": "qtrle",
        "hap": "hap",
        "rawvideo": "rawvideo",
        "png": "png",
        "magicyuv": "magicyuv",
        "cineform": "cfhd",
        "h263": "h263",
        "vp6": "vp6",
        "screenvideo": "flashsv2",
    }

    if key in default_encoder_by_key:
        return default_encoder_by_key[key]

    try:
        if key in defin.CODEC_ENCODER_CANDIDATES:
            enc_list = defin.CODEC_ENCODER_CANDIDATES[key]
            if enc_list:
                return enc_list[0]
    except Exception:
        pass

    return "libx264"


def _pick_lossless_encoder_for_codec(src_codec: str) -> str:
    c = (src_codec or "").lower()
    if c in ("h264", "avc1"):
        return "libx264"
    if c in ("hevc", "h265", "hev1"):
        return "libx265"
    if c in ("vp9", "vp8"):
        return "libvpx-vp9"
    if c in ("av1",):
        return "libaom-av1"
    if c in ("png", "mjpeg"):
        return "png" if c == "png" else "mjpeg"
    if c in ("ffv1", "huffyuv", "utvideo"):
        return c
    if c in ("prores", "prores_ks"):
        return "prores_ks"
    if c in ("dnxhd", "dnxhr", "dnx"):
        return "dnxhd"
    return "libx264"


def _has_filter(name: str, ffmpeg_bin: str = "ffmpeg") -> bool:
    try:
        out = subprocess.check_output(
            [ffmpeg_bin, "-hide_banner", "-filters"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        for ln in out.splitlines():
            if not ln or ln[0] not in {" ", "."}:
                continue
            parts = ln.split()
            if len(parts) >= 2 and parts[1] == name:
                return True
        return False
    except Exception:
        return False


def _lossless_video_args_for_encoder(enc_name: str) -> List[str]:
    e = (enc_name or "").lower()
    if e == "libx264rgb":
        return ["-c:v", "libx264rgb", "-preset", "medium", "-qp", "0"]
    return _lossless_video_args(enc_name)


def _lossless_video_args(enc_name: str) -> List[str]:
    e = (enc_name or "").lower()

    if e == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-g",
            "1",
            "-bf",
            "0",
            "-qp",
            "0",
            "-x264-params",
            "qp=0:no-psy=1:aq-mode=0:mbtree=0:deblock=0,0",
        ]
    if e == "libx264rgb":
        return [
            "-c:v",
            "libx264rgb",
            "-preset",
            "medium",
            "-g",
            "1",
            "-bf",
            "0",
            "-x264-params",
            "qp=0:no-psy=1:aq-mode=0:mbtree=0:deblock=0,0",
        ]
    if e == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-rc",
            "constqp",
            "-qp",
            "0",
            "-tune",
            "lossless",
            "-g",
            "2",
            "-bf",
            "0",
        ]
    if e == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "1", "-g", "1", "-bf", "0"]
    if e == "h264_vaapi":
        return ["-c:v", "h264_vaapi", "-qp", "0", "-g", "1", "-bf", "0"]

    if e == "libx265":
        return [
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-x265-params",
            "lossless=1:sao=0",
        ]
    if e == "hevc_nvenc":
        return [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p7",
            "-rc",
            "constqp",
            "-qp",
            "0",
            "-tune",
            "lossless",
            "-g",
            "2",
            "-bf",
            "0",
        ]
    if e == "hevc_qsv":
        return ["-c:v", "hevc_qsv", "-global_quality", "1", "-g", "1", "-bf", "0"]
    if e == "hevc_vaapi":
        return ["-c:v", "hevc_vaapi", "-qp", "0", "-g", "1", "-bf", "0"]

    if e == "libvpx-vp9":
        return [
            "-c:v",
            "libvpx-vp9",
            "-lossless",
            "1",
            "-row-mt",
            "1",
            "-cpu-used",
            "2",
            "-g",
            "1",
        ]
    if e == "vp9_qsv":
        return ["-c:v", "vp9_qsv", "-global_quality", "1", "-g", "1", "-bf", "0"]
    if e == "vp9_vaapi":
        return ["-c:v", "vp9_vaapi", "-qp", "0", "-b:v", "0", "-g", "1", "-bf", "0"]

    if e in ("libvpx", "libvpx-v8"):
        return [
            "-c:v",
            "libvpx",
            "-b:v",
            "0",
            "-crf",
            "2",
            "-deadline",
            "best",
            "-cpu-used",
            "0",
            "-g",
            "240",
            "-keyint_min",
            "23",
            "-auto-alt-ref",
            "1",
            "-lag-in-frames",
            "25",
            "-qmin",
            "0",
            "-qmax",
            "8",
            "-sharpness",
            "0",
        ]
    if e == "vp8_vaapi":
        return ["-c:v", "vp8_vaapi", "-qp", "0", "-b:v", "0", "-g", "1"]

    if e == "libaom-av1":
        return [
            "-c:v",
            "libaom-av1",
            "-row-mt",
            "1",
            "-cpu-used",
            "3",
            "-g",
            "1",
            "-aom-params",
            "lossless=1",
        ]
    if e == "libsvtav1":
        return ["-c:v", "libsvtav1", "-qp", "0", "-preset", "6", "-g", "1"]
    if e == "av1_qsv":
        return ["-c:v", "av1_qsv", "-global_quality", "1", "-g", "1", "-bf", "0"]
    if e == "av1_vaapi":
        return ["-c:v", "av1_vaapi", "-qp", "0", "-b:v", "0", "-g", "1", "-bf", "0"]

    if e == "png":
        return ["-c:v", "png"]
    if e == "mjpeg":
        return ["-c:v", "mjpeg", "-q:v", "2"]
    if e in ("ffv1", "huffyuv", "utvideo"):
        return (
            ["-c:v", e]
            if e != "ffv1"
            else ["-c:v", "ffv1", "-level", "3", "-coder", "1"]
        )
    if e == "prores_ks":
        return ["-c:v", "prores_ks", "-profile:v", "3"]
    if e == "dnxhd":
        return ["-c:v", "dnxhd"]
    if e == "magicyuv":
        return ["-c:v", "magicyuv"]

    if e == "qtrle":
        return ["-c:v", "qtrle"]

    if e == "jpeg2000":
        return ["-c:v", "jpeg2000", "-pred", "1"]
    if e == "libopenjpeg":
        return ["-c:v", "libopenjpeg", "-qscale:v", "1"]

    if e in ("theora", "libtheora"):
        return ["-c:v", "libtheora", "-q:v", "10"]

    if e == "rawvideo":
        # Rohvideo ist per se mathematisch lossless
        return ["-c:v", "rawvideo"]

    if e == "magicyuv":
        # MagicYUV ist lossless (FourCC/TAG wird unten in Quirks gesetzt, falls nötig)
        return ["-c:v", "magicyuv"]

    if e in ("cfhd", "cineform"):
        return ["-c:v", "cfhd"]

    if e == "hap":
        return ["-c:v", "hap", "-chunks", "4"]

    fam = _encoder_family(e)
    if fam == "vp9":
        return [
            "-c:v",
            "libvpx-vp9",
            "-lossless",
            "1",
            "-row-mt",
            "1",
            "-cpu-used",
            "2",
            "-g",
            "1",
        ]
    if fam == "av1":
        return [
            "-c:v",
            "libaom-av1",
            "-row-mt",
            "1",
            "-cpu-used",
            "3",
            "-g",
            "1",
            "-aom-params",
            "lossless=1",
        ]
    if fam == "hevc":
        return [
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-x265-params",
            "lossless=1:sao=0",
        ]
    if fam == "h264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-g",
            "1",
            "-bf",
            "0",
            "-qp",
            "0",
        ]

    return ["-c:v", e]


def _h264_near_lossless_args_mp4() -> List[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "6",
        "-tune",
        "grain",
        "-x264-params",
        "aq-mode=3:aq-strength=1.0:psy-rd=0.6:deblock=0,0",
    ]


def _h264_near_lossless_args_mkv() -> List[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "6",
        "-tune",
        "grain",
        "-x264-params",
        "aq-mode=3:aq-strength=1.0:psy-rd=0.6:deblock=0,0",
    ]


def _pick_lossless_encoder_for_source(
    src_codec: Optional[str], src_pix_fmt: Optional[str]
) -> str:
    src_codec = (src_codec or "").lower()
    src_pix_fmt = (src_pix_fmt or "").lower()

    if _encoder_supports_pix_fmt("libx264rgb", src_pix_fmt):
        return "libx264rgb"

    if _encoder_supports_pix_fmt("libx265", src_pix_fmt):
        return "libx265"

    enc = _pick_lossless_encoder_for_codec(src_codec)

    if src_pix_fmt and not _encoder_supports_pix_fmt(enc, src_pix_fmt):
        if _encoder_supports_pix_fmt("libx265", src_pix_fmt):
            return "libx265"
        if _encoder_supports_pix_fmt("ffv1", src_pix_fmt):
            return "ffv1"
    return enc


def _choose_strict_lossless_encoder(
    *,
    target_container: str,
    src_codec: str,
    src_pix_fmt: Optional[str],
    desired_codec_key: Optional[str],
) -> LosslessChoice:
    cont = (target_container or "").lower()
    src = (src_codec or "").lower()
    want = _normalize_codec_key(desired_codec_key) if desired_codec_key else None

    def ok(enc: str) -> bool:
        return _encoder_supports_pix_fmt(enc, (src_pix_fmt or "yuv420p"))

    # Alte Sonderfälle beibehalten
    if want in {"mpeg4", "mpeg2video", "mpeg1video", "vp8"}:
        enc = "ffv1"
        return LosslessChoice(enc, _lossless_video_args(enc), True, [])

    # WEBM: (Re-)aktivierte strikt-lossless Wege für VP9/AV1
    if cont == "webm":
        if want in {None, "vp9"}:
            enc = "libvpx-vp9"
            return LosslessChoice(enc, _lossless_video_args(enc), True, [])
        if want == "av1":
            # libaom-av1 strikt, svt-av1 als (nicht-strikte) Ausweichmöglichkeit
            if ok("libaom-av1"):
                return LosslessChoice(
                    "libaom-av1", _lossless_video_args("libaom-av1"), True, []
                )
            if ok("libsvtav1"):
                return LosslessChoice(
                    "libsvtav1", _lossless_video_args("libsvtav1"), False, []
                )
            return LosslessChoice(
                "libaom-av1", _lossless_video_args("libaom-av1"), True, []
            )
        # Fallback: VP9
        return LosslessChoice(
            "libvpx-vp9", _lossless_video_args("libvpx-vp9"), True, []
        )

    # MP4/M4V/MOV
    if cont in ("mp4", "m4v", "mov"):
        if want == "av1":
            if ok("libaom-av1"):
                return LosslessChoice(
                    "libaom-av1", _lossless_video_args("libaom-av1"), False, []
                )
            if ok("libsvtav1"):
                return LosslessChoice(
                    "libsvtav1", _lossless_video_args("libsvtav1"), False, []
                )
            return LosslessChoice(
                "libaom-av1", _lossless_video_args("libaom-av1"), False, []
            )

        if want == "cineform" and ok("cfhd"):
            return LosslessChoice("cfhd", _lossless_video_args("cfhd"), True, [])

        # True lossless Codecs, die MOV/MP4 typischerweise tragen können
        if want in {"qtrle", "png"}:
            if want == "qtrle" and ok("qtrle"):
                return LosslessChoice("qtrle", _lossless_video_args("qtrle"), True, [])
            if want == "png" and ok("png"):
                return LosslessChoice("png", _lossless_video_args("png"), True, [])

        if cont == "mov" and want == "hap" and ok("hap"):
            return LosslessChoice("hap", _lossless_video_args("hap"), True, [])

        if want == "jpeg2000":
            if ok("jpeg2000"):
                return LosslessChoice(
                    "jpeg2000", _lossless_video_args("jpeg2000"), True, []
                )
            if ok("libopenjpeg"):
                return LosslessChoice(
                    "libopenjpeg", _lossless_video_args("libopenjpeg"), True, []
                )
            return LosslessChoice(
                "jpeg2000", _lossless_video_args("jpeg2000"), True, []
            )

        # HEVC (mit hvc1-Tag für mp4/m4v)
        extra_hevc_tag = ["-tag:v", "hvc1"] if cont in ("mp4", "m4v") else []
        if want == "hevc" or "hevc" in src or "265" in src:
            if ok("libx265"):
                return LosslessChoice(
                    "libx265", _lossless_video_args("libx265"), True, extra_hevc_tag
                )
            if ok("hevc_nvenc"):
                return LosslessChoice(
                    "hevc_nvenc",
                    _lossless_video_args("hevc_nvenc"),
                    False,
                    extra_hevc_tag,
                )
            return LosslessChoice("libx264", _lossless_video_args("libx264"), True, [])

        if want == "h264":
            if (
                src_pix_fmt
                and _encoder_supports_pix_fmt("libx264rgb", src_pix_fmt)
                and ok("libx264rgb")
            ):
                return LosslessChoice(
                    "libx264rgb", _lossless_video_args("libx264rgb"), False, []
                )
            return LosslessChoice("libx264", _lossless_video_args("libx264"), True, [])

        # Fallbacks
        if ok("libx264"):
            return LosslessChoice("libx264", _lossless_video_args("libx264"), True, [])
        if ok("h264_nvenc"):
            return LosslessChoice(
                "h264_nvenc", _lossless_video_args("h264_nvenc"), False, []
            )
        return LosslessChoice("libx264", _lossless_video_args("libx264"), True, [])

    # MKV / Matroska
    if cont in ("mkv", "matroska"):
        if want == "h264":
            if (
                src_pix_fmt
                and _encoder_supports_pix_fmt("libx264rgb", src_pix_fmt)
                and ok("libx264rgb")
            ):
                return LosslessChoice(
                    "libx264rgb", _lossless_video_args("libx264rgb"), False, []
                )
            if ok("libx264"):
                return LosslessChoice(
                    "libx264", _lossless_video_args("libx264"), True, []
                )
            if ok("h264_nvenc"):
                return LosslessChoice(
                    "h264_nvenc", _lossless_video_args("h264_nvenc"), False, []
                )
            return LosslessChoice("libx264", _lossless_video_args("libx264"), True, [])

        if want == "hevc":
            if ok("libx265"):
                return LosslessChoice(
                    "libx265", _lossless_video_args("libx265"), True, []
                )
            if ok("hevc_nvenc"):
                return LosslessChoice(
                    "hevc_nvenc", _lossless_video_args("hevc_nvenc"), False, []
                )
            return LosslessChoice("libx265", _lossless_video_args("libx265"), True, [])

        if want == "av1":
            if ok("libaom-av1"):
                return LosslessChoice(
                    "libaom-av1", _lossless_video_args("libaom-av1"), True, []
                )
            if ok("libsvtav1"):
                return LosslessChoice(
                    "libsvtav1", _lossless_video_args("libsvtav1"), False, []
                )
            return LosslessChoice(
                "libaom-av1", _lossless_video_args("libaom-av1"), True, []
            )

        if want == "vp9":
            enc = "libvpx-vp9"
            return LosslessChoice(enc, _lossless_video_args(enc), True, [])

        # Wieder hinzugefügte true-lossless Varianten für MKV
        if want in {
            "ffv1",
            "huffyuv",
            "utvideo",
            "rawvideo",
            "magicyuv",
            "ffvhuff",
            "png",
            "qtrle",
            "ljpeg",
            "jpeg2000",
        }:
            enc = _encoder_for_codec(want)
            return LosslessChoice(enc, _lossless_video_args(enc), True, [])

        if want == "jpeg2000":  # falls oben nicht gegriffen
            return LosslessChoice(
                "jpeg2000", ["-c:v", "jpeg2000", "-pred", "1"], True, []
            )

        if want:
            enc = _encoder_for_codec(want)
            return LosslessChoice(enc, _lossless_video_args(enc), True, [])

        enc = _pick_lossless_encoder_for_source(src, src_pix_fmt) or "ffv1"
        return LosslessChoice(enc, _lossless_video_args(enc), True, [])

    # AVI – zusätzliche alte Lossless-Kombinationen zurück
    if cont == "avi":
        if want in {
            "rawvideo",
            "utvideo",
            "huffyuv",
            "magicyuv",
            "ffvhuff",
            "ljpeg",
            "hap",
        }:
            enc = _encoder_for_codec(want)
            return LosslessChoice(enc, _lossless_video_args(enc), True, [])
        if want == "cineform" and ok("cfhd"):
            return LosslessChoice("cfhd", _lossless_video_args("cfhd"), True, [])
        # seltene, aber vorhandene lossless Varianten
        if want in {"zlib", "mszh"} and ok(want):
            return LosslessChoice(want, _lossless_video_args(want), True, [])
        # sinnvolle Defaults
        if ok("utvideo"):
            return LosslessChoice("utvideo", _lossless_video_args("utvideo"), True, [])
        if ok("huffyuv"):
            return LosslessChoice("huffyuv", _lossless_video_args("huffyuv"), True, [])
        if ok("ffvhuff"):
            return LosslessChoice("ffvhuff", _lossless_video_args("ffvhuff"), True, [])
        if ok("ljpeg"):
            return LosslessChoice("ljpeg", _lossless_video_args("ljpeg"), True, [])
        return LosslessChoice("rawvideo", _lossless_video_args("rawvideo"), True, [])

    # Generischer Fallback
    return LosslessChoice("ffv1", _lossless_video_args("ffv1"), True, [])


def _build_h264_near_lossless_chain(
    meta: Dict[str, Optional[str]],
    src_w: Optional[int],
    src_h: Optional[int],
    *,
    prefer_zscale: bool = True,
) -> Tuple[str, List[str]]:
    in_mat = _norm_matrix(meta.get("colorspace"))
    in_rng = _norm_range(meta.get("color_range"))
    out_mat = "bt709"
    out_rng = "limited"

    need_scale = bool((src_w or 0) % 2 or (src_h or 0) % 2)

    vf_parts: List[str] = []
    extra: List[str] = []

    if prefer_zscale and _has_filter("zscale"):
        if need_scale:
            vf_parts.append(
                f"zscale=width=trunc(iw/2)*2:height=trunc(ih/2)*2:"
                f"matrixin={in_mat}:matrix={out_mat}:rangein={in_rng}:range={out_rng}:"
                f"dither=none"
            )
        vf_parts.append("format=yuv420p")
    else:
        if need_scale:
            vf_parts.append(
                "scale=trunc(iw/2)*2:trunc(ih/2)*2:"
                "flags=+full_chroma_int+bitexact"
                f":in_color_matrix={in_mat}:out_color_matrix={out_mat}"
                f":in_range={in_rng}:out_range={out_rng}"
            )
            extra += ["-sws_flags", "+full_chroma_int+bitexact"]
        vf_parts.append("format=yuv420p")

    return ",".join(vf_parts), extra


def _mpeg4_near_lossless_args_mp4() -> List[str]:
    return [
        "-c:v",
        "mpeg4",
        "-qscale:v",
        "1",
        "-mbd",
        "2",
        "-trellis",
        "2",
        "-cmp",
        "2",
        "-subcmp",
        "2",
        "-g",
        "240",
        "-bf",
        "2",
    ]


def _build_mpeg4_near_lossless_chain(
    meta: Dict[str, Optional[str]],
    src_w: Optional[int],
    src_h: Optional[int],
    *,
    prefer_zscale: bool = True,
) -> Tuple[str, List[str]]:
    in_mat = _norm_matrix(meta.get("colorspace"))
    in_rng = _norm_range(meta.get("color_range"))
    out_mat = "bt709"
    out_rng = "limited"

    need_scale = bool((src_w or 0) % 2 or (src_h or 0) % 2)
    vf_parts: List[str] = []
    extra: List[str] = []

    if prefer_zscale and _has_filter("zscale"):
        if need_scale:
            vf_parts.append(
                f"zscale=width=trunc(iw/2)*2:height=trunc(ih/2)*2:"
                f"matrixin={in_mat}:matrix={out_mat}:rangein={in_rng}:range={out_rng}:dither=none"
            )
        vf_parts.append("format=yuv420p")
    else:
        if need_scale:
            vf_parts.append(
                "scale=trunc(iw/2)*2:trunc(ih/2)*2:"
                "flags=+full_chroma_int+bitexact"
                f":in_color_matrix={in_mat}:out_color_matrix={out_mat}"
                f":in_range={in_rng}:out_range={out_rng}"
            )
            extra += ["-sws_flags", "+full_chroma_int+bitexact"]
        vf_parts.append("format=yuv420p")

    return ",".join(vf_parts), extra


def _mpeg2_near_lossless_args() -> List[str]:
    return ["-c:v", "mpeg2video", "-qscale:v", "1", "-g", "15", "-bf", "2"]


def _mpeg1_near_lossless_args() -> List[str]:
    return ["-c:v", "mpeg1video", "-qscale:v", "1", "-g", "15"]


def _dedupe_vf_inplace(cmd: List[str]) -> None:
    """
    Vereinfacht und dedupliziert die -vf-Kette.
    Spezialfall VAAPI: 'format=nv12' unmittelbar VOR 'hwupload' MUSS bestehen bleiben.
    Alle weiteren 'format='-Filter nach 'hwupload' werden entfernt.
    Ohne 'hwupload' bleibt wie gehabt nur das letzte 'format=' erhalten.
    """
    if "-vf" not in cmd:
        return
    try:
        i = cmd.index("-vf")
        vf = str(cmd[i + 1])
    except Exception:
        return

    parts_in = [p.strip() for p in vf.split(",") if p.strip()]
    if not parts_in:
        return

    # Suche 'hwupload'
    try:
        hw_idx = parts_in.index("hwupload")
    except ValueError:
        hw_idx = -1

    out_parts: List[str] = []

    if hw_idx >= 0:
        # VAAPI-Fall: behalte das letzte 'format=' VOR hwupload (typisch 'format=nv12')
        pre_fmt_keep = None
        for idx, p in enumerate(parts_in[:hw_idx]):
            if p.startswith("format="):
                pre_fmt_keep = p

        for idx, p in enumerate(parts_in):
            if p.startswith("format="):
                # nur vor hwupload und genau der ausgewählte Eintrag
                if idx < hw_idx and p == pre_fmt_keep:
                    out_parts.append(p)
                else:
                    continue  # alle anderen 'format=' verwerfen
            else:
                out_parts.append(p)

    else:
        # Kein hwupload: Standardverhalten (nur letztes 'format=' behalten)
        last_format = None
        for p in parts_in:
            if p.startswith("format="):
                last_format = p
        seen: set[str] = set()
        for p in parts_in:
            if p.startswith("format="):
                # nur letztes behalten -> später anhängen
                continue
            sig = p.lower()
            if sig not in seen:
                seen.add(sig)
                out_parts.append(p)
        if last_format:
            out_parts.append(last_format)

    cmd[i + 1] = ",".join(out_parts)


# ============================================================
#                      HW / Probing
# ============================================================
@lru_cache(maxsize=1)
def _has_qsv_support(ffmpeg_bin: str = "ffmpeg") -> bool:
    try:
        out = subprocess.check_output(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        if "qsv" not in out.lower():
            return False
    except Exception:
        return False
    if os.name == "posix" and not glob.glob("/dev/dri/renderD*"):
        return False
    return True


def _vaapi_usable(dev: str) -> bool:
    try:
        rc = subprocess.call(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-init_hw_device",
                f"vaapi=va:{dev}",
                "-f",
                "lavfi",
                "-i",
                "color=size=16x16:rate=1:duration=0.1",
                "-vf",
                "format=nv12,hwupload",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return rc == 0
    except Exception:
        return False


def _first_vaapi_render_node() -> str | None:
    nodes = sorted(glob.glob("/dev/dri/renderD*"))
    return nodes[0] if nodes else None


def _inject_vaapi_device(ffmpeg_cmd: List[str]) -> bool:
    """
    Fügt VAAPI-Init nur hinzu, wenn verfügbar und noch nicht gesetzt.
    """
    toks = [str(t) for t in ffmpeg_cmd]
    if "-init_hw_device" in toks or "-filter_hw_device" in toks:
        return True
    dev = _first_vaapi_render_node()
    if not dev or not _vaapi_usable(dev):
        return False
    ffmpeg_cmd += ["-init_hw_device", f"vaapi=va:{dev}", "-filter_hw_device", "va"]
    return True


def _nvenc_usable(ffmpeg_bin: str = "ffmpeg") -> bool:
    try:
        rc = subprocess.call(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=16x16:rate=1:duration=0.1",
                "-t",
                "0.05",
                "-c:v",
                "hevc_nvenc",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return rc == 0
    except Exception:
        return False


@lru_cache(maxsize=4)
def _run_ffmpeg_encoders(ffmpeg_bin: str = "ffmpeg") -> str:
    """Liest die Encoderliste einmalig pro ffmpeg-Binary ein (Caching!)."""
    try:
        return subprocess.check_output(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return ""


def _parse_encoder_names(encoders_output: str) -> Set[str]:
    """Extrahiert nur echte Encoder-Namen zuverlässig aus der ffmpeg-Liste."""
    names: Set[str] = set()
    for line in encoders_output.splitlines():
        # Zeilen mit Encodern beginnen i.d.R. mit Leerzeichen und einem Fähnchenblock (VASF.)
        # Beispiel: " V..... libx264             libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"
        line = line.rstrip()
        if not line or not re.match(r"^\s*[VASF\.]+", line):
            continue
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def _pick_sw_av1_encoder(ffmpeg_bin: str = "ffmpeg") -> Optional[str]:
    txt = _run_ffmpeg_encoders(ffmpeg_bin)
    avail = set(_parse_encoder_names(txt))
    if "libsvtav1" in avail:
        return "libsvtav1"
    if "libaom-av1" in avail:
        return "libaom-av1"
    return None


def _strip_nvenc_private_opts(ffmpeg_cmd: List[str]) -> None:
    _strip_options(
        ffmpeg_cmd,
        {"-rc", "-cq", "-spatial-aq", "-aq-strength", "-preset", "-multipass", "-b:v"},
    )


def _build_video_preset_args_from_defs(
    presets: Mapping[str, PresetLike], preset_name: str, encoder: str, container: str
) -> List[str]:
    spec = _get_preset_spec(presets, preset_name)
    spd = str(spec.get("speed", "medium"))

    q_raw = spec.get("quality", 24)
    try:
        qual = int(q_raw) if q_raw is not None else 24
    except (TypeError, ValueError):
        qual = 24

    fam = _encoder_family(encoder)

    args: List[str] = []

    # ===== Lossless-Preset =====
    if spec.get("lossless"):
        ll_args = _lossless_video_args_for_encoder(encoder)
        lc = container.lower()
        ef = _encoder_family(encoder)
        if lc == "mp4" and ef == "hevc":
            ll_args += ["-tag:v", "hvc1"]
            if spec.get("faststart", False):
                ll_args += ["-movflags", "+faststart"]

        if ef == "h264":
            if lc in ("mp4", "m4v"):
                return _h264_near_lossless_args_mp4()
            if lc in ("mkv", "matroska", "mov"):
                return _h264_near_lossless_args_mkv()
        if lc == "flv":
            return ["-c:v", "flashsv2", "-g", "1"]
        return ll_args

    if fam == "h264":
        if encoder == "libx264":
            args += ["-preset", spd, "-crf", str(qual)]
            if container == "mp4" and spec.get("faststart", False):
                args += ["-movflags", "+faststart"]
        elif encoder.endswith("_nvenc"):
            pmap = {"ultrafast": "p4", "veryfast": "p5", "medium": "p6", "slow": "p7"}
            args += [
                "-preset",
                pmap.get(spd, "p5"),
                "-rc",
                "vbr",
                "-cq",
                str(qual),
                "-b:v",
                "0",
                "-spatial-aq",
                "1",
                "-aq-strength",
                "8",
            ]
            if container == "mp4" and spec.get("faststart", False):
                args += ["-movflags", "+faststart"]
        else:
            bps = {
                "slow": "8M",
                "medium": "6M",
                "veryfast": "3M",
                "ultrafast": "1.5M",
            }.get(spd, "4M")
            args += ["-b:v", bps]
            if container == "mp4" and spec.get("faststart", False):
                args += ["-movflags", "+faststart"]

    elif fam == "hevc":
        if encoder == "libx265":
            args += ["-preset", spd, "-crf", str(qual)]
            if container == "mp4":
                args += ["-tag:v", "hvc1"]
                if spec.get("faststart", False):
                    args += ["-movflags", "+faststart"]
        elif encoder.endswith("_nvenc"):
            pmap = {"ultrafast": "p4", "veryfast": "p5", "medium": "p6", "slow": "p7"}
            args += [
                "-preset",
                pmap.get(spd, "p5"),
                "-rc",
                "vbr",
                "-cq",
                str(qual),
                "-b:v",
                "0",
                "-spatial-aq",
                "1",
                "-aq-strength",
                "8",
            ]
            if container == "mp4":
                args += ["-tag:v", "hvc1"]
                if spec.get("faststart", False):
                    args += ["-movflags", "+faststart"]
        else:
            bps = {
                "slow": "6M",
                "medium": "4M",
                "veryfast": "2.5M",
                "ultrafast": "1.5M",
            }.get(spd, "3M")
            args += ["-b:v", bps]
            if container == "mp4":
                args += ["-tag:v", "hvc1"]
                if spec.get("faststart", False):
                    args += ["-movflags", "+faststart"]

    elif fam == "av1":
        if encoder == "libsvtav1":
            pmap = {"slow": "4", "medium": "6", "veryfast": "8", "ultrafast": "10"}
            args += ["-crf", str(qual), "-preset", pmap.get(spd, "6")]
        elif encoder == "libaom-av1":
            cmap = {"slow": "0", "medium": "2", "veryfast": "6", "ultrafast": "8"}
            args += ["-crf", str(qual), "-b:v", "0", "-cpu-used", cmap.get(spd, "4")]
        elif encoder.endswith("_nvenc"):
            pmap = {"ultrafast": "p4", "veryfast": "p5", "medium": "p6", "slow": "p7"}
            args += [
                "-preset",
                pmap.get(spd, "p5"),
                "-rc",
                "vbr",
                "-cq",
                str(qual),
                "-b:v",
                "0",
            ]
        else:
            bps = {
                "slow": "4M",
                "medium": "3M",
                "veryfast": "2M",
                "ultrafast": "1.2M",
            }.get(spd, "2.5M")
            args += ["-b:v", bps]
        if container == "mp4" and spec.get("faststart", False):
            args += ["-movflags", "+faststart"]

    elif fam == "vp9":
        if encoder == "libvpx-vp9":
            cpu = {"slow": "0", "medium": "2", "veryfast": "4", "ultrafast": "5"}.get(
                spd, "2"
            )
            args += ["-crf", str(qual), "-b:v", "0", "-row-mt", "1", "-cpu-used", cpu]
        else:
            bps = {
                "slow": "3M",
                "medium": "2.5M",
                "veryfast": "1.8M",
                "ultrafast": "1.2M",
            }.get(spd, "2M")
            args += ["-b:v", bps]

    elif fam == "vp8":
        cpu_map = {"ultrafast": "5", "veryfast": "4", "medium": "2", "slow": "0"}
        deadline = "best" if spd == "slow" else "good"
        cpu = cpu_map.get(spd, "2")

        if encoder in ("libvpx", "libvpx-v8"):
            args += [
                "-b:v",
                "0",
                "-crf",
                str(qual),
                "-deadline",
                deadline,
                "-cpu-used",
                cpu,
                "-auto-alt-ref",
                "1",
                "-lag-in-frames",
                "25",
                "-qmin",
                "0",
                "-qmax",
                "36",
                "-g",
                "240",
                "-keyint_min",
                "23",
                "-static-thresh",
                "0",
            ]
        elif encoder.endswith("_vaapi"):
            qp = max(10, min(30, int(qual)))
            args += ["-qp", str(qp), "-b:v", "0", "-g", "240"]
        else:
            bps = {
                "slow": "8M",
                "medium": "6M",
                "veryfast": "4M",
                "ultrafast": "3M",
            }.get(spd, "5M")
            args += ["-b:v", bps]

    elif fam == "prores":
        args += ["-profile:v", str(spec.get("profile", 3))]
        if "pix_fmt" in spec:
            args += ["-pix_fmt", spec["pix_fmt"]]

    elif fam == "mpeg4":
        if preset_name.lower() == "ultra" or qual <= 16:
            q = 1
        elif qual <= 20:
            q = 2
        else:
            q = 3
        args += [
            "-qscale:v",
            str(q),
            "-mbd",
            "2",
            "-trellis",
            "2",
            "-cmp",
            "2",
            "-subcmp",
            "2",
            "-g",
            "240",
            "-bf",
            "2",
        ]
        if container == "mp4" and spec.get("faststart", False):
            args += ["-movflags", "+faststart"]
        return args

    elif fam == "mjpeg":
        q = max(2, min(8, int(round((qual - 16) / 4))))
        args += ["-q:v", str(q)]

    elif fam == "mpeg2video":
        if qual <= 16:
            q = 1
        elif qual <= 20:
            q = 2
        else:
            q = 3
        args += [
            "-qscale:v",
            str(q),
            "-qmin",
            "1",
            "-qmax",
            str(max(3, q + 1)),
            "-g",
            "15",
            "-bf",
            "2",
        ]

    else:
        bps = {
            "slow": "6M",
            "medium": "4M",
            "veryfast": "2.5M",
            "ultrafast": "1.5M",
        }.get(spd, "3M")
        args += ["-b:v", bps]

    if encoder in ("libopenjpeg", "jpeg2000"):
        _strip_options(args, {"-b:v", "-vb", "-maxrate", "-minrate", "-crf"})
        if spec.get("lossless"):
            if "-c:v" in args:
                i = args.index("-c:v")
                args[i + 1] = "jpeg2000"
            else:
                args += ["-c:v", "jpeg2000"]
            args += ["-pred", "1"]
            return args

        qscale = max(1, min(8, int(round((qual - 16) / 4.0)) + 1))
        if "-c:v" in args:
            i = args.index("-c:v")
            args[i + 1] = "jpeg2000"
        else:
            args += ["-c:v", "jpeg2000"]
        args += ["-pred", "0", "-qscale:v", str(qscale)]
        return args

    # ---- optionale Pixelformat-Tendenzen aus Preset ----
    prefer_10bit = bool(spec.get("prefer_10bit", False))
    force_420 = bool(spec.get("force_yuv420p", False))  # z.B. Messenger/Web

    if "-pix_fmt" not in args:
        if force_420:
            # maximal kompatibel (Messenger/Web)
            args += ["-pix_fmt", "yuv420p"]
        elif prefer_10bit:
            lc = (container or "").lower()
            fam = _encoder_family(encoder)
            # MP4/Web → 420/10 wenn möglich (HEVC/AV1)
            if lc in ("mp4", "m4v"):
                for pf in ("yuv420p10le",):
                    if _encoder_supports_pix_fmt(encoder, pf):
                        args += ["-pix_fmt", pf]
                        break
            else:
                # MOV/MKV → 422/10 bevorzugt
                for pf in ("yuv422p10le", "yuv444p10le"):
                    if _encoder_supports_pix_fmt(encoder, pf):
                        args += ["-pix_fmt", pf]
                        break
    return args


def _replace_or_insert_vf_before_output(cmd: List[str], new_vf: str) -> None:
    out_idx = _output_index(cmd)
    # Falls -vf bereits existiert (vor dem Output), ersetze dort
    if "-vf" in cmd[:out_idx]:
        i = cmd.index("-vf")
        cmd[i + 1] = new_vf
    else:
        _insert_before_output(cmd, ["-vf", new_vf])
    _dedupe_vf_inplace(cmd)


def _container_supports_chapters(container: str) -> bool:
    """
    Konservativ: Kapitel sinnvoll in MKV/Matroska, MOV/MP4/M4V.
    (AVI/FLV/MPEG lassen wir bewusst aus, um Edgecases zu vermeiden.)
    """
    c = (container or "").lower()
    return c in {"mkv", "matroska", "mov", "mp4", "m4v"}


def _input_has_chapters(
    input_path: Optional[Path], ffprobe_bin: str = "ffprobe"
) -> bool:
    """
    Prüft mit ffprobe, ob Kapitel vorhanden sind. So vermeiden wir nutzlose
    -map_chapters 0 Aufrufe/Warnungen bei kapitel-losen Quellen.
    """
    if not input_path:
        return False
    try:
        out = subprocess.check_output(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_chapters",
                str(input_path),
            ],
            text=True,
        )
        data = cast(Dict[str, Any], json.loads(out))
        ch = data.get("chapters")
        return isinstance(ch, list) and len(ch) > 0
    except Exception:
        return False


def _probe_subtitle_streams(
    input_path: Path, ffprobe_bin: str = "ffprobe"
) -> List[Tuple[int, str, str]]:
    """
    Liefert [(index, codec_name_lower, codec_tag_lower)] für alle Subtitle-Streams.
    codec_name/tag können leer sein -> dann ''.
    """
    if not input_path:
        return []
    try:
        out = subprocess.check_output(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name,codec_tag_string",
                "-of",
                "json",
                str(input_path),
            ],
            text=True,
        )
        data = cast(Dict[str, Any], json.loads(out))
        streams = cast(List[Dict[str, Any]], data.get("streams") or [])
        res: List[Tuple[int, str, str]] = []
        for st in streams:
            idx = st.get("index")
            if not isinstance(idx, int):
                continue
            cname = (st.get("codec_name") or "").strip().lower()
            ctag = (st.get("codec_tag_string") or "").strip().lower()
            res.append((idx, cname, ctag))
        return res
    except Exception:
        return []


# === H.263/FLV: erlaubte Größen erzwingen ==============================
def _enforce_h263_size(
    container: str, codec_key: str, src_w: int | None, src_h: int | None
) -> Tuple[Optional[str], Optional[str]]:
    c = (container or "").lower()
    k = (codec_key or "").lower()
    if c != "flv" or k != "h263":
        return None, None

    allowed = [(128, 96), (176, 144), (352, 288), (704, 576), (1408, 1152)]
    sw = int(src_w or 0)
    sh = int(src_h or 0)
    if sw <= 0 or sh <= 0:
        w, h = (704, 576)
    else:
        cand = [s for s in allowed if s[0] <= sw and s[1] <= sh]
        w, h = max(cand, key=lambda t: t[0] * t[1]) if cand else (704, 576)

    want_169 = False
    try:
        want_169 = abs((sw / sh) - (16 / 9)) < 0.1
    except Exception:
        pass

    if (w, h) == (704, 576) and want_169:
        sar = "16/11"
    else:
        sar = "1"

    vf = f"scale={w}:{h},setsar={sar}"
    return vf, "yuv420p"


def _parse_scale(scale: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not scale or ":" not in str(scale):
        return None, None
    a, b = str(scale).split(":", 1)
    try:
        return int(a), int(b)
    except Exception:
        return None, None


def _append_vf(cmd: List[str], new_filter: str) -> None:
    if not new_filter:
        return
    out_idx = _output_index(cmd)
    if "-vf" in cmd[:out_idx]:
        i = cmd.index("-vf")
        current = str(cmd[i + 1])
        cmd[i + 1] = (current + "," + new_filter) if current else new_filter
    else:
        _insert_before_output(cmd, ["-vf", new_filter])
    _dedupe_vf_inplace(cmd)


def _strip_filter_from_vf(vf: str, key_prefix: str) -> str:
    key = key_prefix if key_prefix.endswith("=") else key_prefix + "="
    parts = [p for p in vf.split(",") if not p.strip().startswith(key)]
    return ",".join(parts) if parts else ""


def _set_format_in_vf(cmd: List[str], fmt: str) -> None:
    if not fmt:
        return
    # vorhandene -vf-Kette holen/ersetzen
    out_idx = _output_index(cmd)
    if "-vf" in cmd[:out_idx]:
        i = cmd.index("-vf")
        vf = str(cmd[i + 1])
        parts = [p for p in vf.split(",") if not p.strip().startswith("format=")]
        parts.append(f"format={fmt}")
        cmd[i + 1] = ",".join(parts)
        _dedupe_vf_inplace(cmd)
    else:
        _insert_before_output(cmd, ["-vf", f"format={fmt}"])


def _vaapiify_vf(vf_chain: str | None, want_scale: bool) -> str:
    vf = vf_chain or ""
    if "scale=" in vf:
        vf = vf.replace("scale=", "scale_vaapi=")
    if "format=" in vf:
        vf = _strip_filter_from_vf(vf, "format=")
    vf = "format=nv12,hwupload" + ("," if vf else "") + vf
    return vf


def _fix_orphaned_vf_inplace(cmd: List[str]) -> None:
    """
    Stellt sicher, dass '-vf' einen Wert hat. Sucht notfalls einen plausiblen
    Filtergraph im Command und hängt ihn an; andernfalls entfernt '-vf'.
    """
    try:
        while "-vf" in cmd:
            i = cmd.index("-vf")
            need = (i + 1 >= len(cmd)) or str(cmd[i + 1]).startswith("-")
            if not need:
                break
            # Suche einen plausiblen Filtergraph irgendwo in cmd
            val_idx = None
            for k, t in enumerate(cmd):
                if k in (i, i + 1):
                    continue
                s = str(t)
                if s.startswith("-"):
                    continue
                ls = s.lower()
                if ("=" in ls) or ls.startswith(
                    ("scale", "zscale", "fps", "format", "setsar", "setdar", "hwupload")
                ):
                    val_idx = k
                    break
            if val_idx is not None:
                val = cmd.pop(val_idx)
                cmd.insert(i + 1, str(val))
            else:
                # Lieber entfernen als ein kaputtes '-vf' stehen lassen
                del cmd[i]
    except Exception:
        pass


def _nearest_nominal_fps(fps: float | None) -> float:
    if not fps:
        return 25.0
    choices = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]
    return min(choices, key=lambda x: abs(x - fps))


# === -movflags immer mit Wert zusammenhalten ==========================
def _sanitize_movflags_inplace(cmd: List[str]) -> None:
    """
    Sorgt dafür, dass '-movflags' direkt einen Wert (z.B. '+faststart') hat.
    Einsame '+faststart' werden an das nächstliegende '-movflags' gehängt oder
    zu einem neuen Paar gemacht. Doppelte '+faststart' werden entfernt.
    """
    if not cmd:
        return

    # 1) Einsame '+faststart' markieren
    fast_idx: List[int] = [i for i, t in enumerate(cmd) if str(t) == "+faststart"]

    # 2) Für jedes '-movflags' sicherstellen, dass direkt danach ein Wert steht
    i = 0
    used_fast: set[int] = set()
    while i < len(cmd):
        if cmd[i] != "-movflags":
            i += 1
            continue
        val_idx = i + 1
        need_val = (val_idx >= len(cmd)) or str(cmd[val_idx]).startswith("-")
        if need_val:
            # Suche ein bereits vorhandenes '+faststart' (vorzugsweise später)
            pick: Optional[int] = None
            for j in fast_idx:
                if j in used_fast:
                    continue
                pick = j
                break
            if pick is not None:
                val = cmd.pop(pick)
                if pick < i:  # Verschiebung korrigieren
                    i -= 1
                cmd.insert(i + 1, val)
                used_fast.add(i + 1)
            else:
                # Kein existierender Wert -> Default
                cmd.insert(i + 1, "+faststart")
        i += 2  # springe über Paar hinweg

    # 3) Falls '+faststart' ohne '-movflags' existiert → davor '-movflags' setzen
    # (nur das erste behalten, weitere löschen)
    # Sammle verbliebene einsame '+faststart'
    lonely = [
        k
        for k in range(len(cmd))
        if cmd[k] == "+faststart" and (k == 0 or cmd[k - 1] != "-movflags")
    ]
    if lonely:
        first = lonely[0]
        cmd.insert(first, "-movflags")
        # Übrige entfernen
        for k in reversed(lonely[1:]):
            del cmd[k]

    # 4) Doppelte '+faststart' hinter *demselben* -movflags entfernen
    i = 0
    while i < len(cmd) - 2:
        if cmd[i] == "-movflags" and cmd[i + 1] == "+faststart":
            # lösche weitere '+faststart' direkt hinter diesem Paar
            j = i + 2
            while j < len(cmd) and cmd[j] == "+faststart":
                del cmd[j]
            i += 2
        else:
            i += 1


# ============================================================
#                     DNx-* Sonderfälle
# ============================================================


def _replace_audio_with_pcm_if_mov(cmd: List[str], container: str):
    if container.lower() != "mov":
        return
    _strip_options(cmd, {"-b:a"})
    _set_kv_arg(cmd, "-c:a", "pcm_s16le")
    if "-ac" not in cmd:
        cmd += ["-ac", "2"]


def _apply_dnx_rules_if_needed(
    final_cmd: List[str],
    codec_key: str,
    container: str,
    target_w: int,
    target_h: int,
    src_fps: float | None,
    preset_name: Optional[str] = None,
) -> List[str]:
    ck = (codec_key or "").lower()
    if ck not in ("dnxhd", "dnxhr", "dnx"):
        return final_cmd

    _set_kv_arg(final_cmd, "-c:v", "dnxhd")
    _strip_options(final_cmd, {"-b:v", "-vb", "-crf"})

    W, H = int(target_w), int(target_h)
    nfps = _nearest_nominal_fps(src_fps)

    is_dnxhd_size = (W, H) in {(1920, 1080), (1280, 720), (1440, 1080)}
    if is_dnxhd_size:
        # DNxHD: feste Profile/Bitraten
        _strip_options(final_cmd, {"-profile:v"})
        _set_kv_arg(final_cmd, "-pix_fmt", "yuv422p")
        if (W, H) == (1920, 1080):
            bv = "220M" if nfps >= 50.0 else "145M"
        elif (W, H) == (1280, 720):
            bv = "90M" if nfps >= 50.0 else "60M"
        else:
            bv = "145M"
        _set_kv_arg(final_cmd, "-b:v", bv)
    else:
        # DNxHR
        want_10bit = preset_name == "cinema"
        if want_10bit:
            _set_kv_arg(final_cmd, "-profile:v", "dnxhr_hqx")
            _set_kv_arg(final_cmd, "-pix_fmt", "yuv422p10le")
        else:
            _set_kv_arg(final_cmd, "-profile:v", "dnxhr_hq")
            _set_kv_arg(final_cmd, "-pix_fmt", "yuv422p")
        _strip_options(final_cmd, {"-b:v", "-vb"})

    _replace_audio_with_pcm_if_mov(final_cmd, container)
    return final_cmd


def _wants_mezzanine_dnx(container: str, codec_key: str) -> bool:
    k = _normalize_codec_key(codec_key or "") or ""
    return (container or "").lower() == "mov" and k in {"dnxhd", "dnxhr", "dnx"}


def _probe_video_codec(path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _hw_encoder_usable(enc: str) -> bool:
    core, hw = _encoder_core_hw(enc)
    if hw == "qsv":
        return _has_qsv_support()
    if hw == "vaapi":
        node = _first_vaapi_render_node()
        return bool(node and _vaapi_usable(node))
    return True


def _policy_candidates_same_family(
    want_core: str, preferred_encoder: Optional[str], available: set[str]
) -> List[str]:
    want_core = (want_core or "").lower()
    out: List[str] = []
    if preferred_encoder and preferred_encoder in available:
        core_i, _ = _encoder_core_hw(preferred_encoder)
        if core_i == want_core:
            out.append(preferred_encoder)

    for enc in defin.CODEC_FALLBACK_POLICY.get(want_core, []):
        if enc in available and _encoder_family(enc) == want_core and enc not in out:
            out.append(enc)

    base = _encoder_for_codec(want_core)
    if base in available and base not in out:
        out.append(base)

    for enc in sorted(available):
        if _encoder_family(enc) == want_core and enc not in out:
            out.append(enc)

    out = [e for e in out if _hw_encoder_usable(e)]
    return out


def _preset_or_arg(spec: Mapping[str, Any], key: str, arg_val: Any) -> Any:
    """
    Gibt bevorzugt den vom Nutzer übergebenen arg_val zurück; andernfalls den Preset-Wert.
    Typing-Fix: spec ist ein Mapping[str, Any], damit Pylance nicht mit PresetLike meckert.
    """
    return arg_val if arg_val is not None else spec.get(key)


def _encoder_core_hw(encoder: str) -> Tuple[str, str]:
    e = (encoder or "").lower()
    if e.endswith("_qsv"):
        hw = "qsv"
    elif e.endswith("_nvenc"):
        hw = "nvenc"
    elif e.endswith("_vaapi"):
        hw = "vaapi"
    elif e.endswith("_amf"):
        hw = "amf"
    elif e.endswith("_videotoolbox"):
        hw = "vtb"
    else:
        hw = "sw"

    if e.startswith("h264_") or e == "libx264":
        core = "h264"
    elif e.startswith("hevc_") or e == "libx265":
        core = "hevc"
    elif e.startswith("av1_") or e in ("libsvtav1", "libaom-av1", "rav1e"):
        core = "av1"
    elif e in ("libvpx-vp9", "vp9_qsv", "vp9_vaapi"):
        core = "vp9"
    elif e in ("libvpx-vp8", "vp8_qsv", "vp8_vaapi"):
        core = "vp8"
    else:
        core = "other"
    return core, hw


def _required_alignment(enc: str) -> int:
    core, hw = _encoder_core_hw(enc or "")
    if hw == "qsv":
        return 16
    if hw == "nvenc":
        return 8 if core == "av1" else 2
    if hw in ("vaapi", "amf", "vtb"):
        return 2
    if core in ("vp9", "vp8", "h264", "hevc", "av1"):
        return 2
    return 2


# ============================================================
#                  Try & Fallback Encoding
# ============================================================
def _codec_core_from_key(codec_key: str) -> str:
    ck = (codec_key or "").lower()
    if ck in ("mpeg2", "mpeg2video"):
        return "mpeg2video"
    if ck in ("mpeg1", "mpeg1video"):
        return "mpeg1video"
    return ck


def _build_cmd_for_encoder(
    base_cmd: List[str],
    encoder: str,
    preset_name: str,
    container: str,
    vf_chain: Optional[str],
) -> List[str]:
    cmd = base_cmd[:]
    if "-c:v" in cmd:
        j = cmd.index("-c:v")
        cmd[j + 1] = encoder
    else:
        cmd += ["-c:v", encoder]
    cmd += _build_video_preset_args_from_defs(
        defin.CONVERT_PRESET,
        preset_name=preset_name,
        encoder=encoder,
        container=container,
    )
    if vf_chain:
        _replace_or_insert_vf_before_output(cmd, vf_chain)

    return cmd


def _append_if_absent(vf: str, frag: str) -> str:
    key = frag.split("=")[0]
    if any(p.strip().startswith(key + "=") or p.strip() == key for p in vf.split(",")):
        return vf
    return vf + ("," if vf else "") + frag


def _inject_qsv_device(
    ffmpeg_cmd: List[str],
    chosen_ffmpeg_encoder: Optional[str] = None,
    ffmpeg_bin="ffmpeg",
):
    enc = (chosen_ffmpeg_encoder or "").lower()
    if not enc.endswith("_qsv"):
        return
    if not _has_qsv_support(ffmpeg_bin):
        return
    if "-init_hw_device" in ffmpeg_cmd:
        return
    ffmpeg_cmd += ["-init_hw_device", "qsv=hw", "-filter_hw_device", "hw"]


def _try_quick_encode(cmd: List[str]) -> bool:
    """
    Führt einen 1-Sekunden-Encode gegen eine temporäre Datei aus.
    Erwartet 'cmd' inkl. (Dummy)-Output am Ende – wie im Aufrufer bereits üblich.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mkv")
    path = tmp.name
    try:
        tmp.close()
    except Exception:
        pass

    test_cmd = cmd[:-1] + ["-y", "-t", "1", path]
    try:
        rc = subprocess.call(
            test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return rc == 0
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def _strip_hw_device_options(ffmpeg_cmd: List[str]) -> None:
    """
    Entfernt alle HW-Device-Optionen, ohne Index-Fehler zu riskieren.
    """
    _strip_options(
        ffmpeg_cmd,
        {"-init_hw_device", "-filter_hw_device", "-hwaccel", "-hwaccel_device"},
    )
