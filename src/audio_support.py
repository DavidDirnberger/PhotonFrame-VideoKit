#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

from VideoEncodersCodecs_support import _get_preset_spec

import definitions as defin


def _as_int_or_none(val: Any) -> Optional[int]:
    """Versucht val nach int zu casten, sonst None."""
    try:
        if val is None:
            return None
        # bool absichern, falls jemand True/False in Presets hat
        if isinstance(val, bool):
            return int(val)
        return int(str(val))
    except Exception:
        return None


def _load_preset_spec(preset_name: Optional[str]) -> Dict[str, Any]:
    """
    Holt ein Preset als Dict[str, Any]. Typisiert so,
    dass Pylance/pyright nicht 'Unknown' propagiert.
    """
    spec: Dict[str, Any] = {}
    if preset_name:
        try:
            # _get_preset_spec liefert Mapping[str, Any] (PresetLike)
            loaded: Mapping[str, Any] = cast(
                Mapping[str, Any], _get_preset_spec(defin.CONVERT_PRESET, preset_name)
            )
            # in ein echtes Dict kopieren, damit .get klar typisierbar ist
            spec = dict(loaded)
        except Exception:
            spec = {}
    return spec


def _probe_audio_codec(path: Path) -> Tuple[Optional[str], Optional[int]]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,channels",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
        data = cast(Dict[str, Any], json.loads(out))
        streams = cast(List[Dict[str, Any]], data.get("streams") or [])
        st: Dict[str, Any] = streams[0] if streams else {}
        codec = st.get("codec_name")
        channels = st.get("channels")
        return (cast(Optional[str], codec), cast(Optional[int], channels))
    except Exception:
        return None, None


# ============================================================
#                      Audio-Args
# ============================================================


def _avi_audio_args_for_source(
    input_path: Path, preset_name: Optional[str]
) -> List[str]:
    """
    AVI-Verhalten bleibt wie gehabt; lediglich Typisierung ist klarer.
    """
    codec, _ = _probe_audio_codec(input_path)
    avi_safe = {"pcm_s16le", "pcm_s24le", "mp3", "ac3"}
    if codec and codec.lower() in avi_safe:
        return ["-c:a", "copy"]
    return ["-c:a", "pcm_s16le"]


def _webm_audio_args_for_source(
    input_path: Path,
    preset_name: Optional[str],
    *,
    force_stereo_if_unspecified: bool = True,
) -> List[str]:
    """
    Liefert saubere Opus-Default-Settings für WebM.
    Streng typisiert, um Pylance-'Unknown'-Warnungen zu vermeiden.
    """
    spec: Dict[str, Any] = _load_preset_spec(preset_name)

    default_bps = "160k"
    # Spezifische Bitrate aus Preset oder Fallback
    raw_bitrate: Any = spec.get("audio_bitrate", default_bps)
    target_bitrate: str = str(raw_bitrate if raw_bitrate else default_bps)

    # Wunschkanäle aus Preset ziehen und sauber in Optional[int] mappen
    desired_ch: Optional[int] = _as_int_or_none(spec.get("audio_channels"))
    if desired_ch is None and force_stereo_if_unspecified:
        desired_ch = 2

    # Quelle prüfen
    _acodec, _ch = _probe_audio_codec(input_path)
    src_ch: int = _as_int_or_none(_ch) or 2

    # Wenn Mono/Stereo explizit gewünscht → hart setzen
    if desired_ch in (1, 2):
        return ["-c:a", "libopus", "-b:a", target_bitrate, "-ac", str(int(desired_ch))]

    # Sonst: Standard auf libopus, bei >2 Kanälen Mapping-Familie 1 (Surround) erlauben
    args: List[str] = ["-c:a", "libopus", "-b:a", target_bitrate]
    if src_ch > 2:
        args += ["-mapping_family", "1"]
    return args


def build_audio_args(
    container: str, preset_name: Optional[str] = None, input_path: Optional[Path] = None
) -> List[str]:
    """
    Allgemeiner Audio-Arg-Builder mit klaren Typen und robustem Downmix.
    Fix: 'audio_bitrate' aus Presets wird strikt validiert (nur numerisch).
         Ungültige Werte (z.B. 'pcm') werden verworfen → Default-Bitrate.
    """
    spec: Dict[str, Any] = _load_preset_spec(preset_name)
    c = (container or "").lower()

    # Container-Defaults
    default_bps_by_container: Dict[str, str] = {
        "webm": "160k",
        "mp4": "192k",
        "m4v": "192k",
        "mov": "192k",
        "avi": "384k",
        "mpeg": "192k",
        "flv": "160k",
    }

    # --- Bitrate aus Preset prüfen ---
    def _is_numeric_bitrate(v: Any) -> bool:
        if v is None:
            return False
        s = str(v).strip().lower()
        # erlaubt: 128k, 192k, 3m, 256000
        return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:[kKmM])?", s))

    preset_bitrate_any: Any = spec.get("audio_bitrate")
    preset_bitrate: Optional[str] = (
        str(preset_bitrate_any).strip()
        if _is_numeric_bitrate(preset_bitrate_any)
        else None
    )

    def pick_bitrate(cont: str) -> str:
        return preset_bitrate or default_bps_by_container.get(cont, "192k")

    # gewünschte Kanalzahl (nur 1/2 erzwingen wir hart)
    desired_ch: Optional[int] = _as_int_or_none(spec.get("audio_channels"))

    # --- Container-spezifisch ---
    if c in ("mp4", "m4v", "mov"):
        args: List[str] = ["-c:a", "aac"]
        # Bitrate nur setzen, wenn numerisch
        bps = pick_bitrate(c)
        if _is_numeric_bitrate(bps):
            args += ["-b:a", bps]
        if desired_ch in (1, 2):
            args += ["-ac", str(int(desired_ch))]
            return args
        if input_path:
            _codec, _ch = _probe_audio_codec(input_path)
            src_ch: int = _as_int_or_none(_ch) or 2
            if src_ch > 2:
                args += ["-ac", "2"]
        return args

    if c == "webm":
        # Copy nur, wenn schon Opus ≤ Stereo und keine explizite Kanaländerung
        if input_path:
            acodec, _ch = _probe_audio_codec(input_path)
            ch_src: int = _as_int_or_none(_ch) or 2
            if (
                (acodec or "").lower() == "opus"
                and ch_src <= 2
                and desired_ch in (None, 1, 2)
            ):
                if desired_ch in (1, 2) and ch_src != desired_ch:
                    # Kanäle ändern → neu encoden
                    pass
                else:
                    return ["-c:a", "copy"]

        args = ["-c:a", "libopus"]
        bps = pick_bitrate(c)
        if _is_numeric_bitrate(bps):
            args += ["-b:a", bps]
        if desired_ch in (1, 2):
            args += ["-ac", str(int(desired_ch))]
        else:
            # Web-Kompatibilität: Stereo erzwingen, wenn unbekannt/>2
            args += ["-ac", "2"]
        return args

    if c == "avi":
        bps = pick_bitrate(c)
        args = ["-c:a", "ac3"]
        if _is_numeric_bitrate(bps):
            args += ["-b:a", bps]
        return args

    if c == "mpeg":
        # klassisch MP2 @48kHz/stereo
        return ["-c:a", "mp2", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

    if c == "flv":
        return ["-c:a", "aac", "-b:a", pick_bitrate(c), "-ar", "44100", "-ac", "2"]

    # Default: passthrough
    return ["-c:a", "copy"]
