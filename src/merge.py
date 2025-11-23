#!/usr/bin/env python3
# merge.py
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import consoleOutput as co
import definitions as defin
import helpers as he
import process_wrappers as pw
import userInteraction as ui
import video_thumbnail as vt

# Plan-API (nur für Encoder-Mapping & Container-Quirks)
import VideoEncodersCodecs as vec
from ffmpeg_perf import autotune_final_cmd

# Local modules
from i18n import _, tr

StrPath = Union[str, os.PathLike[str], Path]

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten / Endungen
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS: Tuple[str, ...] = tuple(
    sorted(str(x).lower() for x in defin.VIDEO_EXTENSIONS)
)
AUDIO_EXTS: Tuple[str, ...] = tuple(
    sorted(str(x).lower() for x in defin.AUDIO_EXTENSIONS)
)
SUB_EXTS: Tuple[str, ...] = tuple(sorted(str(x).lower() for x in defin.SUB_EXTENSIONS))


# ──────────────────────────────────────────────────────────────────────────────
# Argumente
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MergeArgs:
    files: List[str] = field(default_factory=list)
    # 'match-first' | 'no-scale' | 'smallest' | 'average' | 'largest' | 'fixed'
    target_res: Optional[str] = None
    offset: Optional[str] = (
        "0"  # Start-Offset (Video+Video: vor erstem Clip; Video+Audio/Subs: Offset des „anderen“)
    )
    audio_offset: Optional[str] = "0"  # Offset für zusätzliche Audiospuren
    subtitle_offset: Optional[str] = "0"  # Offset für zusätzliche Untertitelspuren
    pause: Optional[str] = "0"  # Pause zwischen Clips (nur Video+Video)
    burn_subtitle: Optional[bool] = False
    subtitle_name: Optional[str] = None
    audio_name: Optional[str] = None  # optionaler Titel für neu hinzugefügte Audiospur
    extend: Optional[bool] = None
    format: Optional[str] = None
    codec: Optional[str] = None
    preset: Optional[str] = None
    output: Optional[str] = None
    outro: Optional[str] = "500ms"


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen (Probe / Auswahl)
# ──────────────────────────────────────────────────────────────────────────────


def _ffprobe_streams_json(path: str, select: Optional[str] = None) -> Dict[str, Any]:
    """
    Liefert das ffprobe-JSON für Streams. Wenn select gesetzt ist (z.B. 's'),
    wird -select_streams verwendet.
    """
    cmd = ["ffprobe", "-v", "error"]
    if select:
        cmd += ["-select_streams", str(select)]
    cmd += [
        "-show_entries",
        "stream=index,codec_name,codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        return cast(Dict[str, Any], json.loads(out))
    except Exception:
        return {"streams": []}


def _count_streams_of_type(path: str, typ: str) -> int:
    """
    Zählt Streams mit codec_type==typ ('audio'|'subtitle'|...).
    """
    data = _ffprobe_streams_json(path, select=None)
    streams = cast(List[Dict[str, Any]], data.get("streams") or [])
    return sum(1 for s in streams if (s.get("codec_type") or "").lower() == typ.lower())


def _probe_sub_count(path: str) -> int:
    """
    Ersetzt die alte CSV-Variante: wirklich nur Untertitel zählen.
    """
    return _count_streams_of_type(path, "subtitle")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Alpha-Warnung
# ──────────────────────────────────────────────────────────────────────────────


def _warn_alpha_drop(container: str, codec_key: str) -> None:
    container_obj = (
        getattr(defin, "CONVERT_FORMAT_DESCRIPTIONS", {}).get(container, {})
        if hasattr(defin, "CONVERT_FORMAT_DESCRIPTIONS")
        else {}
    ) or {}
    codec_obj = (
        getattr(defin, "VIDEO_CODECS", {}).get(codec_key, {})
        if hasattr(defin, "VIDEO_CODECS")
        else {}
    ) or {}

    cont_label = str(container_obj.get("name", container) or container)
    codec_label = str(codec_obj.get("name", codec_key) or codec_key)

    co.print_warning(
        tr(
            {
                "de": (
                    f"Quelle enthält Transparenz (Alpha). Die gewählte Kombination "
                    f"{cont_label}/{codec_label} unterstützt keinen Alphakanal – Transparenz wird verworfen. "
                    f"Tipp: MOV + ProRes 4444, MKV/WebM + VP9/AV1 (Alpha) oder MKV/AVI + UtVideo (RGBA)."
                ),
                "en": (
                    f"Source contains transparency (alpha). The chosen combination "
                    f"{cont_label}/{codec_label} does not support alpha — transparency will be dropped. "
                    f"Tip: MOV + ProRes 4444, MKV/WebM + VP9/AV1 (alpha), or MKV/AVI + UtVideo (RGBA)."
                ),
            }
        )
    )


def _choose_target_size(videos: Sequence[str], mode: str) -> Tuple[int, int]:
    sizes = [he.probe_wh(v) for v in videos]
    if not sizes:
        return (1, 1)
    if mode == "smallest":
        return (min(w for w, _ in sizes), min(h for _, h in sizes))
    if mode in ("largest", "no-scale"):
        return (max(w for w, _ in sizes), max(h for _, h in sizes))
    return (
        sum(w for w, _ in sizes) // len(sizes),
        sum(h for _, h in sizes) // len(sizes),
    )


def _detect_target_fps_min(videos: Sequence[str]) -> int:
    """Kleinstes sinnvolles FPS – auf common Rates gerundet."""
    vals: List[float] = []
    for v in videos:
        _, _, _, fps = he.probe_wh_fmt_fps(v)
        if fps > 0.5:
            vals.append(fps)
    if not vals:
        try:
            mn, _mx = he.get_framerate_range(videos)
            if isinstance(mn, (int, float)) and mn > 0:
                vals.append(float(mn))
        except Exception:
            pass
    if not vals:
        return 25
    cand = min(vals)
    common = [12, 15, 23.976, 24, 25, 29.97, 30, 50, 59.94, 60]
    for c in common:
        if abs(cand - c) < 0.3:
            return int(round(c))
    return int(round(cand))


def _build_video_quality_args(encoder: str, preset_name: Optional[str]) -> List[str]:
    """
    Gibt Encoder-Qualitäts-/Speed-Flags passend zum Preset zurück.
    Konservativ (funktioniert „breit“); wenn etwas nicht passt, nutzt ffmpeg Default.
    """
    if not preset_name:
        return []
    p = preset_name.lower()
    enc = (encoder or "").lower()

    # libx264
    if enc in ("libx264", "h264"):
        crf = {
            "lossless": "0",
            "max": "16",
            "hq": "18",
            "casual": "20",
            "fast": "22",
            "tiny": "28",
        }.get(p, "20")
        speed = {
            "lossless": "veryslow",
            "max": "slow",
            "hq": "medium",
            "casual": "medium",
            "fast": "faster",
            "tiny": "veryfast",
        }.get(p, "medium")
        args = ["-preset", speed, "-crf", crf]
        if p == "lossless":
            # QP 0 ist bei x264 „technisch“ lossless, CRF 0 ebenso – CRF 0 genügt
            pass
        return args

    # libx265
    if enc in ("libx265", "hevc"):
        crf = {
            "lossless": "0",
            "max": "18",
            "hq": "20",
            "casual": "22",
            "fast": "26",
            "tiny": "30",
        }.get(p, "22")
        speed = {
            "lossless": "veryslow",
            "max": "slow",
            "hq": "medium",
            "casual": "medium",
            "fast": "faster",
            "tiny": "veryfast",
        }.get(p, "medium")
        args = ["-preset", speed, "-crf", crf]
        if p == "lossless":
            args += ["-x265-params", "lossless=1"]
        return args

    # NVENC (moderne p1..p7 Presets; fällt auf Defaults zurück wenn nicht vorhanden)
    if enc in ("h264_nvenc", "hevc_nvenc"):
        pmap = {
            "lossless": "p7",
            "max": "p7",
            "hq": "p5",
            "casual": "p4",
            "fast": "p3",
            "tiny": "p1",
        }
        args = ["-preset", pmap.get(p, "p4")]
        if p == "lossless":
            # konstante Quantisierung, visuell lossless
            args += ["-rc", "constqp", "-qp", "0"]
        return args

    # andere Encoder: keine spezifischen zusätzlichen Flags
    return []


def _maybe_add_faststart(
    cmd: List[str], container: str, preset_name: Optional[str]
) -> List[str]:
    try:
        if container.lower() == "mp4" and preset_name:
            pdef = defin.CONVERT_PRESET.get(preset_name, {}) or {}
            if pdef.get("faststart"):
                # vor Output, nach Codecs
                # (wir hängen es am Ende vor die Ausgabedatei an – wird später korrekt einsortiert)
                cmd += ["-movflags", "+faststart"]
    except Exception:
        pass
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Pixelformat / Alpha Policy
# ──────────────────────────────────────────────────────────────────────────────


def _pix_fmt_has_alpha(p: Optional[str]) -> bool:
    if not p:
        return False
    p = p.lower()
    if p.startswith(("yuva", "ayuv", "ya")):
        return True
    if any(x in p for x in ("rgba", "bgra", "argb", "abgr", "gbrap")):
        return True
    return False


def _container_codec_supports_alpha(container: str, codec_key: str) -> bool:
    c = (container or "").lower()
    k = (vec.normalize_codec_key(codec_key) or codec_key or "").lower()

    # Lossless/Intermediate Codecs mit RGBA
    if k in ("png", "ffv1", "huffyuv", "utvideo", "magicyuv", "rawvideo", "qtrle"):
        # Tragen Alpha in gängigen Containern
        return c in ("mkv", "mov", "avi")

    # Web-Codec Alpha
    if k in ("vp9", "av1"):
        return c in ("webm", "mkv")

    # ProRes 4444 Alpha: zuverlässig in MOV
    if k in ("prores", "prores_ks"):
        return c in ("mov",)

    return False


def _choose_best_pix_fmt(
    container: str,
    codec_key: str,
    src_pix_fmts: Sequence[Optional[str]],
    want_alpha: bool,
) -> str:
    c = (container or "").lower()
    k = (vec.normalize_codec_key(codec_key) or codec_key or "").lower()

    any_422_or_more = any(
        (
            p
            and any(
                x in p.lower() for x in ("422", "444", "gbrp", "rgb", "rgba", "gbrap")
            )
        )
        for p in src_pix_fmts
    )

    # explizit für Alpha-Fälle zuerst
    if want_alpha and _container_codec_supports_alpha(c, k):
        if k in ("png", "ffv1", "huffyuv", "utvideo", "magicyuv", "rawvideo", "qtrle"):
            return "rgba"  # 8-bit RGBA, breit unterstützt
        if k in ("vp9", "av1"):
            return "yuva420p"  # Web-Alpha
        if k in ("prores", "prores_ks"):
            return "yuva444p10le"  # ProRes 4444
        return "rgba"

    # Non-Alpha Defaults pro Codec
    if k in ("png",):
        return "rgb24"
    if k in ("h264", "hevc", "vp9", "av1"):
        return "yuv420p"
    if k in ("prores", "prores_ks"):
        return "yuv422p10le"
    if k.startswith("dnx"):
        return "yuv422p"
    if k == "utvideo":
        if c in ("avi", "mkv"):
            return "yuv422p" if any_422_or_more else "yuv420p"
        return "yuv420p"
    return "yuv420p"


# ──────────────────────────────────────────────────────────────────────────────
# UI / Auswahl Container & Codec (mit Beschreibungen)
# ──────────────────────────────────────────────────────────────────────────────


def _first_video(files: Sequence[str]) -> Optional[Path]:
    for f in files:
        if _ext_matches(f, VIDEO_EXTS):
            return Path(f)
    return None


def _detect_container_from_path(p: Path) -> str:
    try:
        c = vec.detect_container_from_path(p) or (p.suffix.lstrip(".").lower() or "mp4")
        return "mp4" if c == "m4v" else c
    except Exception:
        c = p.suffix.lstrip(".").lower() or "mp4"
        return "mp4" if c == "m4v" else c


def _normalize_codec_key(s: Optional[str]) -> Optional[str]:
    try:
        return vec.normalize_codec_key(s)
    except Exception:
        return (s or "").lower() if s else None


def _choose_container_and_codec(
    files: Sequence[str], a: MergeArgs, batch_mode: bool
) -> Tuple[str, str, Optional[str]]:
    """
    Wählt Ziel-Container + -Codec.
    Interaktiv (batch_mode=False) jetzt IMMER über ui.pick_format/ui.pick_codec.
    Batch-Logik unverändert (a.format/a.codec inkl. 'keep' werden respektiert).
    """

    def _norm_container_key(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        x = s.strip().lower()
        if x in ("m4v",):
            return "mp4"
        if x in ("matroska",):
            return "mkv"
        return x

    # 1) Quelle / Defaults
    first_vid = _first_video(files) or Path(files[0])
    src_container = _detect_container_from_path(first_vid)  # mp4/mkv/...
    src_codec = vec.probe_video_codec(first_vid) or "h264"
    src_codec_key = _normalize_codec_key(src_codec) or "h264"

    # =========================
    # A) BATCH-MODUS (wie zuvor)
    # =========================
    if batch_mode:
        # Container bestimmen
        req = _norm_container_key(a.format)
        if req in (None, "", "keep", "same", "source", "src"):
            target_container = src_container
        else:
            target_container = req

        # Output-Hint (Dateiendung) darf Container überstimmen
        if isinstance(a.output, str) and a.output.strip():
            hinted = _container_from_output_hint(a.output, target_container)
            if hinted:
                target_container = hinted

        # Mögliche Encoder/Codecs pro Container
        enc_maps = vec.prepare_encoder_maps(
            list(files),
            target_container,
            defin.CONVERT_FORMAT_DESCRIPTIONS,
            prefer_hw=True,
            ffmpeg_bin="ffmpeg",
        )
        enc_map: Dict[str, Optional[str]] = enc_maps.get(target_container, {}) or {}

        # keep möglich?
        try:
            keep_allowed = bool(
                vec.container_allows_codec(target_container, src_codec_key)
            )
        except Exception:
            keep_allowed = False

        # Codec wählen (Batch)
        req_codec = _normalize_codec_key(a.codec) if a.codec else None
        if req_codec in ("keep", "copy"):
            chosen_codec_key = (
                src_codec_key
                if keep_allowed
                else (
                    list(enc_map.keys())[0]
                    if enc_map
                    else vec.suggest_codec_for_container(target_container)
                )
            )
            preferred_encoder = enc_map.get(chosen_codec_key) or vec.encoder_for_codec(
                chosen_codec_key
            )
        elif isinstance(req_codec, str) and req_codec:
            key_res, enc_res = vec.resolve_codec_key_with_container_fallback(
                target_container,
                req_codec,
                ffmpeg_bin="ffmpeg",
                allow_cross_codec_fallback=True,
            )
            chosen_codec_key = key_res or req_codec
            preferred_encoder = (
                enc_res
                or enc_map.get(chosen_codec_key)
                or vec.encoder_for_codec(chosen_codec_key)
            )
        else:
            chosen_codec_key = (
                src_codec_key
                if keep_allowed
                else (
                    list(enc_map.keys())[0]
                    if enc_map
                    else vec.suggest_codec_for_container(target_container)
                )
            )
            preferred_encoder = enc_map.get(chosen_codec_key) or vec.encoder_for_codec(
                chosen_codec_key
            )

        # Safety-Fallback, falls kein Encoder ermittelt wurde
        if not preferred_encoder:
            _t, enc_res = vec.resolve_codec_key_with_container_fallback(
                target_container,
                chosen_codec_key,
                ffmpeg_bin="ffmpeg",
                allow_cross_codec_fallback=True,
            )
            preferred_encoder = enc_res or vec.encoder_for_codec(chosen_codec_key)

        return target_container, chosen_codec_key, preferred_encoder

    # ==================================
    # B) INTERAKTIV — IMMER pick_* nutzen
    # ==================================
    # 1) Container immer fragen
    default_fmt = (
        _norm_container_key(getattr(a, "format", None)) or src_container or "mp4"
    )
    sel_fmt = ui.pick_format(default=default_fmt, back_button=False)

    chosen_container_key = (sel_fmt or default_fmt or "mp4").lower()
    if chosen_container_key in ("keep", "same", "src", "source"):
        target_container = src_container
    else:
        target_container = chosen_container_key

    # 2) Output-Hint (Dateiendung in a.output) kann den Container überschreiben,
    #    falls schon gesetzt (interaktiv meist nicht der Fall, aber der Vollständigkeit halber)
    if isinstance(a.output, str) and a.output.strip():
        hinted = _container_from_output_hint(a.output, target_container)
        if hinted:
            target_container = hinted

    # 3) Encoder-Map für den (ggf. geänderten) Container
    enc_maps = vec.prepare_encoder_maps(
        list(files),
        target_container,
        defin.CONVERT_FORMAT_DESCRIPTIONS,
        prefer_hw=True,
        ffmpeg_bin="ffmpeg",
    )
    enc_map: Dict[str, Optional[str]] = enc_maps.get(target_container, {}) or {}

    # 4) Codec immer fragen (über pick_codec); sinnvoller Default:
    #    - bevorzugt Quellcodec, wenn Container ihn erlaubt
    #    - sonst erster zulässiger Codec der Map
    try:
        keep_allowed = bool(vec.container_allows_codec(target_container, src_codec_key))
    except Exception:
        keep_allowed = False
    default_codec_key = (
        src_codec_key
        if keep_allowed
        else (
            list(enc_map.keys())[0]
            if enc_map
            else vec.suggest_codec_for_container(target_container)
        )
    )

    req_default = _normalize_codec_key(getattr(a, "codec", None)) or default_codec_key
    sel_codec = ui.pick_codec(
        container=target_container, default=req_default, back_button=False
    )

    if isinstance(sel_codec, str) and sel_codec.strip():
        chosen_codec_key = sel_codec.strip().lower()
    else:
        chosen_codec_key = req_default

    # 5) Encoder robust ermitteln
    preferred_encoder = enc_map.get(chosen_codec_key) or None
    if not preferred_encoder:
        _t, enc_res = vec.resolve_codec_key_with_container_fallback(
            target_container,
            chosen_codec_key,
            ffmpeg_bin="ffmpeg",
            allow_cross_codec_fallback=True,
        )
        preferred_encoder = enc_res or vec.encoder_for_codec(chosen_codec_key)

    return target_container, chosen_codec_key, preferred_encoder


# ──────────────────────────────────────────────────────────────────────────────
# Ausgabe / Zusammenfassung
# ──────────────────────────────────────────────────────────────────────────────


def _print_summary(
    video_files: Sequence[str],
    out_path: Path,
    tgt_label: str,
    fps: int,
    pause_sec: float,
    offset_sec: float,
    audio_offset_sec: float,
    subtitle_offset_sec: float,
    container: str,
    vcodec_key: str,
    pix_fmt: str,
    alpha_on: bool,
    *,
    preset_name: Optional[str] = None,
    extend_opt: Optional[bool] = None,
    audio_files: Optional[Sequence[str]] = None,
    subtitle_files: Optional[Sequence[str]] = None,
) -> None:
    print()
    co.print_headline("  " + _("selected_parameters") + "   ", "bright_blue")
    co.print_seperator()

    if preset_name is not None:
        preset_obj = (
            defin.CONVERT_PRESET.get(preset_name, {})
            if hasattr(defin, "CONVERT_PRESET")
            else {}
        ) or {}
        preset_label = str(preset_obj.get("name", preset_name))
        co.print_value_info(_("preset"), preset_label)

    container_obj = (
        defin.CONVERT_FORMAT_DESCRIPTIONS.get(container, {})
        if hasattr(defin, "CONVERT_FORMAT_DESCRIPTIONS")
        else {}
    ) or {}
    container_label = str(container_obj.get("name", container))
    co.print_value_info(_("container"), container_label or container)
    codec_obj = (
        defin.VIDEO_CODECS.get(vcodec_key, {}) if hasattr(defin, "VIDEO_CODECS") else {}
    ) or {}
    codec_label = str(codec_obj.get("name", vcodec_key))
    co.print_value_info(_("video_codec"), codec_label or vcodec_key)
    co.print_value_info(_("target_resolution"), tgt_label)
    co.print_value_info("FPS", str(fps))
    co.print_value_info("PixFmt", pix_fmt)
    co.print_value_info("Alpha", he.yesno(alpha_on))
    if extend_opt is not None:
        co.print_value_info(
            tr({"de": "Extend (verlängern)", "en": "Extend"}), he.yesno(extend_opt)
        )
    if abs(offset_sec) > 0:
        co.print_value_info(_("offset"), he.format_time(offset_sec))
    if abs(audio_offset_sec) > 0:
        co.print_value_info(_("audio_offset"), he.format_time(audio_offset_sec))
    if abs(subtitle_offset_sec) > 0:
        co.print_value_info(_("subtitle_offset"), he.format_time(subtitle_offset_sec))
    if pause_sec > 0:
        co.print_value_info(_("pause"), he.format_time(pause_sec))

    # Kategorien – nur ausgeben, wenn vorhanden
    if video_files:
        print("\n  " + _("video_files") + ":")
        for f in video_files:
            co.print_line(f"   - {Path(f).name}", color="soft_blue")

    if audio_files:
        print("\n  " + tr({"de": "Audiodateien", "en": "Audio files"}) + ":")
        for f in audio_files:
            co.print_line(f"   - {Path(f).name}", color="soft_blue")

    if subtitle_files:
        print("\n  " + tr({"de": "Untertitel", "en": "Subtitles"}) + ":")
        for f in subtitle_files:
            co.print_line(f"   - {Path(f).name}", color="soft_blue")

    co.print_line(f"\n  {_('output_name')}:")
    co.print_line(f"    {str(out_path)}", color="soft_blue")
    print(" ")
    co.print_seperator()


def _default_out_name(
    files: Sequence[str], container: str, purpose: str = "merged"
) -> str:
    vids = [Path(f) for f in files if Path(f).suffix.lower() in VIDEO_EXTS]
    base = vids[0] if vids else Path(files[0])
    return str(base.with_stem(base.stem + f"_{purpose}").with_suffix(f".{container}"))


# ──────────────────────────────────────────────────────────────────────────────
# Klassifizierung
# ──────────────────────────────────────────────────────────────────────────────


def _ext_matches(p: str | Path, exts: Sequence[str]) -> bool:
    suf = Path(p).suffix.lower()
    suf_nodot = suf[1:] if suf.startswith(".") else suf
    return (suf in exts) or (suf_nodot in exts)


def _classify_merge(files: Sequence[str]) -> str:
    vids = [f for f in files if _ext_matches(f, VIDEO_EXTS)]
    auds = [f for f in files if _ext_matches(f, AUDIO_EXTS)]
    subs = [f for f in files if _ext_matches(f, SUB_EXTS)]

    if len(vids) >= 2:
        return "video_concat"

    if len(vids) == 1:
        if len(auds) >= 1 or len(subs) >= 1:
            return "video_plus_assets"  # NEU: ein Video + (Audios/Subs beliebig)
    return "video_concat"


# ──────────────────────────────────────────────────────────────────────────────
# Filter-Baustein
# ──────────────────────────────────────────────────────────────────────────────


def _video_chain_expr(
    strategy: str,
    fit_mode: str,
    W: int,
    H: int,
    fps: int,
    pix_fmt: str,
    pad_alpha: bool,
) -> str:
    """Gibt die Filter-Teil-Expression (ohne Labels) für einen Videozweig zurück.
    WICHTIG: Pixelformat jetzt bewusst AM ANFANG, damit color/pad mit Alpha arbeiten können.
    """
    pad_col = "black@0" if pad_alpha else "black"
    parts: List[str] = []

    # 0) Ziel-Pixelformat zuerst, damit nachfolgende Filter (pad, color, scale) einen Alphakanal haben
    parts.append(f"format={pix_fmt}")

    # 1) Größen-/AR-Anpassung
    if strategy == "no-scale":
        parts.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={pad_col}")
    elif strategy == "match-first":
        if fit_mode == "pad":
            parts.append(
                f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease"
            )
            parts.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={pad_col}")
        else:
            parts.append(
                f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=increase"
            )
            parts.append(f"crop={W}:{H}")
    elif strategy in ("smallest", "average", "largest", "fixed"):
        # Einheitlich auf Zielgröße skalieren (keine AR-Bewahrung → harte Vereinheitlichung)
        parts.append(f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=disable")

    # 2) Vereinheitlichung
    parts.append(f"fps=fps={fps}")
    parts.append("setsar=1")

    return ",".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Filtergraph-Builder (ein Durchlauf, robust) – Video+Video
# ──────────────────────────────────────────────────────────────────────────────


def _build_concat_filtergraph(
    *,
    inputs: List[Path],
    dims: Tuple[int, int],
    fps: int,
    pix_fmt: str,
    strategy: str,
    fit_mode: str,
    offset_sec: float,
    pause_sec: float,
    pad_alpha: bool,
    # NEW:
    tail_pad_sec: float = 0.0,
) -> Tuple[List[str], str]:
    W, H = dims
    cmd_inputs: List[str] = []

    file_meta: List[Tuple[bool, float]] = []
    for p in inputs:
        has_a = he.probe_has_audio(str(p))
        dur = he.probe_duration(str(p))
        file_meta.append((has_a, dur))
        cmd_inputs += ["-i", str(p)]

    seg_v_labels: List[str] = []
    seg_a_labels: List[str] = []
    chains: List[str] = []

    vexpr = _video_chain_expr(strategy, fit_mode, W, H, fps, pix_fmt, pad_alpha)

    # Startoffset
    seg_idx = 0
    if offset_sec > 0.0001:
        vlab = f"[vs{seg_idx}]"
        alab = f"[as{seg_idx}]"
        seg_v_labels.append(vlab)
        seg_a_labels.append(alab)
        vsrc = f"color=c={'black@0' if pad_alpha else 'black'}:s={W}x{H}:d={max(0.0, offset_sec)},{vexpr}"
        asrc = f"anullsrc=r=48000:cl=stereo:d={max(0.0, offset_sec)}"
        chains.append(f"{vsrc}{vlab}")
        chains.append(f"{asrc}{alab}")
        seg_idx += 1

    # Segmente
    for i, (path, (has_a, dur)) in enumerate(zip(inputs, file_meta)):
        in_v = f"[{i}:v:0]"
        out_v = f"[vs{seg_idx}]"
        seg_v_labels.append(out_v)
        chains.append(f"{in_v}{vexpr}{out_v}")

        out_a = f"[as{seg_idx}]"
        seg_a_labels.append(out_a)
        if has_a:
            a_chain = f"[{i}:a:0]aformat=sample_fmts=s16:channel_layouts=stereo,aresample=48000"
            if dur > 0.1:
                a_chain += f",atrim=0:{dur}"
            a_chain += f",asetpts=PTS-STARTPTS{out_a}"
            chains.append(a_chain)
        else:
            dval = max(0.0, dur) if dur > 0.0 else 36000
            chains.append(f"anullsrc=r=48000:cl=stereo:d={dval}{out_a}")

        seg_idx += 1

        if pause_sec > 0.0001 and i < (len(inputs) - 1):
            vlab = f"[vs{seg_idx}]"
            alab = f"[as{seg_idx}]"
            seg_v_labels.append(vlab)
            seg_a_labels.append(alab)
            vsrc = f"color=c={'black@0' if pad_alpha else 'black'}:s={W}x{H}:d={max(0.0, pause_sec)},{vexpr}"
            asrc = f"anullsrc=r=48000:cl=stereo:d={max(0.0, pause_sec)}"
            chains.append(f"{vsrc}{vlab}")
            chains.append(f"{asrc}{alab}")
            seg_idx += 1

    # NEW: schwarzer Tail, falls benötigt
    if tail_pad_sec > 0.0001:
        vlab = f"[vs{seg_idx}]"
        alab = f"[as{seg_idx}]"
        seg_v_labels.append(vlab)
        seg_a_labels.append(alab)
        vsrc = f"color=c={'black@0' if pad_alpha else 'black'}:s={W}x{H}:r={fps}:d={max(0.0, tail_pad_sec)},{vexpr}"
        asrc = f"anullsrc=r=48000:cl=stereo:d={max(0.0, tail_pad_sec)}"
        chains.append(f"{vsrc}{vlab}")
        chains.append(f"{asrc}{alab}")
        seg_idx += 1

    assert len(seg_v_labels) == len(
        seg_a_labels
    ), f"concat label mismatch: {len(seg_v_labels)} video vs {len(seg_a_labels)} audio segments"
    n = len(seg_v_labels)
    pair_order = "".join(f"{seg_v_labels[i]}{seg_a_labels[i]}" for i in range(n))
    chains.append(f"{pair_order}concat=n={n}:v=1:a=1[v][a]")

    filter_complex = ";".join(chains)
    return cmd_inputs, filter_complex


def _force_encoded_audio_args_for_container(container: str) -> List[str]:
    c = (container or "").lower()
    # konservative, weit kompatible Defaults:
    if c in ("mp4", "m4v", "mov"):
        return ["-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    if c == "webm":
        return ["-c:a", "libopus", "-b:a", "160k", "-ac", "2"]
    if c == "avi":
        return ["-c:a", "ac3", "-b:a", "384k", "-ac", "2"]
    if c == "mpeg":
        return ["-c:a", "mp2", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    # MKV/MOV vertragen praktisch alles – AAC ist ein solider Default
    return ["-c:a", "aac", "-b:a", "192k"]


def _avoid_copy_for_filtered_audio(audio_args: List[str], container: str) -> List[str]:
    """Wenn der Audio-Output aus dem Filtergraph kommt, darf -c:a copy NICHT verwendet werden."""
    args = audio_args[:]
    try:
        i = args.index("-c:a")
        val = (args[i + 1] if i + 1 < len(args) else "") or ""
        if val.lower() == "copy":
            # komplette Audio-Args für diesen Fall neu setzen (einfach & robust)
            return _force_encoded_audio_args_for_container(container)
    except ValueError:
        pass
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Hilfs-UI: Strategie wählen & Ziel-Parameter berechnen
# ──────────────────────────────────────────────────────────────────────────────


def _parse_wh_string(s: str) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    txt = s.strip().lower()
    txt = re.sub(r"[^\dx:]", ":", txt.replace("×", "x"))
    txt = txt.replace("x", ":").replace("/", ":").replace(",", ":").replace(" ", ":")
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", txt)
    if not m:
        return None
    w = int(m.group(1) or 0)
    h = int(m.group(2) or 0)
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def _pick_fixed_resolution_interactive() -> Tuple[int, int, str]:
    res_dict = getattr(defin, "RESOLUTIONS", getattr(defin, "Resolutions", {}))
    if not isinstance(res_dict, dict) or not res_dict:
        return (1920, 1080, "1920x1080")

    keys = [k for k in res_dict.keys() if str(k).lower() != "original"]
    labels = [
        str(tr((res_dict[k].get("name") if isinstance(res_dict[k], dict) else k)))
        for k in keys
    ]

    sel = ui.ask_user(
        prompt=_("choose_resolution"),
        options=keys,
        display_labels=labels,
        default=0,
        explanation=None,
        back_button=False,
    )

    if isinstance(sel, int) and 0 <= sel < len(keys):
        chosen_key = keys[sel]
    elif isinstance(sel, str) and sel in keys:
        chosen_key = sel
    else:
        chosen_key = keys[0]

    spec = res_dict.get(chosen_key, {})
    scale_str = (spec or {}).get("scale") if isinstance(spec, dict) else None

    if str(chosen_key).lower() == "custom" or not scale_str:
        while True:
            custom_in = ui.ask_two_ints(
                _("enter_custom_resolution"),
                output_sep="x",
                default="1920:1080",
                explanation=_("enter_custom_resolution_explanation"),
            )
            ok, msg = vec.validate_custom_scale_leq(
                None, None, custom_in, require_even=True
            )
            if ok:
                wh = custom_in.split("x")
                return int(wh[0]), int(wh[1]), custom_in
            co.print_fail(msg)

    wh = _parse_wh_string(str(scale_str)) or (1920, 1080)
    label = str(tr((spec.get("name") if isinstance(spec, dict) else chosen_key)))
    return (wh[0], wh[1], label)


def _pick_strategy_and_dims(
    a: MergeArgs, batch_mode: bool, videos: List[str]
) -> Tuple[str, str, Tuple[int, int], int, bool, str]:
    """Return (strategy, fit_mode, (W,H), fps, all_have_alpha, tgt_label)."""
    strategy_keys = [
        "match-first",
        "no-scale",
        "smallest",
        "average",
        "largest",
        "fixed",
    ]
    strategy_labels = [
        _("match_first_resolution"),
        _("no_scale_pad_to_max"),
        _("smallest"),
        _("average"),
        _("largest"),
        tr({"de": "Auf feste Auflösung skalieren", "en": "Scale to fixed resolution"}),
    ]

    strategy = a.target_res
    if not batch_mode:
        sel_res = ui.ask_user(
            prompt=_("choose_target_resolution_strategy"),
            options=strategy_keys,
            display_labels=strategy_labels,
            default=0,
            explanation=None,
            back_button=False,
        )
        strategy = (
            sel_res
            if isinstance(sel_res, str)
            else strategy_keys[int(sel_res) if isinstance(sel_res, int) else 0]
        )

    chosen_fixed: Optional[Tuple[int, int, str]] = None
    if isinstance(strategy, str):
        if strategy == "first":
            strategy = "match-first"

        # Falls direkt "WxH" übergeben wurde → als fixed behandeln
        if isinstance(strategy, str) and re.match(
            r"^\s*\d+\s*(?:x|×|:)\s*\d+\s*$", strategy, flags=re.I
        ):
            wh = _parse_wh_string(strategy)
            if wh:
                chosen_fixed = (wh[0], wh[1], f"{wh[0]}x{wh[1]}")
                strategy = "fixed"
        if strategy.startswith("fixed:"):
            wh = _parse_wh_string(strategy.split(":", 1)[1])
            if wh:
                chosen_fixed = (wh[0], wh[1], f"{wh[0]}x{wh[1]}")
                strategy = "fixed"

    if strategy not in strategy_keys:
        strategy = "match-first"
    a.target_res = strategy

    fit_mode = "pad"
    if (strategy == "match-first") and (not batch_mode):
        sel_fit = ui.ask_user(
            prompt=_("fit_mode_prompt"),
            options=["pad", "crop"],
            display_labels=[_("pad_to_target"), _("crop_to_target")],
            default=0,
            explanation=None,
            back_button=False,
        )
        fit_mode = (
            sel_fit
            if isinstance(sel_fit, str)
            else ["pad", "crop"][int(sel_fit) if isinstance(sel_fit, int) else 0]
        )

    if strategy == "match-first":
        W, H = he.probe_wh(videos[0])
    elif strategy == "fixed":
        if chosen_fixed is None:
            if batch_mode:
                chosen_fixed = (1920, 1080, "1920x1080")
            else:
                chosen_fixed = _pick_fixed_resolution_interactive()
        W, H = chosen_fixed[0], chosen_fixed[1]
    else:
        W, H = _choose_target_size(videos, strategy or "no-scale")

    target_fps = _detect_target_fps_min(videos)

    src_pfs = [he.probe_wh_fmt_fps(v)[2] for v in videos]
    any_have_alpha = any(pf and _pix_fmt_has_alpha(pf) for pf in src_pfs)

    if strategy == "fixed":
        tgt_label = f"fixed → {W}x{H}" + (
            f" ({chosen_fixed[2]})" if chosen_fixed else ""
        )
    else:
        tgt_label = f"{strategy} → {W}x{H}" + (
            f" ({fit_mode})" if strategy == "match-first" else ""
        )

    return strategy, fit_mode, (W, H), target_fps, any_have_alpha, tgt_label


def _infer_default_subtitle_title(sub_path: Path) -> str:
    stem = sub_path.stem

    # 1) Sprache aus Name (nutzt die umfassendere Heuristik)
    lang3 = he.guess_lang_from_name(stem)
    if lang3:
        title = defin.LANG_DISPLAY.get(lang3, lang3.upper())
        return he.sanitize_title(title)

    # 2) VTT-Header "Language:" beachten (falls vorhanden)
    if sub_path.suffix.lower() == ".vtt":
        try:
            with sub_path.open("r", encoding="utf-8", errors="ignore") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    m = re.search(r"^\s*Language\s*:\s*([A-Za-z-]+)\s*$", line, re.I)
                    if m:
                        key = m.group(1).lower()
                        # Versuche auf ISO639-2 zu mappen (wie beim Audio-Pendant)
                        lang3 = defin.LANG_ISO3.get(key)
                        if lang3:
                            title = defin.LANG_DISPLAY.get(lang3, lang3.upper())
                        else:
                            # Fallback: bestehende Map weiterverwenden, wenn vorhanden
                            title = (
                                defin.LANG_ISO3.get(key, key.capitalize())
                                if "LANG_ISO3" in globals()
                                else key.capitalize()
                            )
                        return he.sanitize_title(title)
        except Exception:
            pass

    # 3) Fallback: Dateiname (ohne Endung)
    return he.sanitize_title(stem)


def _infer_default_audio_title_and_lang(audio_path: Path) -> Tuple[str, Optional[str]]:
    """Gibt (Titel, ISO639-2 Sprache) zurück. Sprache kann None sein."""
    stem = audio_path.stem
    # 1) Sprache aus Name
    lang3 = he.guess_lang_from_name(stem)

    # 2) Tags aus Datei
    tags = he.ffprobe_tags(str(audio_path))
    tagblk = (
        (tags.get("streams") or [{}])[0].get("tags", {})
        or tags.get("format", {}).get("tags", {})
        or {}
    )

    # mögliche Keys: language / LANG / TITLE / title / handler_name
    t_title = tagblk.get("title") or tagblk.get("TITLE") or tagblk.get("handler_name")
    t_lang = tagblk.get("language") or tagblk.get("LANG") or tagblk.get("Language")
    if t_lang and not lang3:
        lang3 = defin.LANG_ISO3.get(t_lang.lower(), None)

    # 3) Titel bestimmen
    if t_title:
        title = t_title
    elif lang3:
        title = defin.LANG_DISPLAY.get(lang3, lang3.upper())
    else:
        title = stem

    return he.sanitize_title(title), lang3


def _infer_default_subtitle_title_and_lang(sub_path: Path) -> Tuple[str, Optional[str]]:
    """
    Liefert (Titel, ISO639-2 code) für Untertiteldateien.
    - nutze he.guess_lang_from_name(stem) für robuste Sprachdetektion
    - fallback: vorhandene Logik aus _infer_default_subtitle_title()
    """
    stem = sub_path.stem
    # 1) Versuche robuste Spracherkennung (liefert z. B. 'deu', 'eng', ...)
    lang3 = None
    try:
        lang3 = he.guess_lang_from_name(stem)
    except Exception:
        lang3 = None

    # 2) Titel aus Sprachcode ableiten (Deutsch/English/…)
    if lang3:
        title = defin.LANG_DISPLAY.get(lang3, lang3.upper())
        return he.sanitize_title(title), lang3

    # 3) Fallback: vorhandene simple Logik (sucht 'de', 'eng', 'german', ...)
    title = _infer_default_subtitle_title(sub_path)
    low = title.lower()
    # Wenn der Titel selbst eine Sprachbezeichnung ist, mappe nach ISO3
    if low in defin.LANG_ISO3:
        lang3 = defin.LANG_ISO3[low]
        title = defin.LANG_DISPLAY.get(lang3, title)

    return he.sanitize_title(title), lang3


def _per_streamize_audio_args(args: List[str], idx: int) -> List[str]:
    """
    Wandelt generische Audio-Args (z.B. ['-c:a','libopus','-b:a','160k','-ac','2'])
    in per-Stream-Varianten für die NEUE Spur mit Index idx um:
    ['-c:a:idx','libopus','-b:a:idx','160k','-ac:a:idx','2', ...]
    """
    out: List[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None

        # Zwei-Token-Optionen, die Stream-Specifier unterstützen
        if tok in ("-c:a", "-b:a", "-ar", "-ac", "-filter:a", "-af", "-mapping_family"):
            if nxt is not None:
                out += [f"{tok}:{idx}", str(nxt)]
                i += 2
                continue

        # Einfache Flags einfach übernehmen
        out.append(tok)
        i += 1

    return out


def _strip_global_filters_if_complex(cmd: List[str]) -> List[str]:
    """Entfernt -vf/-filter:v (und optional -af/-filter:a), wenn -filter_complex benutzt wird."""
    if "-filter_complex" not in cmd:
        return cmd

    def _rm(opt: str) -> None:
        while True:
            try:
                i = cmd.index(opt)
                # entferne Option + Wert sofern vorhanden
                del cmd[i : i + 2]
            except ValueError:
                break

    for opt in ("-vf", "-filter:v", "-af", "-filter:a"):
        _rm(opt)
    return cmd


def _enforce_single_pix_fmt(cmd: List[str], desired_pix_fmt: str) -> List[str]:
    """Entfernt alle vorhandenen -pix_fmt und hängt exakt einen mit gewünschtem Format an."""
    out: List[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-pix_fmt":
            i += 2  # Option + Wert überspringen
            continue
        out.append(cmd[i])
        i += 1
    out += ["-pix_fmt", desired_pix_fmt]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Merge-Zweige
# ──────────────────────────────────────────────────────────────────────────────


def _merge_video_concat(
    a: MergeArgs,
    batch_mode: bool,
    video_files: List[str],
    extra_audios: Optional[List[str]] = None,
    extra_subs: Optional[List[str]] = None,
    preserved_cover: Optional[Path] = None,
    *,
    precollected_offsets: bool = False,  # <-- NEU: verhindert doppelte Nachfragen
) -> None:
    extra_audios = [str(x) for x in (extra_audios or [])]
    extra_subs = [str(x) for x in (extra_subs or [])]

    target_container, chosen_codec_key, preferred_encoder = _choose_container_and_codec(
        video_files, a, batch_mode
    )

    preset_name = a.preset
    if not batch_mode:
        if not preset_name:
            # preset_name = _pick_quality_preset_interactive(default_key="casual")
            preset_name = ui.pick_preset("casual", back_button=False)
    else:
        preset_name = preset_name or "casual"

    # Eingangs-Offsets aus Args
    off_sec = max(0.0, he.flex_time_seconds(a.offset))
    pause_sec = max(0.0, he.flex_time_seconds(a.pause))

    off_a_sec = he.flex_time_seconds(getattr(a, "audio_offset", "0"))
    off_s_sec = he.flex_time_seconds(getattr(a, "subtitle_offset", "0"))
    try:
        outro_sec = max(0.0, he.flex_time_seconds(getattr(a, "outro", "1s")))
    except Exception:
        outro_sec = 1.0

    # Interaktive Abfragen NUR, wenn nicht bereits vorgelagert erhoben
    if (not batch_mode) and (not precollected_offsets):
        ret = ui.ask_time_or_other_number(
            _("offset_before_all_clips_optional"), default="0s", exit_on_0=False
        )
        if ret is not None:
            _is_time, val = ret
            off_sec = max(0.0, he.flex_time_seconds(str(val)))

        ret = ui.ask_time_or_other_number(
            _("pause_between_clips"), default="0s", exit_on_0=False
        )
        if ret is not None:
            _is_time, val = ret
            pause_sec = max(0.0, he.flex_time_seconds(str(val)))

        if extra_audios:
            ret = ui.ask_time_or_other_number(
                _("enter_offset_for_audio"), default="0s", exit_on_0=False
            )
            if ret is not None:
                _is_time, val = ret
                off_a_sec = he.flex_time_seconds(str(val))

        if extra_subs:
            ret = ui.ask_time_or_other_number(
                _("enter_offset_for_subtitle"), default="0s", exit_on_0=False
            )
            if ret is not None:
                _is_time, val = ret
                off_s_sec = he.flex_time_seconds(str(val))

    # >>> GLOBALER Offset wirkt auch auf externe Audios/Subs
    eff_off_a = max(0.0, off_sec) + off_a_sec
    eff_off_s = max(0.0, off_sec) + off_s_sec

    if not batch_mode:
        default_name = Path(_default_out_name(video_files, target_container, "merged"))
        out_path = Path(ui.ask_for_filename(default_name, default=default_name.name))
        print()
    else:
        out_path = (
            Path(a.output)
            if a.output
            else Path(_default_out_name(video_files, target_container, "merged"))
        )
        if out_path.suffix == "":
            out_path = out_path.with_suffix(f".{target_container}")

    strategy, fit_mode, (W, H), target_fps, any_have_alpha, tgt_label = (
        _pick_strategy_and_dims(a, batch_mode, video_files)
    )
    alpha_supported = _container_codec_supports_alpha(
        target_container, chosen_codec_key
    )
    want_alpha = any_have_alpha and alpha_supported

    if any_have_alpha and not alpha_supported:
        _warn_alpha_drop(target_container, chosen_codec_key)

    target_pix_fmt = _choose_best_pix_fmt(
        target_container,
        chosen_codec_key,
        [he.probe_wh_fmt_fps(v)[2] for v in video_files],
        want_alpha,
    )

    # Dauer & Extend mit effektiven Offsets
    try:
        concat_vid_dur = (
            sum(he.probe_duration(v) for v in video_files)
            + max(0.0, off_sec)
            + max(0.0, pause_sec) * max(0, len(video_files) - 1)
        )
    except Exception:
        concat_vid_dur = 0.0

    extra_a_durs = [he.probe_duration(x) for x in extra_audios] if extra_audios else []
    extra_s_durs = (
        [he.probe_subtitle_duration(x) for x in extra_subs] if extra_subs else []
    )
    a_ends = [eff_off_a + d for d in extra_a_durs]
    s_ends = [eff_off_s + d for d in extra_s_durs]
    longest_extra = max(a_ends + s_ends, default=0.0)

    if not batch_mode and (longest_extra > concat_vid_dur + 0.05):
        msg = tr(
            {
                "de": f"Audio/Untertitel sind länger ({he.format_time(longest_extra)}) als das Video ({he.format_time(concat_vid_dur)}). Video bis zur längsten Spur verlängern?",
                "en": f"Audio/subtitles are longer ({he.format_time(longest_extra)}) than the video ({he.format_time(concat_vid_dur)}). Extend video to the longest track?",
            }
        )
        res = ui.ask_yes_no(msg, default=False, back_option=False)
        a.extend = bool(res) if res is not None else False

    do_extend = bool(a.extend)
    tail_pad_extend = max(0.0, longest_extra - concat_vid_dur) if do_extend else 0.0
    total_tail = tail_pad_extend + max(0.0, outro_sec)

    # Summary (anzeige bleibt wie bisher; audio/sub offsets werden als "zusätzliche" gezeigt)
    _print_summary(
        video_files=video_files,
        out_path=out_path,
        tgt_label=tgt_label,
        fps=target_fps,
        pause_sec=pause_sec,
        offset_sec=off_sec,
        audio_offset_sec=off_a_sec,
        subtitle_offset_sec=off_s_sec,
        container=target_container,
        vcodec_key=chosen_codec_key,
        pix_fmt=target_pix_fmt,
        alpha_on=want_alpha,
        preset_name=preset_name,
        extend_opt=do_extend,
        audio_files=extra_audios,
        subtitle_files=extra_subs,
    )

    # Filtergraph (mit Lead-In & optional Tail)
    cmd_inputs, filter_complex = _build_concat_filtergraph(
        inputs=[Path(v) for v in video_files],
        dims=(W, H),
        fps=target_fps,
        pix_fmt=target_pix_fmt,
        strategy=strategy,
        fit_mode=fit_mode,
        offset_sec=off_sec,
        pause_sec=pause_sec,
        pad_alpha=_pix_fmt_has_alpha(target_pix_fmt),
        tail_pad_sec=total_tail,
    )

    # --- Extra-Audio-Filter: Stille voran (adelay) oder vorne abschneiden (atrim) ---
    extra_a_filters: List[str] = []
    extra_a_labels: List[str] = []
    ms = int(round(max(0.0, eff_off_a) * 1000))
    vid_n = len(video_files)

    for k, ea in enumerate(extra_audios):
        in_a = f"[{vid_n + k}:a:0]"
        out_a = f"[aext{k}]"
        chain = f"{in_a}aformat=sample_fmts=s16:channel_layouts=stereo,aresample=48000"
        if eff_off_a > 0.0001:
            chain += f",adelay={ms}|{ms}:all=1"
        elif eff_off_a < -0.0001:
            chain += f",atrim=start={-eff_off_a:.6f},asetpts=PTS-STARTPTS"
        chain += out_a
        extra_a_filters.append(chain)
        extra_a_labels.append(out_a)

    if extra_a_filters:
        filter_complex = filter_complex + ";" + ";".join(extra_a_filters)

    # Externe Spuren mit *effektivem* Offset (global + individuell)
    for ea in extra_audios:
        cmd_inputs += ["-i", ea]

    for es in extra_subs:
        if abs(eff_off_s) > 0.0001:
            cmd_inputs += ["-itsoffset", f"{eff_off_s}"]
        cmd_inputs += ["-i", es]

    enc = preferred_encoder or vec.encoder_for_codec(
        vec.normalize_codec_key(chosen_codec_key) or chosen_codec_key
    )
    vq = _build_video_quality_args(enc, preset_name)
    if want_alpha and (enc or "").lower() == "libvpx-vp9":
        vq = ["-auto-alt-ref", "0"] + vq

    base_aargs = vec.build_audio_args(
        target_container, preset_name=preset_name, input_path=Path(video_files[0])
    )
    base_aargs = _avoid_copy_for_filtered_audio(base_aargs, target_container)
    a0_args = _per_streamize_audio_args(base_aargs, 0)

    extra_a_stream_args: List[str] = []
    extra_a_stream_args: List[str] = []
    for k, ea in enumerate(extra_audios):
        per = vec.build_audio_args(
            target_container,
            preset_name=preset_name,
            input_path=Path(ea),
        )
        # WICHTIG: Streams aus dem Filtergraph (aformat/aresample/adelay/atrim) dürfen NICHT "copy" sein
        per = _avoid_copy_for_filtered_audio(per, target_container)
        extra_a_stream_args += _per_streamize_audio_args(per, 1 + k)

    sub_codec_args: List[str] = []
    if extra_subs:
        if target_container in ("mp4", "m4v", "mov"):
            sub_codec_args = ["-c:s", "mov_text"]
        else:
            sub_codec_args = ["-c:s", "copy"]

    vid_n = len(video_files)
    aud_n = len(extra_audios)

    cmd: List[str] = (
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-stats_period",
            "0.5",
            "-fflags",
            "+genpts",
            "-max_interleave_delta",
            "0",
        ]
        + cmd_inputs
        + ["-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-c:v", enc]
        + vq
        + ["-pix_fmt", target_pix_fmt]
        + a0_args
    )
    cmd += ["-avoid_negative_ts", "make_zero"]

    for i in range(aud_n):
        cmd += ["-map", extra_a_labels[i]]
    cmd += extra_a_stream_args

    for i in range(len(extra_subs)):
        cmd += ["-map", f"{vid_n + aud_n + i}:s:0?"]
    cmd += sub_codec_args

    # --- Metadaten (Titel/Language) für zusätzliche Audios/Subs setzen ---
    # Audio: a:0 ist die zusammengefügte Spur aus dem Filtergraph -> externe starten bei a:1
    for k, ea in enumerate(extra_audios):
        title, lang3 = _infer_default_audio_title_and_lang(Path(ea))
        ai = 1 + k
        if title:
            cmd += ["-metadata:s:a:%d" % ai, f"title={title}"]
        if lang3:
            cmd += ["-metadata:s:a:%d" % ai, f"language={lang3}"]

    # Subtitles: keine Basis-Subs im Concat-Pfad -> externe starten bei s:0
    for k, es in enumerate(extra_subs):
        title, lang3 = _infer_default_subtitle_title_and_lang(Path(es))
        si = k
        if title:
            cmd += ["-metadata:s:s:%d" % si, f"title={title}"]
        if lang3:
            cmd += ["-metadata:s:s:%d" % si, f"language={lang3}"]

    cmd = vec.apply_container_codec_quirks(cmd[:], target_container, chosen_codec_key)
    # globale -vf/-filter:* entfernen, falls wir -filter_complex benutzen
    cmd = _strip_global_filters_if_complex(cmd)

    # Falls Quirks -pix_fmt überschrieben haben → sicherstellen, dass unser Zielpixelformat gilt
    cmd = _enforce_single_pix_fmt(cmd, target_pix_fmt)
    if concat_vid_dur > 0.0:
        if total_tail > 0.0001:
            cmd += ["-t", f"{(concat_vid_dur + total_tail):.3f}"]
        elif not do_extend:
            cmd += ["-t", f"{concat_vid_dur:.3f}"]

    cmd = _maybe_add_faststart(cmd, target_container, preset_name)
    cmd.append(str(out_path))
    vec.ensure_pre_output_order(cmd)

    video_total = (
        sum(he.probe_duration(v) for v in video_files)
        + max(0.0, off_sec)
        + max(0.0, pause_sec) * max(0, len(video_files) - 1)
        + total_tail
    )
    total_est = max(video_total, longest_extra)

    cmd = autotune_final_cmd(video_files[0], cmd)

    pw.run_ffmpeg_with_progress(
        out_path.name,
        cmd,
        _("concatenating_preprocessed_clips"),
        _("merged_to_output"),
        total_duration=total_est,
        BATCH_MODE=batch_mode,
    )

    _apply_preserved_thumbnail(out_path, preserved_cover, BATCH_MODE=batch_mode)


# --- Container aus Output-Hinweis ableiten (sonst Container des Videos) ---
def _container_from_output_hint(output: Optional[str], fallback_container: str) -> str:
    try:
        if output:
            ext = Path(output).suffix.lstrip(".").lower()
            if ext:
                return "mp4" if ext == "m4v" else ext
    except Exception:
        pass
    return fallback_container


# --- ggf. von Quirks gesetztes -vf/-filter:v entfernen, wenn -c:v copy verwendet wird ---
def _strip_vf_if_streamcopy(cmd: List[str]) -> List[str]:
    def _strip_pair(opt: str) -> None:
        try:
            i = cmd.index(opt)
            if i >= 0:
                if i + 1 < len(cmd):
                    del cmd[i : i + 2]
                else:
                    cmd.pop(i)
        except ValueError:
            pass

    try:
        i = cmd.index("-c:v")
        if i + 1 < len(cmd) and (cmd[i + 1] or "").lower() == "copy":
            _strip_pair("-vf")
            _strip_pair("-filter:v")
    except ValueError:
        pass
    return cmd


def _merge_video_plus_assets(
    a: MergeArgs,
    batch_mode: bool,
    video_path: Path,
    audio_paths: List[str],
    sub_paths: List[str],
    files: Sequence[str],
    preserved_cover: Optional[Path] = None,
) -> None:
    target_container, chosen_codec_key, _preferred_encoder = (
        _choose_container_and_codec([str(video_path)], a, batch_mode=True)
    )

    def _parse_offset_to_seconds(val: Optional[str], ref_dur: float) -> float:
        raw = str(val or "0").strip()
        if raw.endswith(("%", "p", "P")) or raw.lower().endswith("per"):
            return he.time_or_percent_to_seconds(raw, ref_dur)
        return he.flex_time_seconds(raw)

    vdur = he.probe_duration(str(video_path))

    v_off_sec = _parse_offset_to_seconds(getattr(a, "offset", "0"), vdur)
    a_off_sec = _parse_offset_to_seconds(getattr(a, "audio_offset", "0"), vdur)
    s_off_sec = _parse_offset_to_seconds(getattr(a, "subtitle_offset", "0"), vdur)

    # Interaktiv: Video-Offset immer abfragen; Audio/Sub nur wenn vorhanden
    if not batch_mode:
        ret = ui.ask_time_or_other_number(
            _("offset_before_all_clips_optional"), default="0s", exit_on_0=False
        )
        if ret is not None:
            _is_time, val = ret
            v_off_sec = max(
                0.0, he.flex_time_seconds(str(val))
            )  # positiver Lead-In (Schwarz/Stille)
        if audio_paths:
            ret = ui.ask_time_or_other_number(
                _("enter_offset_for_audio"), default="0s", exit_on_0=False
            )
            if ret is not None:
                _is_time, val = ret
                a_off_sec = he.flex_time_seconds(str(val))
        if sub_paths:
            ret = ui.ask_time_or_other_number(
                _("enter_offset_for_subtitle"), default="0s", exit_on_0=False
            )
            if ret is not None:
                _is_time, val = ret
                s_off_sec = he.flex_time_seconds(str(val))

    # Längenvergleich (mit globalem Offset)
    video_end = max(0.0, v_off_sec) + vdur
    a_ends = [
        max(0.0, v_off_sec) + a_off_sec + he.probe_duration(ap) for ap in audio_paths
    ]
    s_ends = [
        max(0.0, v_off_sec) + s_off_sec + he.probe_subtitle_duration(sp)
        for sp in sub_paths
    ]
    longest_extra_end = max(a_ends + s_ends, default=0.0)
    need_extend = longest_extra_end > (video_end + 0.05)

    if not batch_mode and need_extend and a.extend is None:
        msg = tr(
            {
                "de": f"Externe Spuren sind länger ({he.format_time(longest_extra_end)}) als das Video ({he.format_time(video_end)}). Video verlängern?",
                "en": f"External tracks are longer ({he.format_time(longest_extra_end)}) than the video ({he.format_time(video_end)}). Extend video?",
            }
        )
        res = ui.ask_yes_no(msg, default=False, back_option=False)
        a.extend = bool(res) if res is not None else False

    # A) Verlängern oder B) positiver Lead-In → Concat-Pfad (ohne zweite Nachfrage)
    if bool(a.extend) or (v_off_sec > 0.0001):
        # Werte ins Args-Objekt schreiben, damit der Concat-Pfad sie übernimmt:
        a.offset = f"{v_off_sec}s"
        a.audio_offset = f"{a_off_sec}s"
        a.subtitle_offset = f"{s_off_sec}s"
        _merge_video_concat(
            a,
            batch_mode,
            [str(video_path)],
            audio_paths,
            sub_paths,
            preserved_cover=preserved_cover,
            precollected_offsets=True,  # ← verhindert doppelte Prompts
        )
        return

    # C) Fast-Mux-Pfad (kein Re-encode, v_off_sec <= 0):
    exist_a = he.count_audio_streams(str(video_path))
    exist_s = _probe_sub_count(str(video_path))

    if not batch_mode:
        default_name = Path(_default_out_name(files, target_container, "merged"))
        out_path = Path(ui.ask_for_filename(default_name, default=default_name.name))
        print()
    else:
        out_path = (
            Path(a.output)
            if a.output
            else Path(_default_out_name(files, target_container, "merged"))
        )
        if out_path.suffix == "":
            out_path = out_path.with_suffix(f".{target_container}")

    _print_summary(
        video_files=[str(video_path)],
        out_path=out_path,
        tgt_label=f"match-first → {he.probe_wh(str(video_path))[0]}x{he.probe_wh(str(video_path))[1]}",
        fps=int(round(he.probe_wh_fmt_fps(str(video_path))[3])) or 25,
        pause_sec=0.0,
        offset_sec=v_off_sec,
        audio_offset_sec=a_off_sec,
        subtitle_offset_sec=s_off_sec,
        container=target_container,
        vcodec_key=(
            chosen_codec_key or vec.normalize_codec_key(chosen_codec_key) or "h264"
        ),
        pix_fmt=(he.probe_wh_fmt_fps(str(video_path))[2] or "unknown"),
        alpha_on=_pix_fmt_has_alpha(he.probe_wh_fmt_fps(str(video_path))[2]),
        preset_name=None,
        extend_opt=False,
        audio_files=audio_paths,
        subtitle_files=sub_paths,
    )

    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-fflags",
        "+genpts",
    ]

    # Video – negativer Offset = vorne wegschneiden
    if v_off_sec < -0.0001:
        cmd += ["-ss", f"{-v_off_sec:.6f}"]
    cmd += ["-i", str(video_path)]

    input_indices_audio: List[int] = []
    input_indices_subs: List[int] = []
    next_idx = 1

    # >>> Eff. Offsets = global + individuell (kann auch negativ sein)
    eff_a_off = a_off_sec + v_off_sec
    eff_s_off = s_off_sec + v_off_sec

    for ap in audio_paths:
        if abs(eff_a_off) > 0.0001:
            cmd += ["-itsoffset", f"{eff_a_off}"]
        cmd += ["-i", str(ap)]
        input_indices_audio.append(next_idx)
        next_idx += 1

    for sp in sub_paths:
        if abs(eff_s_off) > 0.0001:
            cmd += ["-itsoffset", f"{eff_s_off}"]
        if Path(sp).suffix.lower() in (".srt", ".sub"):
            cmd += ["-sub_charenc", "UTF-8"]
        cmd += ["-i", str(sp)]
        input_indices_subs.append(next_idx)
        next_idx += 1

    cmd += ["-map", "0:v:0", "-map", "0:a?", "-map", "0:s?"]
    for idx in input_indices_audio:
        cmd += ["-map", f"{idx}:a:0"]
    for idx in input_indices_subs:
        cmd += ["-map", f"{idx}:s:0?"]

    cmd += ["-c:v", "copy", "-c:a", "copy"]
    for k, ap in enumerate(audio_paths):
        per = vec.build_audio_args(
            target_container, preset_name=None, input_path=Path(ap)
        )
        cmd += _per_streamize_audio_args(per, exist_a + k)

    if target_container in ("mp4", "m4v", "mov"):
        cmd += ["-c:s", "mov_text"]
    else:
        cmd += ["-c:s", "copy"]

    for k, ap in enumerate(audio_paths):
        title, lang3 = _infer_default_audio_title_and_lang(Path(ap))
        ai = exist_a + k
        if title:
            cmd += ["-metadata:s:a:%d" % ai, f"title={title}"]
        if lang3:
            cmd += ["-metadata:s:a:%d" % ai, f"language={lang3}"]

    for k, sp in enumerate(sub_paths):
        title, lang3 = _infer_default_subtitle_title_and_lang(Path(sp))
        si = exist_s + k
        if title:
            cmd += ["-metadata:s:s:%d" % si, f"title={title}"]
        if lang3:
            cmd += ["-metadata:s:s:%d" % si, f"language={lang3}"]

    if vdur > 0:
        # Bis Ende des verschobenen Videos laufen lassen
        cmd += ["-t", f"{video_end:.3f}"]

    cmd = vec.apply_container_codec_quirks(cmd[:], target_container, chosen_codec_key)
    cmd = _strip_vf_if_streamcopy(cmd)
    vec.ensure_pre_output_order(cmd)
    cmd.append(str(out_path))

    cmd = autotune_final_cmd(video_path, cmd)

    total_est = max(video_end, longest_extra_end)
    pw.run_ffmpeg_with_progress(
        out_path.name,
        cmd,
        _("merging_tracks"),
        _("merged_to_output"),
        total_duration=total_est,
        BATCH_MODE=batch_mode,
    )
    _apply_preserved_thumbnail(out_path, preserved_cover, BATCH_MODE=batch_mode)


def _to_path_str(x: Any) -> str:
    # Laufzeit-Guards: alles, was real Pfad-ähnlich ist, direkt durch Path; Rest → Stringifizieren
    if isinstance(x, (str, Path, os.PathLike)):
        return str(Path(x))
    return str(Path(str(x)))


def _flatten_picks(x: Any) -> List[str]:
    """Akzeptiert Einzelpfad, Sequenzen, Sets, Generatoren und das Dict-Format von ui.select_files_interactively."""
    from collections.abc import Iterable as CAIterable

    if x is None:
        return []

    # Einzelwerte direkt
    if isinstance(x, (str, Path, os.PathLike)):
        return [_to_path_str(x)]

    # Dict-Format vom Picker: {"videos":[...], "audios":[...], "subs":[...]} o.ä.
    if isinstance(x, dict):
        out: List[str] = []
        for k in ("videos", "audios", "subs", "files"):
            v = x.get(k)
            if v:
                out.extend(_flatten_picks(v))
        return out

    # Generische Iterables (aber keine strings/bytes)
    if isinstance(x, CAIterable):
        out: List[str] = []
        for p in list(x):
            out.append(_to_path_str(p))
        return out

    # Fallback: alles andere stringifizieren
    return [_to_path_str(x)]


def _allowed_path(p: str) -> bool:
    return (
        _ext_matches(p, VIDEO_EXTS)
        or _ext_matches(p, AUDIO_EXTS)
        or _ext_matches(p, SUB_EXTS)
    )


def _extract_first_thumbnail_candidate(paths: Sequence[str]) -> Optional[Path]:
    """
    Durchsucht die übergebenen Pfade in Reihenfolge und extrahiert das erste
    gefundene Thumbnail aus einer Videodatei. Gibt den Pfad zur temporären
    extrahierten Bilddatei zurück (oder None).
    """
    for f in paths:
        try:
            if _ext_matches(f, VIDEO_EXTS):
                p = Path(f)
                if vt.check_thumbnail(p, silent=True):
                    return vt.extract_thumbnail(p)
        except Exception:
            # still und robust weiter
            continue
    return None


def _apply_preserved_thumbnail(
    out_path: Path, preserved_cover: Optional[Path], *, BATCH_MODE: bool
) -> None:
    """
    Betten eines zuvor extrahierten Thumbnails ins Ausgabevideo + Aufräumen.
    Spiegelt die Logik aus dem convert-Modul.
    """
    try:
        if out_path.exists() and preserved_cover and preserved_cover.exists():
            vt.set_thumbnail(
                out_path, value=str(preserved_cover), BATCH_MODE=BATCH_MODE
            )
        else:
            co.print_info(_("no_thumbnail_found"))
    except Exception as e:
        co.print_warning(_("embedding_skipped") + f": {e}")
    finally:
        try:
            if preserved_cover and preserved_cover.exists():
                preserved_cover.unlink(missing_ok=True)
        except Exception:
            pass


def _ensure_args(arglike: MergeArgs | type[MergeArgs]) -> MergeArgs:
    return arglike() if isinstance(arglike, type) else arglike


# ──────────────────────────────────────────────────────────────────────────────
# Kernfunktion
# ──────────────────────────────────────────────────────────────────────────────


def merge(args: MergeArgs | type[MergeArgs]) -> None:
    # 1) “prepare_inputs” darf gerne Batch/Interactive bestimmen – aber nicht die Files hart limitieren
    batch_mode, files_any = he.prepare_inputs(
        args, VIDEO_EXTS, AUDIO_EXTS, SUB_EXTS, files_required=True
    )
    files: List[str] = [p for p in _flatten_picks(files_any) if _allowed_path(p)]

    co.print_start(_("merge_method"))

    # 2) CLI-Positional-Args ROBUST einsammeln (ohne den Wert von -o/--output)
    argv = sys.argv[1:]
    cli_raw: List[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("-o", "--output"):
            # der nächste Token ist der Ausgabename -> nicht als Eingabe behandeln
            skip_next = True
            continue
        if tok.startswith("--output="):
            # Form --output=NAME direkt überspringen
            continue
        # alles, was nicht wie eine Option aussieht, als positional candidate sammeln
        if not tok.startswith("-"):
            cli_raw.append(tok)

    files += [str(Path(a)) for a in cli_raw if _allowed_path(a)]

    # 3) Deduplizieren (Resolve pfadstabil), Reihenfolge bewahren
    seen = set()
    unique_files: List[str] = []
    for f in files:
        try:
            r = str(Path(f).resolve())
        except Exception:
            r = str(Path(f))
        if r not in seen:
            unique_files.append(f)
            seen.add(r)
    files = unique_files

    preserved_cover = _extract_first_thumbnail_candidate(files)

    # 4) Interaktiv: wenn noch <2, Picker anbieten (Mehrfachauswahl erlaubt)
    if not batch_mode and len(files) < 2:
        co.print_line(_("select_second_file_to_merge"), color="soft_blue")
        picks = ui.select_files_interactively(VIDEO_EXTS, AUDIO_EXTS, SUB_EXTS)
        more = [p for p in _flatten_picks(picks) if _allowed_path(p)]
        for p in more:
            rp = str(Path(p).resolve())
            if rp not in seen:
                files.append(p)
                seen.add(rp)

    if len(files) < 2:
        co.print_error(_("need_at_least_two_files"))
        return

    # Falls nur eine Datei vorhanden ist und interaktiv → weitere(n) nachfordern
    if len(files) == 1:
        if batch_mode:
            co.print_error(_("need_at_least_two_files"))
            sys.exit(0)
        else:
            co.print_line(_("select_second_file_to_merge"), color="soft_blue")
            picks = ui.select_files_interactively(VIDEO_EXTS, AUDIO_EXTS, SUB_EXTS)
            if not picks:
                co.print_error(_("need_at_least_two_files"))
                return

            # NEU: Mehrfachauswahl unterstützen (+ robust normalisieren)
            def _to_list(x: Any) -> List[str]:
                if x is None:
                    return []
                if isinstance(x, (list, tuple, set)):
                    return [str(Path(p)) for p in x]
                return [str(Path(x))]

            cand = _to_list(picks)

            # Nur erlaubte Typen übernehmen (Video/Audio/Untertitel)
            def _allowed(p: str) -> bool:
                return (
                    _ext_matches(p, VIDEO_EXTS)
                    or _ext_matches(p, AUDIO_EXTS)
                    or _ext_matches(p, SUB_EXTS)
                )

            cand = [p for p in cand if _allowed(p)]

            # Deduplizieren via resolve(), Reihenfolge bewahren
            seen = {str(Path(x).resolve()) for x in files}
            appended = []
            for p in cand:
                rp = str(Path(p).resolve())
                if rp not in seen:
                    files.append(p)
                    appended.append(p)
                    seen.add(rp)

            # Sicherstellen, dass wir jetzt ≥2 gültige Dateien haben
            if len([f for f in files if _allowed(f)]) < 2:
                co.print_error(_("need_at_least_two_files"))
                return

    # a: MergeArgs = cast(MergeArgs, args if isinstance(args, MergeArgs) else args)
    a = _ensure_args(args)

    # Mergetyp ermitteln
    kind = _classify_merge(files)

    if kind == "video_concat":
        video_files = [f for f in files if _ext_matches(f, VIDEO_EXTS)]
        if len(video_files) < 2:
            co.print_error(_("no_video_files_found"))
            return
        extra_audio_files = [f for f in files if _ext_matches(f, AUDIO_EXTS)]
        extra_sub_files = [f for f in files if _ext_matches(f, SUB_EXTS)]
        _merge_video_concat(
            a,
            batch_mode,
            video_files,
            extra_audio_files,
            extra_sub_files,
            preserved_cover=preserved_cover,
        )
        co.print_finished(_("merge_method"))
        return

    if kind == "video_plus_assets":
        video_path = _first_video(files)
        if not video_path:
            co.print_error(_("no_video_files_found"))
            return
        audios = [f for f in files if _ext_matches(f, AUDIO_EXTS)]
        subs = [f for f in files if _ext_matches(f, SUB_EXTS)]
        _merge_video_plus_assets(
            a,
            batch_mode,
            Path(video_path),
            audios,
            subs,
            files,
            preserved_cover=preserved_cover,
        )
        co.print_finished(_("merge_method"))
        return

    # Fallback/Normalfall: Video concat
    video_files = [f for f in files if _ext_matches(f, VIDEO_EXTS)]
    if len(video_files) < 2:
        co.print_error(_("no_video_files_found"))
        return

    # 🔧 NEU: zusätzliche Audios/Untertitel an den Concat-Pfad übergeben
    extra_audio_files = [f for f in files if _ext_matches(f, AUDIO_EXTS)]
    extra_sub_files = [f for f in files if _ext_matches(f, SUB_EXTS)]

    _merge_video_concat(
        a,
        batch_mode,
        video_files,
        extra_audio_files,
        extra_sub_files,
        preserved_cover=preserved_cover,
    )

    co.print_finished(_("merge_method"))
