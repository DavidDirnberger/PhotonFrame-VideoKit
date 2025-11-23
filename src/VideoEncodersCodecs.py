#!/usr/bin/env python3
# videoEncodersCodecs.py
from __future__ import annotations

import json
import re
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from typing import (
    OrderedDict as OrderedDictType,
)

from VideoEncodersCodecs_support import (
    _active_encoder,
    _append_if_absent,
    _apply_dnx_rules_if_needed,
    _apply_h264_vui_flags_if_needed,
    _build_cmd_for_encoder,
    _build_h264_near_lossless_chain,
    _build_mpeg4_near_lossless_chain,
    _choose_strict_lossless_encoder,
    _codec_core_from_key,
    _container_allows_codec,
    _container_supports_chapters,
    _dedupe_vf_inplace,
    _encoder_core_hw,
    _encoder_family,
    _encoder_for_codec,
    _encoder_supports_pix_fmt,
    _enforce_h263_size,
    _ffprobe_geometry,
    _get_preset_spec,
    _h264_near_lossless_args_mkv,
    _h264_near_lossless_args_mp4,
    _has_filter,
    _inject_qsv_device,
    _inject_vaapi_device,
    _input_has_chapters,
    _insert_before_output,
    _is_stream_copy,
    _lossless_video_args_for_encoder,
    _mpeg1_near_lossless_args,
    _mpeg2_near_lossless_args,
    _mpeg4_near_lossless_args_mp4,
    _norm_matrix,
    _norm_range,
    _normalize_codec_key,
    _nvenc_usable,
    _output_index,
    _parse_encoder_names,
    _parse_scale,
    _pick_sw_av1_encoder,
    _policy_candidates_same_family,
    _preset_or_arg,
    _probe_subtitle_streams,
    _probe_video_codec,
    _required_alignment,
    _run_ffmpeg_encoders,
    _sanitize_movflags_inplace,
    _set_format_in_vf,
    _set_kv_arg,
    _strip_filter_from_vf,
    _strip_hw_device_options,
    _strip_nvenc_private_opts,
    _strip_options,
    _try_quick_encode,
    _vaapiify_vf,
    _wants_mezzanine_dnx,
)

import definitions as defin
from audio_support import (
    _avi_audio_args_for_source,
    _webm_audio_args_for_source,
    build_audio_args,
)

# local
from i18n import _
from video_pixfmt import (
    PixfmtDecision,
    _apply_alpha_rules,
    _assess_alpha_preservation,
    _color_metadata_args,
    _dedupe_color_tags,
    _ensure_quality_defaults,
    _pixfmt_api_decision_for,
    _should_preserve_src_pix_fmt_in_lossless,
    apply_color_signaling,
    infer_target_pix_fmt_from_plan,
    playback_pix_fmt_for,
    probe_color_metadata,
)

# === Lossless Assessment ==============================================
LosslessStatus = Literal["strict", "visual", "none"]


@dataclass
class LosslessAssessment:
    status: LosslessStatus
    reasons: List[str]
    recommended_encoder: Optional[str] = None
    strict_target_pix_fmt: Optional[str] = None
    must_reformat_pix_fmt: bool = False
    must_scale: bool = False
    must_change_fps: bool = False


PresetLike = Mapping[str, Any]
CodecMap = OrderedDictType[str, Optional[str]]


# Unzuverlässige/broken VAAPI-Encoder (Encode-Profile fehlen auf den meisten Systemen)
_BROKEN_OR_UNSTABLE_HW_ENCODERS: Set[str] = {"vp8_vaapi", "vp9_qsv"}


def has_filter(name: str, ffmpeg_bin: str = "ffmpeg") -> bool:
    return _has_filter(name, ffmpeg_bin)


def detect_container_from_path(path: Path) -> Optional[str]:
    ext = path.suffix.lower().lstrip(".")
    return defin.EXT_TO_CONTAINER.get(ext)


def probe_video_codec(path: Path) -> Optional[str]:
    return _probe_video_codec(path)


def ffprobe_geometry(
    path: Path, ffprobe_bin: str = "ffprobe"
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    return _ffprobe_geometry(path, ffprobe_bin)


# ============================================================
#                      Encoder / Presets
# ============================================================
def encoder_for_codec(codec_name: Optional[str]) -> str:
    return _encoder_for_codec(codec_name)


def prepare_encoder_maps(
    files: List[str],
    format_choice: Optional[str],
    convert_format_descriptions: Mapping[str, Mapping[str, Any]],
    prefer_hw: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[str, CodecMap]:
    """
    Liefert pro benötigtem Container eine (typisierte) Encoder-Map.
    Behebt Unknown-Warnungen durch 'CodecMap' und präzisere Signaturen.
    """
    needed_containers: set[str] = set()

    if format_choice is None:
        format_choice = "keep"

    if format_choice != "keep":
        needed_containers.add(format_choice.lower())
    else:
        for f in files:
            c = detect_container_from_path(Path(f))
            if c:
                needed_containers.add(c)

    container_to_map: Dict[str, CodecMap] = {}
    for container in sorted(needed_containers):
        emap: CodecMap = select_encoder_map_for_format(
            container,
            convert_format_descriptions,
            ffmpeg_bin=ffmpeg_bin,
            prefer_hw=prefer_hw,
            include_unavailable=False,
        )
        container_to_map[container] = emap
    return container_to_map


# === Punkt 2: Containerbasierter Codec-Fallback ==============================
@lru_cache(maxsize=4)
def _available_encoders(ffmpeg_bin: str = "ffmpeg") -> Set[str]:
    """Menge verfügbarer Encoder – gecached und wiederverwendbar."""
    return _parse_encoder_names(_run_ffmpeg_encoders(ffmpeg_bin))


def _first_working_encoder_for(
    key: str, container: str, available: set[str]
) -> Optional[str]:
    c = (container or "").lower()
    k = (key or "").lower()

    cand = (
        defin.CONTAINER_CODEC_OVERRIDES.get((c, k))
        or defin.CODEC_ENCODER_CANDIDATES.get(k, [])
        or defin.CODEC_FALLBACK_POLICY.get(k, [])
    )

    for enc in cand:
        if enc in available:
            return enc
    return None


def resolve_codec_key_with_container_fallback(
    container: str,
    requested_key: str,
    ffmpeg_bin: str = "ffmpeg",
    allow_cross_codec_fallback: bool = True,
) -> Tuple[str, Optional[str]]:
    c = (container or "").lower()
    want = normalize_codec_key(requested_key) or ""
    avail = _available_encoders(ffmpeg_bin)

    def ok(key: str) -> Optional[str]:
        return (
            _first_working_encoder_for(key, c, avail)
            if container_allows_codec(c, key)
            else None
        )

    enc = ok(want)
    if enc or not allow_cross_codec_fallback:
        return want, enc

    for alt in defin.FALLBACK_BY_CONTAINER.get(c, ()):
        enc = ok(alt)
        if enc:
            return alt, enc

    for k in defin.CODEC_ENCODER_CANDIDATES.keys():
        enc = ok(k)
        if enc:
            return k, enc

    return want, None


def _candidate_list_for(codec_key: str, container: str) -> List[str]:
    override = defin.CONTAINER_CODEC_OVERRIDES.get(
        (container.lower(), codec_key.lower())
    )
    if override:
        return override
    return defin.CODEC_ENCODER_CANDIDATES.get(codec_key.lower(), [])


def _apply_hw_preference(candidates: List[str], prefer_hw: bool) -> List[str]:
    if prefer_hw:
        return candidates
    sw = [c for c in candidates if not c.endswith(defin.HW_SUFFIXES)]
    return sw if sw else candidates


def _extract_codec_keys_for_format(
    convert_desc: Mapping[str, Mapping[str, Any]], fmt: str
) -> List[str]:
    """
    Liefert die erlaubten Video-Codec-Keys für ein Ziel-Format.
    Fix für Pylance: Vor Iteration konsequent casten (Mapping[str, Any] / Iterable[Any]),
    damit in Comprehensions keine Unknown-Typen entstehen.
    """
    entry_any: Any = convert_desc.get(fmt) or convert_desc.get(fmt.lower()) or {}

    # Fall 1: Mapping mit (möglichen) Feldern
    if isinstance(entry_any, Mapping):
        entry_map: Mapping[str, Any] = cast(Mapping[str, Any], entry_any)
        for key in (
            "codecs",
            "video_codecs",
            "video",
            "allowed_codecs",
            "allowed_video_codecs",
        ):
            if key in entry_map:
                raw: Any = entry_map[key]

                # Sequenz/Set → Iterable[Any] und dann str(...)
                if isinstance(raw, (list, tuple, set)):
                    seq: Iterable[Any] = cast(Iterable[Any], raw)
                    return [str(item) for item in seq]

                # Mapping → Mapping[str, Any] und key-Iteration streng getypt
                if isinstance(raw, Mapping):
                    m: Mapping[str, Any] = cast(Mapping[str, Any], raw)
                    return [str(k) for k in m.keys()]

                # None → leer; sonst Single-Wert in String wandeln
                if raw is None:
                    return []
                return [str(raw)]
        return []

    # Fall 2: direkt eine Sequenz (Liste/Tupel/Set)
    if isinstance(entry_any, (list, tuple, set)):
        seq2: Iterable[Any] = cast(Iterable[Any], entry_any)
        return [str(item) for item in seq2]

    # Unbekannte Struktur → leere Liste
    return []


def _select_encoders_for_format(
    fmt: str,
    convert_format_descriptions: Mapping[str, Mapping[str, Any]],
    ffmpeg_bin: str = "ffmpeg",
    prefer_hw: bool = True,
) -> List[Tuple[str, str]]:
    """
    Wählt pro erlaubtem Codec-Key einen verfügbaren Encoder (unter Berücksichtigung
    der Hardware-Priorität). Typisiert so, dass Pylance keine Unknowns propagiert.
    """
    codec_keys = _extract_codec_keys_for_format(convert_format_descriptions, fmt)
    if not codec_keys:
        return []

    encoders_text = _run_ffmpeg_encoders(ffmpeg_bin)
    available = set(_parse_encoder_names(encoders_text))

    result: List[Tuple[str, str]] = []
    for key in codec_keys:
        candidates = _candidate_list_for(key, fmt)
        if not candidates:
            continue
        candidates = _apply_hw_preference(candidates, prefer_hw)
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            result.append((key, chosen))
    return result


def select_encoder_map_for_format(
    fmt: str,
    convert_format_descriptions: Mapping[str, Mapping[str, Any]],
    ffmpeg_bin: str = "ffmpeg",
    prefer_hw: bool = True,
    include_unavailable: bool = False,
) -> CodecMap:
    """
    Baut ein OrderedDict[codec_key -> encoder_name|None] in der gewünschten
    Reihenfolge. Durch den Rückgabe-Typ 'CodecMap' sind Items klar getypt.
    """
    codec_order = _extract_codec_keys_for_format(convert_format_descriptions, fmt)
    pairs = _select_encoders_for_format(
        fmt, convert_format_descriptions, ffmpeg_bin=ffmpeg_bin, prefer_hw=prefer_hw
    )
    chosen_by_key = {k: v for k, v in pairs}

    out: CodecMap = OrderedDict()
    for key in codec_order:
        if key in chosen_by_key:
            out[key] = chosen_by_key[key]
        elif include_unavailable:
            out[key] = None
    return out


def _pixfmt_apply_decision_inplace(cmd: List[str], decision: PixfmtDecision) -> None:
    """
    Überträgt die Entscheidung in den FFmpeg-Command:
    - hängt 'format=<pf>' an die -vf-Kette, wenn decision.vf_terminal_format gesetzt ist
    - setzt ggf. '-pix_fmt <pf>' wenn decision.pix_fmt_flag gesetzt ist
    - dedupliziert Filter und -color_* Tags
    """
    if not decision:
        return

    # 1) format=... via -vf
    pf_vf = decision.vf_terminal_format
    if isinstance(pf_vf, str) and pf_vf.strip():
        _set_format_in_vf(cmd, pf_vf.strip())

    # 2) explizites -pix_fmt (nur wenn gefordert)
    pf_cli = decision.pix_fmt_flag
    if isinstance(pf_cli, str) and pf_cli.strip():
        _set_kv_arg(cmd, "-pix_fmt", pf_cli.strip())

    # 3) Aufräumen / Dedupe
    dedupe_vf_inplace(cmd)
    _dedupe_color_tags(cmd)
    sanitize_movflags_inplace(cmd)


# ============================================================
#              Container ⇄ Codec Regeln & Audio
# ============================================================
def container_allows_codec(container: str, codec_key: str) -> bool:
    return _container_allows_codec(container, codec_key)


def suggest_codec_for_container(container: str) -> str:
    return {
        "webm": "vp9",
        "mp4": "h264",
        "avi": "mpeg4",
        "flv": "h263",
        "mpeg": "mpeg2video",
        "mkv": "h264",
        "mov": "prores",
    }.get(container.lower(), "h264")


# ---------- All streams mapping (audio+subs where possible) -----------
def build_stream_mapping_args(
    container: str, input_path: Optional[Path] = None
) -> List[str]:
    c = (container or "").lower()
    args: List[str] = ["-map", "0:v:0", "-map", "0:a?"]

    # === MP4/M4V/MOV: wie früher – nur textbasierte Subs → mov_text, KEINE Attachments ===
    if c in ("mp4", "m4v", "mov"):
        mapped_any = False
        if input_path:
            try:
                probe = subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "s",
                        "-show_entries",
                        "stream=index,codec_name",
                        "-of",
                        "json",
                        str(input_path),
                    ],
                    text=True,
                )
                data = cast(Dict[str, Any], json.loads(probe))
                streams = cast(List[Dict[str, Any]], data.get("streams") or [])
                text_ok = {"subrip", "ass", "ssa", "webvtt", "mov_text"}
                for st in streams:
                    idx = st.get("index")
                    codec = (st.get("codec_name") or "").lower()
                    if isinstance(idx, int) and codec in text_ok:
                        args += ["-map", f"0:{idx}"]
                        mapped_any = True
            except Exception:
                mapped_any = False
        if mapped_any:
            args += ["-c:s", "mov_text"]

        # Kapitel nur, wenn unterstützt
        if _container_supports_chapters(c) and _input_has_chapters(input_path):
            args += ["-map_chapters", "0"]
        return args

    # === MKV/Matroska: intelligente Behandlung inkl. Attachments ===
    if c in ("mkv", "matroska"):
        copy_ok = {
            # textbasiert
            "subrip",
            "ssa",
            "ass",
            "webvtt",
            # bildbasiert
            "hdmv_pgs_subtitle",
            "dvd_subtitle",
            "dvb_subtitle",
            "vobsub",
        }
        needs_srt = {
            # mp4/mov-Texttypen, die in MKV nicht direkt passen
            "mov_text",
            "text",
            "eia_608",
            "eia_708",
            "cc_dec",
        }

        out_sub_index = 0
        srt_overrides: List[int] = []

        if input_path:
            for idx, cname, ctag in _probe_subtitle_streams(input_path):
                cn = (cname or "").lower()
                tg = (ctag or "").lower()
                is_tx3g = (tg == "tx3g") or (cn in needs_srt)
                if cn in copy_ok:
                    args += ["-map", f"0:{idx}"]
                    out_sub_index += 1
                elif is_tx3g:
                    args += ["-map", f"0:{idx}"]
                    srt_overrides.append(out_sub_index)
                    out_sub_index += 1
                else:
                    # Unbekannt/inkompatibel → auslassen
                    continue

        if out_sub_index > 0:
            args += ["-c:s", "copy"]
            for i in srt_overrides:
                args += [f"-c:s:{i}", "srt"]

        # Attachments & Kapitel nur bei MKV
        args += ["-map", "0:t?", "-c:t", "copy"]
        if _container_supports_chapters(c) and _input_has_chapters(input_path):
            args += ["-map_chapters", "0"]
        return args

    # === WEBM: sicherheitshalber NUR V/A (Subs/Attachments weglassen) ===
    if c == "webm":
        if _container_supports_chapters(c) and _input_has_chapters(input_path):
            args += ["-map_chapters", "0"]
        return args

    # === AVI: NUR V/A (Subs/Attachments nicht unterstützt) ===
    if c == "avi":
        if _container_supports_chapters(c) and _input_has_chapters(input_path):
            args += ["-map_chapters", "0"]
        return args

    # === Default (wie alte Funktion außerhalb mp4/mkv): konservativ NUR V/A ===
    if _container_supports_chapters(c) and _input_has_chapters(input_path):
        args += ["-map_chapters", "0"]
    return args


# ============================================================
#                    Scale / Filter-Bausteine
# ============================================================


def _compute_safe_scale(
    src_w: int,
    src_h: int,
    req_scale: Optional[str],
    chosen_encoder: Optional[str],
    *,
    allow_upscale: bool = True,
    preserve_ar: bool = True,
) -> Optional[str]:
    if not src_w or not src_h or not req_scale:
        return None

    req_w, req_h = _parse_scale(req_scale)
    if not req_w and not req_h:
        return None

    # Seitenverhältnis robust bestimmen
    try:
        ar = float(src_w) / float(src_h)
        if ar <= 0:
            return None
    except Exception:
        return None

    if preserve_ar:
        if (req_w and req_w > 0) and (req_h and req_h > 0):
            scale = min(req_w / src_w, req_h / src_h)
            w = max(1, int(round(src_w * scale)))
            h = max(1, int(round(w / ar)))
            if h > req_h:
                h = int(req_h)
                w = max(1, int(round(h * ar)))
        elif req_w and req_w > 0:
            w = int(req_w)
            h = max(1, int(round(w / ar)))
        elif req_h and req_h > 0:
            h = int(req_h)
            w = max(1, int(round(h * ar)))
        else:
            return None
    else:
        if not (req_w and req_h and req_w > 0 and req_h > 0):
            return None
        w = int(req_w)
        h = int(req_h)
        if not allow_upscale:
            w = min(w, src_w)
            h = min(h, src_h)

    # Encoder-Ausrichtung
    align = _required_alignment(chosen_encoder or "")
    if align > 1:
        w = max(align, w - (w % align))
        h = max(align, h - (h % align))

    if w == src_w and h == src_h:
        return None
    return f"{w}:{h}"


def compute_effective_wh(
    src_w: int,
    src_h: int,
    req_scale: Optional[str],
    chosen_encoder: Optional[str],
    preserve_ar: bool,
    allow_upscale: bool = True,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Liefert (W,H), wie sie build_transcode_plan/_compute_safe_scale auch wählen würde.
    - preserve_ar=True: skaliert innerhalb der gewünschten Grenzen bei konstantem AR.
    - preserve_ar=False: nimmt exakt req_scale (falls vorhanden) und rundet nur Encoder-Alignment.
    """
    # Versuch 1: exakt wie der Plan rechnen
    safe = _compute_safe_scale(
        src_w,
        src_h,
        req_scale,
        chosen_encoder,
        allow_upscale=allow_upscale,
        preserve_ar=preserve_ar,
    )
    if safe:
        w, h = _parse_scale(safe)
        return (w, h)

    # Versuch 2: ohne AR-Bindung, nur Alignment
    if req_scale:
        w, h = _parse_scale(req_scale)
        if w and (h or h == 0):
            align = _required_alignment(chosen_encoder or "")
            if preserve_ar:
                # Ohne safe-Scale bleibt (None,None) → Quelle zurückgeben
                return (None, None)
            # Exakt erzwingen + Alignment
            if align > 1:
                w = max(align, w - (w % align))
                if h and h > 0:
                    h = max(align, h - (h % align))
            return (w, h if h and h > 0 else None)

    return (None, None)


def infer_output_wh_from_cmd(
    ffmpeg_cmd: List[str], src_w: int, src_h: int
) -> Tuple[int, int]:
    """
    Versucht aus der finalen -vf-Kette die Zielgröße zu ermitteln.
    Unterstützt:
      - scale=WxH
      - scale=trunc(iw/N)*N:trunc(ih/N)*N
      - scale_vaapi=... (wie oben)
      - zscale=width=...:height=... mit Werten oder trunc(iw/N)*N
    Fällt andernfalls auf (src_w, src_h) zurück.
    """
    # 1) -vf/-filter:v lesen
    vf = None
    if "-vf" in ffmpeg_cmd:
        try:
            vf = str(ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1])
        except Exception:
            vf = None
    if not vf and "-filter:v" in ffmpeg_cmd:
        try:
            vf = str(ffmpeg_cmd[ffmpeg_cmd.index("-filter:v") + 1])
        except Exception:
            vf = None
    if not vf:
        return (src_w, src_h)

    # Hilfs-Evaluator: trunc(iw/N)*N → int
    def _eval_dim(expr: str) -> Optional[int]:
        s = expr.strip().lower()
        if s.isdigit():
            return int(s)
        # trunc(iw/N)*N  oder  trunc(ih/N)*N
        m = re.match(r"trunc\(\s*i([wh])\s*/\s*(\d+)\s*\)\s*\*\s*(\d+)", s)
        if m:
            which = m.group(1)  # 'w' oder 'h'
            n1 = int(m.group(2))
            n2 = int(m.group(3))
            if n1 != n2:
                # konservativ: trotzdem mit n1 runden
                pass
            base = src_w if which == "w" else src_h
            if n1 > 0:
                return (base // n1) * n1
        # simple iw/ih
        if s == "iw":
            return src_w
        if s == "ih":
            return src_h
        return None

    # 2) scale=WxH oder scale=expr:expr
    m = re.search(r"(?:^|,)scale\s*=\s*([^:,]+)\s*:\s*([^,]+)(?:,|$)", vf)
    if not m:
        # 2b) scale_vaapi=...
        m = re.search(r"(?:^|,)scale_vaapi\s*=\s*([^:,]+)\s*:\s*([^,]+)(?:,|$)", vf)

    if m:
        w_expr, h_expr = m.group(1), m.group(2)
        w = _eval_dim(w_expr)
        h = _eval_dim(h_expr)
        if w and h:
            return (w, h)

    # 3) zscale=width=…:height=…
    m = re.search(r"(?:^|,)zscale\s*=\s*([^,]+)(?:,|$)", vf)
    if m:
        params = m.group(1)  # "width=...,height=...,..."
        kw = dict()
        for part in params.split(":"):
            if "=" in part:
                k, v = part.split("=", 1)
                kw[k.strip().lower()] = v.strip()
        w = _eval_dim(kw.get("width", "")) if "width" in kw else None
        h = _eval_dim(kw.get("height", "")) if "height" in kw else None
        if w and h:
            return (w, h)

    return (src_w, src_h)


def vf_join(*parts: str | None) -> str:
    tokens: List[str] = []
    for p in parts:
        if not p:
            continue
        tokens += [t.strip() for t in str(p).split(",") if t.strip()]
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return ",".join(out)


def ensure_terminal_pix_fmt(
    cmd: List[str],
    *,
    container: str,
    codec_key: str,
    input_path: Path,
    preset_name: Optional[str] = None,
) -> None:
    if _is_stream_copy(cmd):
        return

    # Zentrale Entscheidung der Pixfmt-/Alpha-/Color-Signalling-Logik
    dec = _pixfmt_api_decision_for(
        cmd=cmd,
        container=container,
        codec_key=codec_key,
        input_path=input_path,
        encoder_hint=None,
        preset_name=preset_name,
    )
    if dec:
        _pixfmt_apply_decision_inplace(cmd, dec)
        return

    # Defensive Mini-Defaults (nur falls API keine Entscheidung liefert)
    c = (container or "").lower()
    if c in ("mp4", "m4v", "webm"):
        _set_format_in_vf(cmd, "yuv420p")
    elif c in ("mkv", "matroska", "mov"):
        _set_format_in_vf(cmd, "yuv420p")

    dedupe_vf_inplace(cmd)


def ensure_pre_output_order(cmd: List[str]) -> None:
    if not cmd:
        return

    # erkennen, ob ein -filter_complex im Spiel ist → defensiver arbeiten
    has_filter_complex = "-filter_complex" in cmd

    DEVICE_SPEC_RE = re.compile(
        r"^(?:vaapi|cuda|qsv|opencl|vulkan|d3d11va|dxva2)=[^,\s]+$", re.IGNORECASE
    )

    # ------------------- 1) Key-Value-Flags definieren -------------------
    # Bisher fehlte hier -filter_complex; außerdem gibt es dynamische Varianten (-c:a:1, -metadata:s:s:2, ...)
    pair_keys = {
        "-vf",
        "-filter:v",
        "-pix_fmt",
        "-color_range",
        "-color_primaries",
        "-color_trc",
        "-colorspace",
        "-strict",
        "-movflags",
        "-map_chapters",
        "-vsync",
        "-video_track_timescale",
        "-tag:v",
        "-vtag",
        "-map_metadata",
        "-x264-params",
        "-x265-params",
        "-aom-params",
        "-threads",
        "-filter_threads",
        "-max_muxing_queue_size",
        "-init_hw_device",
        "-filter_hw_device",
        "-vaapi_device",
        "-hwaccel",
        "-hwaccel_output_format",
        "-extra_hw_frames",
        "-filter_complex",  # ← NEU: den Graph-Wert schützen!
        "-map",
        "-metadata",
        "-disposition",  # ← Basispräfixe (spezifische Varianten fangen wir per Regex ab)
    }

    # Alles, was wie ein *dynamisches* Paar-Flag aussieht (z. B. -c:a:1, -b:a:2, -metadata:s:s:2, …)
    DYNAMIC_PAIR_RE = re.compile(
        r"^-(?:c|codec|b|ar|ac|map|metadata|disposition|filter|af|vf)(?::[^\s]+)*$",
        re.IGNORECASE,
    )

    # ------------------- 2) Tail ermitteln & Werte einsammeln -------------------
    out_idx = _output_index(cmd)
    tail = cmd[out_idx + 1 :]
    del cmd[out_idx + 1 :]

    def _is_valid_value_for(flag: str, val: str) -> bool:
        return not (val or "").startswith("-")

    j = 0
    standalone: List[str] = []
    while j < len(tail):
        tok = str(tail[j])
        is_pair = (tok in pair_keys) or DYNAMIC_PAIR_RE.match(tok) is not None
        if is_pair:
            val = None
            if j + 1 < len(tail) and _is_valid_value_for(tok, str(tail[j + 1])):
                val = str(tail[j + 1])
                j += 2
            else:
                # Spezialfall -init_hw_device: Gerätespezifikationen aus dem Tail ziehen
                if tok == "-init_hw_device":
                    pick = None
                    for k in range(j + 1, len(tail)):
                        v = str(tail[k]).strip()
                        if DEVICE_SPEC_RE.match(v):
                            pick = k
                            break
                    if pick is not None:
                        val = str(tail.pop(pick))
                        j += 1
            if val is not None:
                _insert_before_output(cmd, [tok, val])
            else:
                standalone.append(tok)
                j += 1
        else:
            standalone.append(tok)
            j += 1

    cmd += standalone

    # ------------------- 3) Schutz für Werte hinter Paar-Flags -------------------
    protected_value_idx: Set[int] = set()
    i = 0
    while i < len(cmd):
        t = str(cmd[i])
        is_pair = (t in pair_keys) or DYNAMIC_PAIR_RE.match(t) is not None
        if is_pair and i + 1 < len(cmd):
            v = str(cmd[i + 1])
            if _is_valid_value_for(t, v):
                protected_value_idx.add(i + 1)
            i += 2
            continue
        i += 1

    # Eingabepfade hinter -i ebenfalls schützen
    for k, t in enumerate(cmd):
        if t == "-i" and k + 1 < len(cmd):
            protected_value_idx.add(k + 1)

    # ------------------- 4) Stray Filter-Fragmente → nur wenn KEIN -filter_complex -------------------
    if not has_filter_complex:

        def _looks_like_filter(v: str) -> bool:
            lv = (v or "").strip().lower()
            if not lv or lv.startswith("-"):
                return False
            if DEVICE_SPEC_RE.match(lv):
                return False
            # Filtergraf-typische Syntax
            if any(ch in lv for ch in "[];") or ("," in lv and "=" in lv):
                return True
            # häufige Filterpräfixe (gekürzt)
            for pref in (
                "scale",
                "zscale",
                "format",
                "fps",
                "setpts",
                "setsar",
                "colorspace",
                "hwupload",
                "hwdownload",
                "pad",
                "crop",
                "overlay",
                "aformat",
                "aresample",
                "adelay",
                "amix",
                "asplit",
                "atempo",
            ):
                if lv.startswith(pref):
                    return True
            if "=" in lv:
                key = lv.split("=", 1)[0].strip()
                if key not in {
                    "crf",
                    "cq",
                    "qp",
                    "q",
                    "qmin",
                    "qmax",
                    "pix_fmt",
                    "b",
                    "b:v",
                    "b:a",
                    "r",
                    "g",
                    "profile",
                    "level",
                    "preset",
                    "tune",
                    "movflags",
                    "map",
                    "c",
                    "c:v",
                    "c:a",
                    "c:s",
                }:
                    return True
            return False

        # existierende -vf sammeln
        vf_strings: List[str] = []
        if "-vf" in cmd:
            i = cmd.index("-vf")
            if i + 1 < len(cmd):
                vf_strings.append(str(cmd[i + 1]))

        # Strays vor dem Output einsammeln
        stray: List[str] = []
        idx = 0
        out_idx = _output_index(cmd)
        while idx < min(out_idx, len(cmd)):
            if idx in protected_value_idx:
                idx += 1
                continue
            tok = str(cmd[idx])
            if not tok.startswith("-") and _looks_like_filter(tok):
                stray.append(tok)
                cmd.pop(idx)
                out_idx -= 1
                continue
            idx += 1

        if stray:
            merged = ",".join(
                [s for s in ([vf_strings[0]] if vf_strings else []) + stray if s]
            )
            if "-vf" in cmd:
                j = cmd.index("-vf")
                if j + 1 < len(cmd):
                    cmd[j + 1] = merged
                else:
                    cmd.insert(j + 1, merged)
            else:
                _insert_before_output(cmd, ["-vf", merged])

    # kleiner Cleanup wie gehabt
    sanitize_movflags_inplace(cmd)


def try_encode_with_fallbacks(
    base_cmd: List[str],
    codec_key: str,
    container: str,
    preset_name: str,
    vf_chain: Optional[str],
    ffmpeg_bin: str = "ffmpeg",
    preferred_encoder: Optional[str] = None,
) -> List[str]:
    enc_text = _run_ffmpeg_encoders(ffmpeg_bin)
    available = set(_parse_encoder_names(enc_text))

    desired_core = _codec_core_from_key(codec_key)
    policy_all = defin.CODEC_FALLBACK_POLICY.get(desired_core, [])

    candidates: List[str] = []
    if preferred_encoder:
        core_i, _ = _encoder_core_hw(preferred_encoder)
        if core_i == desired_core and preferred_encoder in available:
            candidates.append(preferred_encoder)

    for enc in policy_all:
        if enc in available:
            core_i, _ = _encoder_core_hw(enc)
            if core_i == desired_core and enc not in candidates:
                candidates.append(enc)

    if not candidates:
        sw_by_core = {
            "h264": "libx264",
            "hevc": "libx265",
            "vp9": "libvpx-vp9",
            "av1": (_pick_sw_av1_encoder() or "libaom-av1"),
            "mpeg2video": "mpeg2video",
            "mpeg1video": "mpeg1video",
        }
        fallback_sw = sw_by_core.get(desired_core) or encoder_for_codec(desired_core)
        if fallback_sw:
            candidates.append(fallback_sw)

    # --- Denylist & Stabilität (wie gehabt) ---
    candidates = [e for e in candidates if e not in _BROKEN_OR_UNSTABLE_HW_ENCODERS]

    # === VP9 – Stabil bevorzugt SW (libvpx-vp9), QSV nur bei ausdrücklichem Wunsch ===
    if desired_core == "vp9":
        if "libvpx-vp9" not in candidates and ("libvpx-vp9" in available):
            candidates.append("libvpx-vp9")
        if not (preferred_encoder and preferred_encoder.strip().lower() == "vp9_qsv"):
            candidates = [e for e in candidates if e != "vp9_qsv"]
        if "libvpx-vp9" in candidates:
            candidates = ["libvpx-vp9"] + [e for e in candidates if e != "libvpx-vp9"]

    # === Preset-basierte HW-Präferenz ===
    p = (preset_name or "").lower()
    hw_first_names = {"web", "casual", "messenger", "mobile", "fast", "stream"}
    sw_first_names = {"cinema", "hq", "archival", "master", "lossless", "pro"}

    prefer_hw = (p in hw_first_names) or (p not in sw_first_names)

    def _is_hw(e: str) -> bool:
        return _encoder_core_hw(e)[1] in {"nvenc", "qsv", "vaapi", "amf", "vtb"}

    # NVENC nur, wenn tatsächlich nutzbar (robuster auf Systemen ohne NVENC)
    filtered: List[str] = []
    for e in candidates:
        if e.endswith("_nvenc") and not _nvenc_usable():
            continue
        filtered.append(e)
    candidates = filtered

    hw_list = [e for e in candidates if _is_hw(e)]
    sw_list = [e for e in candidates if not _is_hw(e)]
    candidates = (hw_list + sw_list) if prefer_hw else (sw_list + hw_list)

    # --- Fix: VP8 niemals via VAAPI encoden (bekannt instabil) ---
    if desired_core == "vp8":
        if "libvpx" in candidates:
            candidates = ["libvpx"] + [e for e in candidates if e != "libvpx"]

    last_cmd: Optional[List[str]] = None
    for enc in candidates:
        cmd0 = base_cmd[:]
        core_i, hw_i = _encoder_core_hw(enc)

        if core_i != desired_core:
            continue

        vf = vf_chain or ""
        if hw_i == "qsv":
            _inject_qsv_device(cmd0, enc)
            align = 16
            if "scale=" not in vf:
                vf = _append_if_absent(
                    vf, f"scale=trunc(iw/{align})*{align}:trunc(ih/{align})*{align}"
                )
            vf = _append_if_absent(vf, "format=nv12")
        elif hw_i == "nvenc":
            align = 8 if core_i == "av1" else 2
            if "scale=" not in vf and align > 1:
                vf = _append_if_absent(
                    vf, f"scale=trunc(iw/{align})*{align}:trunc(ih/{align})*{align}"
                )
            vf = _append_if_absent(vf, "format=yuv420p")
        elif hw_i == "vaapi":
            if not _inject_vaapi_device(cmd0):
                continue
            want_scale = "scale=" in (vf_chain or "")
            vf = _vaapiify_vf(vf_chain, want_scale=True if want_scale else False)

        test_cmd = _build_cmd_for_encoder(cmd0, enc, preset_name, container, vf)
        last_cmd = test_cmd
        if hw_i == "qsv" and core_i == "vp9":
            # QSV verträgt keine libvpx-typischen Flags
            _strip_vp9_libvpx_opts_inplace(test_cmd)

        if _try_quick_encode(test_cmd + ["__probe__.mkv"]):
            return test_cmd

    # Sicherheitsnetz: Wenn zuletzt noch vp9_qsv gewählt wäre, aber libvpx-vp9 verfügbar ist,
    # zwingend auf libvpx-vp9 wechseln (viel stabiler/portable).
    if desired_core == "vp9" and last_cmd:
        try:
            # aktiven Encoder aus last_cmd lesen
            if "-c:v" in last_cmd:
                i = last_cmd.index("-c:v")
                enc_now = str(last_cmd[i + 1]).lower() if i + 1 < len(last_cmd) else ""
            else:
                enc_now = ""
        except Exception:
            enc_now = ""
        if enc_now == "vp9_qsv" and ("libvpx-vp9" in available):
            return _build_cmd_for_encoder(
                base_cmd, "libvpx-vp9", preset_name, container, vf_chain
            )

    if last_cmd:
        return last_cmd
    # Harte Rückfallebene: bei VP9 immer libvpx-vp9, sonst Default-Encoder
    fallback_enc = (
        "libvpx-vp9" if desired_core == "vp9" else encoder_for_codec(desired_core)
    )
    return _build_cmd_for_encoder(
        base_cmd, fallback_enc, preset_name, container, vf_chain
    )


# ============================================================
#     Player-/Container-Quirks (AVI/MPEG4, MOV/J2K, MKV/AV1)
# ============================================================
def apply_container_codec_quirks(
    final_cmd: List[str], container: str, codec_key: str
) -> List[str]:
    c = (container or "").lower()
    k = (codec_key or "").lower()

    # Ordnung der Filter/Flags vor dem Output sicherstellen
    ensure_pre_output_order(final_cmd)
    # ─────────────────────────────────────────────────────────────────────────────

    # AVI + MPEG4 → XVID FourCC + yuv420p
    if c == "avi" and k == "mpeg4":
        if "-tag:v" in final_cmd:
            i = final_cmd.index("-tag:v")
            final_cmd[i + 1] = "XVID"
        else:
            final_cmd += ["-tag:v", "XVID"]
        if "-bf" in final_cmd:
            i = final_cmd.index("-bf")
            final_cmd[i + 1] = "0"
        else:
            final_cmd += ["-bf", "0"]

    # MOV: gute Player-Defaults + Alpha/ProRes-Pflege + MP4/MOV Tags
    if c == "mov":
        if "-vsync" not in final_cmd:
            _insert_before_output(final_cmd, ["-vsync", "2"])
        if "-video_track_timescale" not in final_cmd:
            _insert_before_output(final_cmd, ["-video_track_timescale", "90000"])
        if "-movflags" not in final_cmd:
            _insert_before_output(final_cmd, ["-movflags", "+faststart"])

    # MP4/M4V: korrekte FourCC/brand Tags für H.264/HEVC/AV1
    if c in ("mp4", "m4v"):
        tag = _mov_mp4_vtag_for_codec(codec_key)

        # NEU: Stray FourCC-Werte (avc1/av01/hvc1/hev1/mp4v/vp09) entfernen,
        # wenn sie NICHT unmittelbar der Wert zu -tag:v/-vtag sind.
        stray_fourcc = {"avc1", "av01", "hvc1", "hev1", "mp4v", "vp09"}
        i = 0
        while i < len(final_cmd):
            t = str(final_cmd[i])
            if t in stray_fourcc:
                prev = str(final_cmd[i - 1]) if i > 0 else ""
                if prev not in ("-tag:v", "-vtag"):
                    final_cmd.pop(i)
                    continue  # an gleicher Stelle weitermachen
            i += 1

        # Bevorzugt -tag:v setzen (und -vtag Duplikate entfernen)
        _strip_options(final_cmd, {"-vtag"})
        if tag:
            _set_kv_arg(final_cmd, "-tag:v", tag)

        # Auch in -vf/-filter:v eingeschleuste FourCC-Fragmente entfernen
        def _purge_fourcc_in_vf(_cmd: List[str]) -> None:
            FOURCC = {"avc1", "av01", "hvc1", "hev1", "vp09", "mp4v"}
            for fl in ("-vf", "-filter:v"):
                try:
                    if fl in _cmd:
                        j = _cmd.index(fl)
                        if j + 1 < len(_cmd) and isinstance(_cmd[j + 1], str):
                            parts = [
                                p.strip() for p in _cmd[j + 1].split(",") if p.strip()
                            ]
                            parts = [p for p in parts if p.lower() not in FOURCC]
                            _cmd[j + 1] = ",".join(parts)
                except Exception:
                    pass

        _purge_fourcc_in_vf(final_cmd)

        # Faststart am Ende sicherstellen
        if "-movflags" not in final_cmd:
            final_cmd += ["-movflags", "+faststart"]

    # DNxHD/HR FourCC
    if c in ("mkv", "matroska") and k in ("dnxhd", "dnxhr"):
        if "-vtag" not in final_cmd and "-tag:v" not in final_cmd:
            final_cmd += ["-vtag", "AVdn"]

    # MagicYUV FourCC + Alpha-sicheres Pixelformat
    if c in ("mkv", "matroska") and k == "magicyuv":
        # FourCC erzwingen
        if "-vtag" in final_cmd:
            i = final_cmd.index("-vtag")
            final_cmd[i + 1] = "MAGY"
        elif "-tag:v" in final_cmd:
            i = final_cmd.index("-tag:v")
            final_cmd[i + 1] = "MAGY"
        else:
            final_cmd += ["-vtag", "MAGY"]

    # HAP: Wenn Ziel-Pixelformat Alpha trägt, Encoder-Subformat auf hap_alpha erzwingen
    if k == "hap":
        try:
            pf = (infer_target_pix_fmt_from_plan(final_cmd) or "").lower()
            has_alpha_pf = pf.startswith(("rgba", "bgra", "argb", "gbrap", "yuva"))
            if has_alpha_pf:
                # vorhandene -format Werte entfernen (hap/hap_q etc.), dann hap_alpha vor dem Output einfügen
                _strip_options(final_cmd, {"-format"})
                _insert_before_output(final_cmd, ["-format", "hap_alpha"])
        except Exception:
            pass

    try:
        ip = Path(final_cmd[final_cmd.index("-i") + 1])
        ensure_terminal_pix_fmt(
            final_cmd,
            container=container,
            codec_key=codec_key,
            input_path=ip,
            preset_name=None,
        )
    except Exception:
        pass

    # ── ProRes + Alpha ⇒ Profil 4/5 erzwingen ───────────────────────────
    try:
        if (normalize_codec_key(codec_key) or "") == "prores":
            pf = (infer_target_pix_fmt_from_plan(final_cmd) or "").lower()
            has_alpha_pf = pf.startswith(("yuva444", "gbrap")) or pf in {
                "rgba",
                "bgra",
                "argb",
            }
            if has_alpha_pf:
                # Encoder sicherstellen
                if _active_encoder(final_cmd, "prores_ks") != "prores_ks":
                    _set_kv_arg(final_cmd, "-c:v", "prores_ks")

                # Profil lesen
                cur = None
                if "-profile:v" in final_cmd:
                    i = final_cmd.index("-profile:v")
                    if i + 1 < len(final_cmd):
                        cur = str(final_cmd[i + 1]).strip()

                # Auf 4 setzen, außer Nutzer hat 5 (XQ) gewählt
                if cur not in {"4", "5"}:
                    _set_kv_arg(final_cmd, "-profile:v", "4")
    except Exception:
        pass

    # Nach allen Container-Quirks: Alpha-Formate in "Alpha-verbotenen" Zielen tilgen
    try:
        c = (container or "").lower()
        k = normalize_codec_key(codec_key) or ""
        if c in {"mp4", "m4v", "mov", "webm", "mpeg"} and k in {
            "h264",
            "hevc",
            "av1",
            "vp9",
            "mpeg4",
            "mpeg2video",
            "vp8",
        }:

            def _strip_alpha_format(flag: str) -> None:
                if flag in final_cmd:
                    j = final_cmd.index(flag)
                    if j + 1 < len(final_cmd):
                        s = str(final_cmd[j + 1])
                        parts = [
                            p
                            for p in s.split(",")
                            if not re.match(
                                r"^format=(?:rgba|bgra|argb|abgr|gbrap(?:10|12)?le|gbrap|yuva\d{3}p(?:10|12)?le)$",
                                p.strip(),
                                re.I,
                            )
                        ]
                        final_cmd[j + 1] = ",".join([p for p in parts if p])

            _strip_alpha_format("-vf")
            _strip_alpha_format("-filter:v")
    except Exception:
        pass

    # VP9: erzwinge nicht-alpha Terminal-Formate (säubere evtl. eingeschleuste RGBA/YUVA)
    try:
        if (normalize_codec_key(codec_key) or "") == "vp9":

            def _strip_alpha_format(flag: str) -> None:
                if flag in final_cmd:
                    j = final_cmd.index(flag)
                    if j + 1 < len(final_cmd):
                        s = str(final_cmd[j + 1])
                        parts = [
                            p
                            for p in s.split(",")
                            if not re.match(
                                r"^format=(?:rgba|bgra|argb|abgr|gbrap(?:10|12)?le|gbrap|yuva\d{3}p(?:10|12)?le)$",
                                p.strip(),
                                re.I,
                            )
                        ]
                        final_cmd[j + 1] = ",".join([p for p in parts if p])

            _strip_alpha_format("-vf")
            _strip_alpha_format("-filter:v")
    except Exception:
        pass

    ensure_pre_output_order(final_cmd)
    sanitize_movflags_inplace(final_cmd)
    return final_cmd


# ============================================================
#            Sonstige kleine Helfer
# ============================================================
def probe_min_source_geometry(
    files: Iterable[Union[str, Path]], ffprobe_bin: str = "ffprobe"
) -> Tuple[Optional[int], Optional[int]]:
    min_w: Optional[int] = None
    min_h: Optional[int] = None

    for f in files:
        p = str(Path(f))
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            p,
        ]
        try:
            out = subprocess.check_output(cmd, text=True)
            data = cast(Dict[str, Any], json.loads(out))
            streams = cast(List[Dict[str, Any]], data.get("streams") or [])
            st: Dict[str, Any] = streams[0] if streams else {}
            w = st.get("width")
            h = st.get("height")
            if isinstance(w, int) and isinstance(h, int):
                min_w = w if min_w is None else min(min_w, w)
                min_h = h if min_h is None else min(min_h, h)
        except Exception:
            continue

    return min_w, min_h


def filter_resolution_keys_for_source(
    resolution_defs: Mapping[str, Mapping[str, str]],
    keys: Iterable[str],
    min_w: Optional[int],
    min_h: Optional[int],
) -> List[str]:
    out: List[str] = []
    for k in keys:
        if k in ("original", "custom"):
            out.append(k)
            continue
        try:
            scale = resolution_defs[k]["scale"]
            w_str, h_str = scale.split(":")
            w, h = int(w_str), int(h_str)
        except Exception:
            # Auf Nummer sicher: Unbekannte/ungültige Keys dennoch durchlassen
            out.append(k)
            continue

        if min_w is None or min_h is None or (w <= min_w and h <= min_h):
            out.append(k)
    return out


def validate_custom_scale_leq(
    src_min_w: Optional[int],
    src_min_h: Optional[int],
    user_scale: str,
    require_even: bool = True,
    ignore_boundaries: Optional[bool] = False,
) -> Tuple[bool, str]:
    try:
        w_s, h_s = user_scale.split(":")
        w = int(w_s)
        h = int(h_s)
    except Exception:
        return False, _("enter_2integer_numbers")

    if w <= 0 or h == 0:
        return False, _("width_height_not_allowed")

    if h < 0 and h not in (-1, -2):
        return False, _("wrong_negative_height")

    if ignore_boundaries:
        if require_even:
            w = w - (w % 2)
            h = h - (h % 2)
        return True, f"{w}:{h}"

    def _too_big(num: int, lim: Optional[int]) -> bool:
        return lim is not None and num > lim

    if h > 0:
        if _too_big(w, src_min_w):
            return False, _("width_to_big").format(w_new=w, w_old=src_min_w)
        if _too_big(h, src_min_h):
            return False, _("height_to_big").format(h_new=h, h_old=src_min_h)
        if require_even and ((w % 2) or (h % 2)):
            return False, _("width_height_must_even")
    else:
        if _too_big(w, src_min_w):
            return False, _("width_to_big").format(w_new=w, w_old=src_min_w)
        if require_even and (w % 2):
            return False, _("width_height_must_even")

    return True, ""


def normalize_codec_key(value: Optional[str]) -> Optional[str]:
    return _normalize_codec_key(value)


# ============================================================
#          NEW: Wiederverwendbare FPS-Decimate-Logik
# ============================================================


def build_visual_chain_generic(
    *,
    meta: Dict[str, Optional[str]],
    src_w: Optional[int],
    src_h: Optional[int],
    container: str,
    codec_key: str,
    src_pix_fmt: Optional[str],
    scale_already_planned: bool = False,
    prefer_zscale: bool = True,
) -> Tuple[str, List[str]]:
    in_mat = _norm_matrix(meta.get("colorspace"))
    in_rng = _norm_range(meta.get("color_range"))
    out_mat = "bt709"
    out_rng = "limited"

    need_even = (not scale_already_planned) and bool(
        (src_w or 0) % 2 or (src_h or 0) % 2
    )

    vf_parts: List[str] = []
    extra: List[str] = []

    if prefer_zscale and has_filter("zscale"):
        if need_even:
            vf_parts.append(
                f"zscale=width=trunc(iw/2)*2:height=trunc(ih/2)*2:"
                f"matrixin={in_mat}:matrix={out_mat}:rangein={in_rng}:range={out_rng}:dither=none"
            )
        # KEIN format=… hier – das übernimmt jetzt die Pixfmt-API
    else:
        if need_even:
            vf_parts.append(
                "scale=trunc(iw/2)*2:trunc(ih/2)*2:"
                "flags=+full_chroma_int+bitexact"
                f":in_color_matrix={in_mat}:out_color_matrix={out_mat}"
                f":in_range={in_rng}:out_range={out_rng}"
            )
            extra += ["-sws_flags", "+full_chroma_int+bitexact"]

    return ",".join(vf_parts), extra


_CRF_OR_QSCALE = {
    "h264",
    "hevc",
    "av1",
    "vp9",
    "vp8",
    "mpeg4",
    "mpeg2video",
    "mpeg1video",
}


def _as_str(x: Any, default: str = "libx264") -> str:
    """
    Robust nach str konvertieren, ohne 'x' umzubinden (Pylance-freundlich).
    Handhabt: None, str, Sequenzen (nimmt das erste Element), Mappings (nimmt encoder/codec/key/name),
    Bytes, Fallback auf str(x).
    """
    if x is None:
        return default
    if isinstance(x, str):
        return x

    # Sequenzen: erstes Element rekursiv zu String machen
    if isinstance(x, (list, tuple, set)):
        seq: Iterable[Any] = cast(Iterable[Any], x)
        it = iter(seq)
        try:
            first = next(it)
        except StopIteration:
            return default
        return _as_str(first, default)

    # Mappings: bekannte Felder bevorzugen
    if isinstance(x, dict):
        m: Mapping[str, Any] = cast(Mapping[str, Any], x)
        for key in ("encoder", "codec", "key", "name"):
            v = m.get(key)
            if v is not None:
                return _as_str(v, default)
        return default

    # Bytes → UTF-8 (fehler tolerant)
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            return bytes(x).decode("utf-8", errors="ignore")
        except Exception:
            return default

    # Fallback
    try:
        return str(x)
    except Exception:
        return default


def pick_crf_codec_for_container(container: str, desired_key: str) -> str:
    c = (container or "").lower()
    want = normalize_codec_key(_as_str(desired_key)) or _as_str(desired_key)
    if want in _CRF_OR_QSCALE and container_allows_codec(c, want):
        return want
    order = tuple(defin.FALLBACK_BY_CONTAINER.get(c, ()))
    for key in order:
        k = normalize_codec_key(_as_str(key)) or _as_str(key)
        if k in _CRF_OR_QSCALE and container_allows_codec(c, k):
            return k
    k = normalize_codec_key(_as_str(suggest_codec_for_container(c))) or "h264"
    if k not in _CRF_OR_QSCALE:
        for k2 in (
            "h264",
            "hevc",
            "av1",
            "vp9",
            "mpeg4",
            "mpeg2video",
            "mpeg1video",
            "vp8",
        ):
            if container_allows_codec(c, k2):
                return k2
    return k


def inject_idr_at_t0(cmd: List[str], encoder_guess: Optional[str] = None) -> None:
    enc = (encoder_guess or _active_encoder(cmd, "libx264")).lower()
    fam = _encoder_family(enc)
    if "-force_key_frames" not in cmd:
        cmd += ["-force_key_frames", "expr:gte(t,0)"]
    if fam in ("h264", "hevc"):
        if "-g" not in cmd:
            cmd += ["-g", "12"]
        if "-keyint_min" not in cmd:
            cmd += ["-keyint_min", "1"]
        if fam == "h264" and "-x264-params" not in cmd:
            cmd += ["-x264-params", "scenecut=0"]
        if fam == "hevc" and "-x265-params" not in cmd:
            cmd += ["-x265-params", "scenecut=0"]


def _sanitize_audio_bitrate_inplace(cmd: List[str]) -> None:
    i = 0
    while i < len(cmd) - 1:
        if str(cmd[i]) == "-b:a":
            val = str(cmd[i + 1]).strip().lower()
            # erlaubt: 128k, 192k, 3m, 256000
            if not re.fullmatch(r"\d+(?:\.\d+)?(?:[kKmM])?", val):
                # ungültig (z.B. "pcm") → Flag + Wert entfernen
                del cmd[i : i + 2]
                continue
        i += 1


# === -movflags immer mit Wert zusammenhalten ==========================
def sanitize_movflags_inplace(cmd: List[str]) -> None:
    return _sanitize_movflags_inplace(cmd)


def _supports_odd_dimensions(codec_key: str, container: str) -> bool:
    k = (codec_key or "").lower()
    c = (container or "").lower()
    if c in ("mkv", "matroska") and k in ("ffv1", "huffyuv", "utvideo", "rawvideo"):
        return True
    return False


def postprocess_cmd_all_presets(cmd: List[str], plan) -> List[str]:
    try:
        spec = _get_preset_spec(defin.CONVERT_PRESET, plan.preset_name)
        if bool(spec.get("lossless")):
            # Entscheidung über die Pixfmt-API (auch für Lossless)
            ensure_terminal_pix_fmt(
                cmd,
                container=plan.target_container,
                codec_key=plan.codec_key,
                input_path=plan.input_path,
                preset_name=plan.preset_name,
            )
            _enforce_prores_alpha_profile_inplace(cmd, plan)
            _dedupe_color_tags(cmd)
            return cmd
    except Exception:
        pass

    enc_now = _active_encoder(
        cmd, encoder_for_codec(normalize_codec_key(plan.codec_key) or plan.codec_key)
    )
    if enc_now == "hevc_nvenc" and not _nvenc_usable():
        try:
            i = cmd.index("-c:v")
            cmd[i + 1] = "libx265"
        except ValueError:
            cmd += ["-c:v", "libx265"]
        _strip_nvenc_private_opts(cmd)

    _ensure_quality_defaults(cmd, plan.target_container, plan.codec_key)

    ensure_terminal_pix_fmt(
        cmd,
        container=plan.target_container,
        codec_key=plan.codec_key,
        input_path=plan.input_path,
        preset_name=plan.preset_name,
    )

    _enforce_prores_alpha_profile_inplace(cmd, plan)

    _dedupe_color_tags(cmd)

    try:
        if "-vf" in cmd and "fps=" in cmd[cmd.index("-vf") + 1] and "-vsync" not in cmd:
            cmd += ["-vsync", "2"]
    except Exception:
        pass

    try:
        if getattr(plan, "force_key_at_start", False):
            inject_idr_at_t0(
                cmd,
                _active_encoder(
                    cmd,
                    encoder_for_codec(
                        normalize_codec_key(plan.codec_key) or plan.codec_key
                    ),
                ),
            )
    except Exception:
        pass

    cmd = apply_container_codec_quirks(cmd, plan.target_container, plan.codec_key)
    ensure_pre_output_order(cmd)
    sanitize_movflags_inplace(cmd)
    return cmd


def _enforce_prores_alpha_profile_inplace(cmd: List[str], plan) -> None:
    """
    Wenn prores + Alpha-Zielpixelformat erkannt wird, Profil 4 (4444) und prores_ks erzwingen.
    Verhindert, dass ffmpeg yuva444p10le -> yuv444p10le 'herabstuft'.
    """
    try:
        k = normalize_codec_key(getattr(plan, "codec_key", "") or "")
    except Exception:
        k = ""
    if k != "prores":
        return

    pf = (infer_target_pix_fmt_from_plan(cmd) or "").lower()
    has_alpha_pf = pf.startswith(("yuva444", "gbrap")) or pf in {"rgba", "bgra", "argb"}
    if not has_alpha_pf:
        return

    # Sicherstellen, dass prores_ks genutzt wird
    if _active_encoder(cmd, "prores_ks") != "prores_ks":
        _set_kv_arg(cmd, "-c:v", "prores_ks")

    # Profil prüfen/setzen: 4 (4444) oder 5 (4444 XQ) – wenn noch nicht passend -> auf 4 setzen
    current = None
    try:
        if "-profile:v" in cmd:
            i = cmd.index("-profile:v")
            if i + 1 < len(cmd):
                current = str(cmd[i + 1]).strip()
    except Exception:
        current = None

    if current not in {"4", "5"}:
        _set_kv_arg(cmd, "-profile:v", "4")


def _parse_int_pair(scale: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        a, b = scale.split(":")
        return int(a), int(b)
    except Exception:
        return None, None


def _norm_fps_str_user(v: str) -> str:
    v = (v or "").strip()
    if v in {"23.976", "29.97", "59.94"}:
        return {"23.976": "24000/1001", "29.97": "30000/1001", "59.94": "60000/1001"}[v]
    if re.fullmatch(r"\d+/\d+", v):
        return v
    try:
        f = Fraction(float(v)).limit_denominator(1001)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        # Fallback: 25 fps, falls Eingabe völlig schief ist
        return "25"


def _mov_mp4_vtag_for_codec(codec_key: str) -> Optional[str]:
    k = normalize_codec_key(codec_key or "") or ""
    return {
        "h264": "avc1",
        "hevc": "hvc1",
        "av1": "av01",
        "mpeg4": "mp4v",
    }.get(k)


def _should_try_stream_copy(
    *,
    target_container: str,
    desired_codec_key: Optional[str],
    input_path: Path,
    req_scale: Optional[str],
    user_fps_rational: Optional[str],
    preset_max_fps: Optional[int],
) -> Tuple[bool, str, str]:
    src_codec = (probe_video_codec(input_path) or "").lower()
    src_key = normalize_codec_key(src_codec) or ""
    want_key = normalize_codec_key(desired_codec_key or "") or src_key

    no_processing = (
        (not req_scale) and (not user_fps_rational) and (preset_max_fps in (None, 0))
    )
    if not no_processing:
        return (False, src_key, want_key)

    if not src_key or not container_allows_codec(target_container, src_key):
        return (False, src_key, want_key)

    if want_key not in {"copy", src_key}:
        return (False, src_key, want_key)

    return (True, src_key, want_key)


def find_lossless_combinations(
    *,
    input_path: Path,
    container_to_codecmap: Dict[str, CodecMap],
    req_scale: Optional[str] = None,
    user_fps_rational: Optional[str] = None,
    preset_max_fps: Optional[int] = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    sanity_probe_copy: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Liefert je Container, welche Codec-Keys als 'copy' bzw. 'strict' lossless
    möglich wären. Die Parametertypen sind präzise (CodecMap), sodass
    'emap', 'codec_key' und 'pref_enc' nicht mehr 'Unknown' sind.
    """

    def _cmd_from_plan(plan: Any) -> Optional[List[str]]:
        if hasattr(plan, "final_cmd_without_output"):
            return list(getattr(plan, "final_cmd_without_output"))
        if isinstance(plan, (list, tuple)):
            return list(plan)
        return None

    def _has_scale_or_fps_filters(tokens: List[str]) -> bool:
        toks = [str(t).lower() for t in tokens]

        def _arg_after(flag: str) -> str:
            if flag in toks:
                i = toks.index(flag)
                if i + 1 < len(toks):
                    return toks[i + 1]
            return ""

        vf = _arg_after("-vf")
        filtv = _arg_after("-filter:v")
        joined = " ".join(toks)
        return (
            ("scale=" in vf)
            or ("fps=" in vf)
            or ("scale=" in filtv)
            or ("fps=" in filtv)
            or (" -r " in f" {joined} ")
        )

    def _video_codec_from_tokens(tokens: List[str]) -> Optional[str]:
        toks = [str(t).lower() for t in tokens]
        for flag in ("-c:v", "-codec:v", "-vcodec"):
            if flag in toks:
                i = toks.index(flag)
                if i + 1 < len(toks):
                    return toks[i + 1]
        return None

    def _has_any_video_filter(cmd_tokens: List[str]) -> bool:
        toks = [str(t).lower() for t in cmd_tokens]

        def val(flag: str) -> str:
            return (
                toks[toks.index(flag) + 1]
                if flag in toks and toks.index(flag) + 1 < len(toks)
                else ""
            )

        vf = val("-vf") + "," + val("-filter:v")
        joined = " ".join(toks)
        return (
            any(
                x in vf
                for x in ("scale=", "fps=", "format=", "zscale=", "setdar", "setsar")
            )
            or " -r " in joined
        )

    def _is_stream_copy_tokens(tokens: List[str]) -> bool:
        toks = [str(t).lower() for t in tokens]
        if "-c:v" in toks:
            i = toks.index("-c:v")
            if i + 1 < len(toks) and toks[i + 1] == "copy":
                return not _has_any_video_filter(tokens)
        return False

    LOSSLESS_CODECS: Set[str] = {
        "ffv1",
        "huffyuv",
        "utvideo",
        "rawvideo",
        "png",
        "magicyuv",
        "ljpeg",
        "zlib",
        "zmbv",
        "qtrle",
        "jpeg2000",
    }

    def _is_strict_lossless_tokens(tokens: List[str]) -> bool:
        if _has_scale_or_fps_filters(tokens):
            return False
        c = (_video_codec_from_tokens(tokens) or "").lower()
        if c == "copy":
            return False
        if c in LOSSLESS_CODECS:
            return True
        if c in ("libx264", "h264", "libx265", "hevc"):
            s = " ".join(str(t).lower() for t in tokens)
            return ("lossless=1" in s) or ("-qp 0" in s) or ("-crf 0" in s)
        return False

    changes = bool(
        req_scale
        or user_fps_rational
        or (preset_max_fps is not None and preset_max_fps > 0)
    )
    result: Dict[str, Dict[str, List[str]]] = {}

    for cont, emap in container_to_codecmap.items():
        # Lokale Typinfo (explizit), damit Pylance Schleifenvariablen typisiert
        emap = cast(CodecMap, emap)
        copy_list: List[str] = []
        strict_list: List[str] = []

        for codec_key, pref_enc in emap.items():
            want = normalize_codec_key(codec_key) or ""
            if not want or not container_allows_codec(cont, want):
                continue

            try:
                plan = build_transcode_plan(
                    input_path=input_path,
                    target_container=cont,
                    preset_name="lossless",
                    codec_key=want,
                    preferred_encoder=(pref_enc or None),
                    req_scale=None,
                    src_w=None,
                    src_h=None,
                    src_fps=None,
                    user_fps_rational=None,
                    preset_max_fps=None,
                )
            except Exception:
                continue

            tokens = _cmd_from_plan(plan)
            if not tokens:
                continue

            if not changes and _is_stream_copy_tokens(tokens):
                copy_list.append(codec_key)
                continue

            if _is_strict_lossless_tokens(tokens):
                strict_list.append(codec_key)

        result[cont] = {"copy": copy_list, "strict": strict_list}

    return result


def dedupe_vf_inplace(cmd: List[str]) -> None:
    return _dedupe_vf_inplace(cmd)


@dataclass
class CodecChoiceValidation:
    ok: bool
    errors: list[str]
    allowed_codecs_for_format: list[str]  # laut Formatspezifikation
    container_allowed: bool  # ob der Container den Codec überhaupt zulässt


# Eingabe-Synonyme (inkl. Tippfehler "orginal")
_COPY_ALIASES = {"copy", "streamcopy", "stream-copy", "bitstream-copy"}
_KEEP_ALIASES = {
    "keep",
    "original",
    "orginal",
    "same",
    "source",
    "orig",
    "same-as-source",
}


def _norm_choice(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _normalize_container_choice(raw: Optional[str]) -> str:
    v = _norm_choice(raw)
    # Format "copy" interpretieren wir wie "keep" (Container beibehalten/remux)
    if not v or v in _KEEP_ALIASES or v in _COPY_ALIASES:
        return "keep"
    return v


def _normalize_codec_choice(raw: Optional[str]) -> str:
    v = _norm_choice(raw)
    if v in _KEEP_ALIASES:
        return "keep"
    if v in _COPY_ALIASES:
        return "copy"
    return v


def _guess_container_from_path(p: Path) -> str:
    ext = (p.suffix or "").lower().lstrip(".")
    mapping = {
        "m4v": "mp4",
        "mp4": "mp4",
        "mov": "mov",
        "webm": "webm",
        "mkv": "mkv",
        "avi": "avi",
        "mpeg": "mpeg",
        "mpg": "mpeg",
    }
    return mapping.get(ext, ext or "mkv")  # mkv als defensiver Default


def _allowed_codecs_from_format_spec(
    convert_format_descriptions: Mapping[str, Mapping[str, Any]], fmt: str
) -> list[str]:
    try:
        return [
            normalize_codec_key(x) or str(x)
            for x in _extract_codec_keys_for_format(convert_format_descriptions, fmt)
        ]
    except Exception:
        return []


def _validate_format_codec_choice(
    *,
    target_container: str,
    codec_key: str,
    convert_format_descriptions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> CodecChoiceValidation:
    """
    Validiert eine (Format/Container, Codec)-Kombination.
    - Bei 'copy'/'keep' (ohne bekannten Source-Codec) wird die harte Validierung übersprungen.
    """
    errors: list[str] = []
    c = (target_container or "").lower()
    k = normalize_codec_key(codec_key or "") or ""

    # Wenn kein konkreter Codec vorliegt (copy/keep ohne Source-Codec),
    # überspringen wir die harte Prüfung – die Kompatibilität wird später
    # bei der tatsächlichen Remux/Encode-Phase entschieden.
    if not k:
        return CodecChoiceValidation(
            ok=True, errors=[], allowed_codecs_for_format=[], container_allowed=True
        )

    container_ok = container_allows_codec(c, k)
    if not container_ok:
        errors.append(
            _("validate_container_disallows_codec").format(container=c, codec=k)
        )

    allowed_list: list[str] = []
    if convert_format_descriptions:
        allowed_list = list(
            dict.fromkeys(
                _allowed_codecs_from_format_spec(convert_format_descriptions, c)
            )
        )
        if allowed_list and (k not in allowed_list):
            errors.append(
                _("validate_codec_not_defined_for_format").format(
                    codec=k, container=c, allowed=", ".join(allowed_list)
                )
            )

    return CodecChoiceValidation(
        ok=(not errors),
        errors=errors,
        allowed_codecs_for_format=allowed_list,
        container_allowed=container_ok,
    )


def plan_warnings_and_errors_for_choice(
    *,
    input_path: Path,
    target_container: str,
    codec_key: str,
    convert_format_descriptions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    encoder_name: Optional[str] = None,
    src_codec: Optional[str] = None,  # NEU: falls bekannt, für copy/keep-Validierung
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """
    Komfort-Wrapper: gibt (warnings, errors, suggestions) zurück.
    - Erkenne 'keep'/'original'/'orginal'/'copy'
    - Warnung bei Alpha-Verlust
    - Fehler bei ungültigen Format/Codec-Kombinationen
    """
    warnings: list[str] = []
    errors: list[str] = []
    suggestions: list[tuple[str, str]] = []

    norm_fmt = _normalize_container_choice(target_container)
    norm_codec = _normalize_codec_choice(codec_key)

    # Effektiver Container: bei 'keep' vom Input ableiten
    effective_container = (
        _guess_container_from_path(input_path) if norm_fmt == "keep" else norm_fmt
    )

    # Effektiver Codec:
    # - copy  => Streamcopy → Source-Codec falls bekannt, sonst leer (Validierung wird übersprungen)
    # - keep  => Re-Encode mit Source-Codec (falls bekannt); sonst leer (späterer Fallback)
    if norm_codec == "copy":
        effective_codec = normalize_codec_key(src_codec or "") or ""
        stream_copy = True
    elif norm_codec == "keep":
        effective_codec = normalize_codec_key(src_codec or "") or ""
        stream_copy = False  # "keep" heißt re-encoden mit gleichem Codec
    else:
        effective_codec = normalize_codec_key(norm_codec or "") or ""
        stream_copy = False

    # (1) Container/Codec validieren – nur, wenn beides konkret ist
    if effective_container and effective_codec:
        v = _validate_format_codec_choice(
            target_container=effective_container,
            codec_key=effective_codec,
            convert_format_descriptions=convert_format_descriptions,
        )
        if not v.ok:
            errors.extend(v.errors)
    else:
        # copy/keep ohne bekannten Source-Codec → harte Vorprüfung nicht möglich
        warnings.append(_("prevalidate_skipped_copy_keep"))

    # (2) Alpha-Erhaltung prüfen
    a = _assess_alpha_preservation(
        input_path=input_path,
        target_container=effective_container,
        codec_key=effective_codec,
        encoder_name=encoder_name,
        stream_copy=stream_copy,
    )
    if a.will_lose_alpha and a.reason:
        warnings.append(_("alpha_loss_prefix").format(reason=a.reason))
        suggestions.extend(a.suggestions)

    return (warnings, errors, suggestions)


def _assess_strict_lossless_feasibility(
    *,
    input_path: Path,
    target_container: str,
    desired_codec_key: str,
    preferred_encoder: Optional[str],
    req_scale: Optional[str],
    src_w: Optional[int],
    src_h: Optional[int],
    src_fps: Optional[float],
    user_fps_rational: Optional[str],
    preset_max_fps: Optional[int],
    ffprobe_bin: str = "ffprobe",
) -> LosslessAssessment:
    want = _normalize_codec_key(desired_codec_key or "") or ""

    must_scale = bool(req_scale)
    must_change_fps = bool(user_fps_rational) or (
        preset_max_fps is not None and preset_max_fps > 0
    )

    NON_STRICT = {
        "mpeg4",
        "mpeg2video",
        "mpeg1video",
        "vp8",
        "theora",
        "prores",
        "dnxhd",
        "mjpeg",
    }
    if want in NON_STRICT:
        msg = (
            "VP8 unterstützt keinen mathematisch verlustfreien Modus (nur near-lossless)."
            if want == "vp8"
            else "Gewählter Codec unterstützt kein mathematisch verlustfreies Encoding."
        )
        reasons = [msg]
        return LosslessAssessment(
            status="visual",
            reasons=reasons,
            recommended_encoder=None,
            strict_target_pix_fmt=None,
            must_reformat_pix_fmt=False,
            must_scale=must_scale,
            must_change_fps=must_change_fps,
        )

    _w, _h, src_pix_fmt = _ffprobe_geometry(input_path, ffprobe_bin=ffprobe_bin)
    src_w = src_w or _w
    src_h = src_h or _h
    src_codec = (_probe_video_codec(input_path) or "").lower()
    cont = (target_container or "").lower()
    want = _normalize_codec_key(desired_codec_key or "") or ""
    reasons: List[str] = []

    must_scale = bool(req_scale)
    must_change_fps = bool(user_fps_rational) or (
        preset_max_fps is not None and preset_max_fps > 0
    )
    if must_scale:
        reasons.append("Skalierung angefordert → nicht bitgenau.")
    if must_change_fps:
        reasons.append("FPS-Änderung angefordert → nicht bitgenau.")

    keep_pix, _h264_prof = _should_preserve_src_pix_fmt_in_lossless(
        cont, encoder_for_codec(want or src_codec), src_pix_fmt
    )
    must_reformat_pix = not keep_pix
    if must_reformat_pix and src_pix_fmt:
        reasons.append(
            f"Container/Encoder erfordern Pixelformat-Wechsel (Quelle: {src_pix_fmt})."
        )

    odd_dims = bool(src_w and src_h and ((src_w % 2) or (src_h % 2)))
    if odd_dims and not _supports_odd_dimensions(want or src_codec, cont):
        if src_w is not None and src_h is not None:
            reasons.append(
                f"Quelle hat ungerade Abmessungen ({src_w}×{src_h}), Encoder/Container verlangen even → Reformat nötig."
            )

    chosen_enc = preferred_encoder or encoder_for_codec(want or src_codec)
    if src_pix_fmt and not _encoder_supports_pix_fmt(chosen_enc, src_pix_fmt):
        reasons.append(
            f"Encoder {chosen_enc} unterstützt {src_pix_fmt} nicht → Reformat nötig."
        )
        must_reformat_pix = True

    if (
        not must_scale
        and not must_change_fps
        and not must_reformat_pix
        and not (odd_dims and not _supports_odd_dimensions(want or src_codec, cont))
    ):
        choice = _choose_strict_lossless_encoder(
            target_container=target_container,
            src_codec=src_codec,
            src_pix_fmt=src_pix_fmt,
            desired_codec_key=want,
        )
        return LosslessAssessment(
            status="strict",
            reasons=reasons,
            recommended_encoder=choice.encoder,
            strict_target_pix_fmt=src_pix_fmt,
            must_reformat_pix_fmt=False,
            must_scale=False,
            must_change_fps=False,
        )

    if want == "vp8":
        reasons.append(
            "VP8 unterstützt keinen mathematisch verlustfreien Modus (nur near-lossless)."
        )

    if want in {"mpeg4", "mpeg2video", "mpeg1video"}:
        reasons.append(
            "Gewählter Codec unterstützt kein mathematisch verlustfreies Encoding."
        )
        return LosslessAssessment(
            status="visual",
            reasons=reasons,
            recommended_encoder=None,
            strict_target_pix_fmt=None,
            must_reformat_pix_fmt=False,
            must_scale=bool(req_scale),
            must_change_fps=bool(user_fps_rational)
            or (preset_max_fps not in (None, 0)),
        )

    return LosslessAssessment(
        status=(
            "visual"
            if (
                want
                in {"h264", "hevc", "av1", "vp9", "vp8", "prores", "dnxhd", "dnxhr"}
                or cont in {"mp4", "mov", "webm"}
            )
            else "none"
        ),
        reasons=reasons,
        recommended_encoder=None,
        strict_target_pix_fmt=None,
        must_reformat_pix_fmt=must_reformat_pix,
        must_scale=must_scale,
        must_change_fps=must_change_fps,
    )


# ============================================================
#          Plan-API für wiederverwendbare Pipelines
# ============================================================
@dataclass
class TranscodePlan:
    """Vollständiger FFmpeg-Plan ohne Ausgabepfad."""

    input_path: Path
    target_container: str
    preset_name: str
    codec_key: str
    safe_scale: Optional[str]
    vf_chain: Optional[str]
    src_w: Optional[int]
    src_h: Optional[int]
    src_fps: Optional[float]
    target_w: int
    target_h: int
    final_cmd_without_output: List[str]
    force_pix_fmt: Optional[str] = None
    skip_alignment_scale: bool = False
    force_key_at_start: bool = False


# ---------- Stream-Copy-Plan (Remux) ----------
def _build_stream_copy_plan(
    *,
    input_path: Path,
    target_container: str,
    preset_name: str,
    desired_codec_key: Optional[str],
    src_w: Optional[int],
    src_h: Optional[int],
    src_fps: Optional[float],
    req_scale: Optional[str],
    user_fps_rational: Optional[str],
    preset_max_fps: Optional[int],
    force_key_at_start: bool,
) -> Optional[TranscodePlan]:
    ok, src_key, want_key = _should_try_stream_copy(
        target_container=target_container,
        desired_codec_key=desired_codec_key,
        input_path=input_path,
        req_scale=req_scale,
        user_fps_rational=user_fps_rational,
        preset_max_fps=preset_max_fps,
    )
    if not ok:
        return None

    base_cmd: List[str] = [
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
    base_cmd += build_stream_mapping_args(target_container, input_path=input_path)
    base_cmd += ["-map_metadata", "0"]
    # Video copy
    base_cmd += ["-c:v", "copy"]

    # Audio: container-gerecht
    if target_container.lower() == "avi":
        base_cmd += _avi_audio_args_for_source(input_path, preset_name)
    elif target_container.lower() == "webm":
        base_cmd += _webm_audio_args_for_source(input_path, preset_name)
    else:
        base_cmd += build_audio_args(target_container, preset_name, input_path)

    _sanitize_audio_bitrate_inplace(base_cmd)

    # Container-Tags (FourCC)
    vtag = _mov_mp4_vtag_for_codec(src_key)
    if vtag and target_container.lower() in ("mp4", "m4v", "mov"):
        base_cmd += ["-tag:v", vtag]
        if target_container.lower() in ("mp4", "m4v"):
            base_cmd += ["-movflags", "+faststart"]

    # Geometrie aus Quelle holen
    if src_w is None or src_h is None:
        _w, _h, _ = ffprobe_geometry(input_path)
        src_w = _w or 0
        src_h = _h or 0

    plan = TranscodePlan(
        input_path=input_path,
        target_container=target_container,
        preset_name=preset_name,
        codec_key=src_key,
        safe_scale=None,
        vf_chain=None,
        src_w=src_w,
        src_h=src_h,
        src_fps=src_fps,
        target_w=int(src_w or 0),
        target_h=int(src_h or 0),
        final_cmd_without_output=base_cmd,
        force_pix_fmt=None,
        skip_alignment_scale=True,
        force_key_at_start=bool(force_key_at_start),
    )
    return plan


# ---------- Hauptplan ----------
def _build_user_vf_chain(
    *,
    container: str,
    src_w: Optional[int],
    src_h: Optional[int],
    src_fps: Optional[float],
    req_scale: Optional[str],
    user_fps_rational: Optional[str],
    preset_max_fps: Optional[int],
    chosen_encoder: Optional[str] = None,
    preserve_ar: bool = True,
) -> Tuple[Optional[str], List[str], Optional[str], int, int]:
    safe_scale = _compute_safe_scale(
        int(src_w or 0) or 0,
        int(src_h or 0) or 0,
        req_scale,
        chosen_encoder or "",
        allow_upscale=True,
        preserve_ar=preserve_ar,
    )

    if safe_scale:
        tw, th = _parse_int_pair(safe_scale)
        target_w = int(tw or src_w or 0)
        target_h = int(th or src_h or 0)
    else:
        target_w = int(src_w or 0)
        target_h = int(src_h or 0)

    parts: List[str] = []
    if safe_scale:
        parts.append(f"scale={safe_scale}")
        parts.append("setsar=1")
        if target_w and target_h:
            parts.append(f"setdar={target_w}/{target_h}")
    else:
        parts.append("setsar=1")

    # WICHTIG: KEIN format=… mehr hier → die Pixfmt-API entscheidet später
    enforce_cfr_value: Optional[str] = None
    if user_fps_rational:
        enforce_cfr_value = _norm_fps_str_user(user_fps_rational)
    elif (
        (preset_max_fps is not None)
        and src_fps
        and (src_fps > float(preset_max_fps) + 0.05)
    ):
        enforce_cfr_value = str(int(preset_max_fps))

    extra_mux: List[str] = []
    if enforce_cfr_value:
        parts.append(f"fps={enforce_cfr_value}")
        extra_mux += ["-vsync", "2"]
        if container.lower() == "mp4":
            extra_mux += ["-video_track_timescale", "90000", "-movflags", "+faststart"]

    vf_chain = ",".join(parts) if parts else None
    return vf_chain, extra_mux, safe_scale, target_w, target_h


def build_transcode_plan(
    *,
    input_path: Path,
    target_container: str,
    preset_name: str,
    codec_key: str,
    preferred_encoder: Optional[str],
    req_scale: Optional[str],
    src_w: Optional[int],
    src_h: Optional[int],
    src_fps: Optional[float],
    user_fps_rational: Optional[str],
    preset_max_fps: Optional[int],
    force_key_at_start: Optional[bool] = False,
    preserve_ar: bool = True,
) -> TranscodePlan:
    spec = _get_preset_spec(defin.CONVERT_PRESET, preset_name)
    is_lossless = bool(spec.get("lossless"))

    if is_lossless and target_container.lower() == "flv":
        codec_key = "screenvideo"
        preferred_encoder = "flashsv2"

    req_scale = _preset_or_arg(spec, "scale", req_scale)
    preset_max_fps = _preset_or_arg(spec, "max_fps", preset_max_fps)

    force_mezzanine_dnx = _wants_mezzanine_dnx(target_container, codec_key)

    if src_w is None or src_h is None:
        _w, _h, _pix = ffprobe_geometry(input_path)
        src_w = src_w or _w
        src_h = src_h or _h
        src_pix_fmt = _pix
    else:
        _, _, src_pix_fmt = ffprobe_geometry(input_path)

    # 0) SMART COPY (vor allem anderen), wenn kein Keyframe-Zwang am Start
    sc_plan = None
    if not force_key_at_start:
        sc_plan = _build_stream_copy_plan(
            input_path=input_path,
            target_container=target_container,
            preset_name=preset_name,
            desired_codec_key=codec_key,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            req_scale=req_scale,
            user_fps_rational=user_fps_rational,
            preset_max_fps=preset_max_fps,
            force_key_at_start=bool(force_key_at_start),
        )
    if sc_plan is not None:
        return sc_plan

    # 1) Basiscmd + Mapping + Audio
    base_cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
    ]
    if target_container.lower() == "webm":
        base_cmd += ["-fflags", "+genpts"]
    base_cmd += ["-i", str(input_path)]
    base_cmd += build_stream_mapping_args(target_container, input_path=input_path)
    base_cmd += ["-map_metadata", "0"]
    lc = target_container.lower()
    if lc == "avi":
        base_cmd += _avi_audio_args_for_source(input_path, preset_name)
    elif lc == "webm":
        base_cmd += _webm_audio_args_for_source(input_path, preset_name)
    else:
        base_cmd += build_audio_args(target_container, preset_name, input_path)

    _sanitize_audio_bitrate_inplace(base_cmd)

    # 2) User-Filters (Scale/FPS)
    user_vf_chain, user_mux_extras, user_safe_scale, user_tgt_w, user_tgt_h = (
        _build_user_vf_chain(
            container=target_container,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            req_scale=req_scale,
            user_fps_rational=user_fps_rational,
            preset_max_fps=preset_max_fps,
            chosen_encoder=preferred_encoder,
            preserve_ar=preserve_ar,
        )
    )

    # H.263 (FLV)
    if normalize_codec_key(codec_key) == "h263" and target_container.lower() == "flv":
        vf_frag, pix = _enforce_h263_size(target_container, "h263", src_w, src_h)
        if vf_frag:
            if user_vf_chain:
                user_vf_chain = _strip_filter_from_vf(user_vf_chain, "scale")
                user_vf_chain = _strip_filter_from_vf(user_vf_chain, "setsar")
                user_vf_chain = _strip_filter_from_vf(user_vf_chain, "format")
            user_vf_chain = vf_join(vf_frag, user_vf_chain, "format=yuv420p")
            m = re.search(r"scale=(\d+):(\d+)", vf_frag)
            if m:
                user_safe_scale = f"{m.group(1)}:{m.group(2)}"
                user_tgt_w, user_tgt_h = int(m.group(1)), int(m.group(2))

    if user_mux_extras:
        base_cmd += user_mux_extras

    chosen_codec_key: str = codec_key
    chosen_preferred_encoder: Optional[str] = preferred_encoder
    force_pix_fmt: Optional[str] = None
    skip_alignment_scale = False

    # 3) Container-Fallback OHNE Cross-Codec
    if normalize_codec_key(chosen_codec_key) != "copy":
        resolved_key, enc_hint = resolve_codec_key_with_container_fallback(
            target_container,
            chosen_codec_key,
            ffmpeg_bin="ffmpeg",
            allow_cross_codec_fallback=False,
        )
        if resolved_key != chosen_codec_key:
            if chosen_preferred_encoder and (
                _encoder_family(chosen_preferred_encoder)
                != (normalize_codec_key(resolved_key) or "")
            ):
                chosen_preferred_encoder = None
            chosen_codec_key = resolved_key

    # 3.5) DNx in MOV: Mezzanine fast-path
    if force_mezzanine_dnx:
        final_cmd = base_cmd[:]
        tw, th = (src_w or 0, src_h or 0)
        if user_vf_chain:
            final_cmd += ["-vf", user_vf_chain]
            tw = int(user_tgt_w or tw)
            th = int(user_tgt_h or th)
        chosen_codec_key = "dnxhd"
        final_cmd = _apply_dnx_rules_if_needed(
            final_cmd,
            codec_key=chosen_codec_key,
            container=target_container,
            target_w=tw,
            target_h=th,
            src_fps=src_fps,
            preset_name=preset_name,
        )
        final_cmd = apply_container_codec_quirks(
            final_cmd, container=target_container, codec_key=chosen_codec_key
        )

        meta = probe_color_metadata(input_path)
        final_cmd += _color_metadata_args(meta)
        dedupe_vf_inplace(final_cmd)

        sanitize_movflags_inplace(final_cmd)
        return TranscodePlan(
            input_path=input_path,
            target_container=target_container,
            preset_name=preset_name,
            codec_key=chosen_codec_key,
            safe_scale=user_safe_scale,
            vf_chain=user_vf_chain,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            target_w=tw,
            target_h=th,
            final_cmd_without_output=final_cmd,
            force_pix_fmt=None,
            skip_alignment_scale=True,
            force_key_at_start=bool(force_key_at_start),
        )

    # 4) Lossless-Machbarkeit evaluieren
    ll = _assess_strict_lossless_feasibility(
        input_path=input_path,
        target_container=target_container,
        desired_codec_key=chosen_codec_key,
        preferred_encoder=preferred_encoder,
        req_scale=req_scale,
        src_w=src_w,
        src_h=src_h,
        src_fps=src_fps,
        user_fps_rational=user_fps_rational,
        preset_max_fps=preset_max_fps,
    )

    # ===================== LOSSLESS-PFAD =====================
    if is_lossless:
        final_cmd = base_cmd[:]
        skip_alignment_scale = True
        safe_scale = None
        vf_chain = None

        # MP4/M4V + H.264 → near-lossless (kein qp=0)
        if (
            target_container.lower() in ("mp4", "m4v")
            and normalize_codec_key(codec_key) == "h264"
        ):
            meta = probe_color_metadata(input_path)
            vff, extra = _build_h264_near_lossless_chain(meta, src_w, src_h)
            if extra:
                final_cmd += extra
            final_cmd += _h264_near_lossless_args_mp4()
            if vff:
                final_cmd += ["-vf", vff]
            final_cmd = _apply_h264_vui_flags_if_needed(final_cmd, meta, src_pix_fmt)

            dedupe_vf_inplace(final_cmd)

            tw, th = (src_w or 0, src_h or 0)
            sanitize_movflags_inplace(final_cmd)
            return TranscodePlan(
                input_path=input_path,
                target_container=target_container,
                preset_name=preset_name,
                codec_key="h264",
                safe_scale=None,
                vf_chain=vff,
                src_w=src_w,
                src_h=src_h,
                src_fps=src_fps,
                target_w=tw,
                target_h=th,
                final_cmd_without_output=final_cmd,
                force_pix_fmt=None,
                skip_alignment_scale=True,
                force_key_at_start=bool(force_key_at_start),
            )

        # MKV/MOV + H.264 → ebenfalls near-lossless (kein qp=0, keine I-only-GOP)
        if (
            target_container.lower() in ("mkv", "matroska", "mov")
            and normalize_codec_key(codec_key) == "h264"
        ):
            meta = probe_color_metadata(input_path)
            vff, extra = _build_h264_near_lossless_chain(meta, src_w, src_h)
            if extra:
                final_cmd += extra
            final_cmd += _h264_near_lossless_args_mkv()
            if vff:
                final_cmd += ["-vf", vff]

            final_cmd = _apply_h264_vui_flags_if_needed(final_cmd, meta, src_pix_fmt)

            _strip_options(
                final_cmd, {"-pix_fmt"}
            )  # keine -pix_fmt, wir setzen format=... in -vf
            dedupe_vf_inplace(final_cmd)

            tw, th = (src_w or 0, src_h or 0)
            sanitize_movflags_inplace(final_cmd)
            return TranscodePlan(
                input_path=input_path,
                target_container=target_container,
                preset_name=preset_name,
                codec_key="h264",
                safe_scale=None,
                vf_chain=vff,
                src_w=src_w,
                src_h=src_h,
                src_fps=src_fps,
                target_w=tw,
                target_h=th,
                final_cmd_without_output=final_cmd,
                force_pix_fmt=None,
                skip_alignment_scale=True,
                force_key_at_start=bool(force_key_at_start),
            )

        # MP4/M4V + MPEG-4 → near-lossless
        if (
            target_container.lower() in ("mp4", "m4v")
            and normalize_codec_key(codec_key) == "mpeg4"
        ):
            meta = probe_color_metadata(input_path)
            vff, extra = _build_mpeg4_near_lossless_chain(meta, src_w, src_h)
            if extra:
                final_cmd += extra
            final_cmd += _mpeg4_near_lossless_args_mp4()
            if vff:
                final_cmd += ["-vf", vff]
            final_cmd += _color_metadata_args(meta)

            dedupe_vf_inplace(final_cmd)

            tw, th = (src_w or 0, src_h or 0)
            sanitize_movflags_inplace(final_cmd)
            return TranscodePlan(
                input_path=input_path,
                target_container=target_container,
                preset_name=preset_name,
                codec_key="mpeg4",
                safe_scale=None,
                vf_chain=vff,
                src_w=src_w,
                src_h=src_h,
                src_fps=src_fps,
                target_w=tw,
                target_h=th,
                final_cmd_without_output=final_cmd,
                force_pix_fmt=None,
                skip_alignment_scale=True,
                force_key_at_start=bool(force_key_at_start),
            )

        if target_container.lower() == "mpeg":
            args = (
                ["-c:v", "mpeg2video", "-qscale:v", "1", "-g", "15", "-bf", "2"]
                if normalize_codec_key(codec_key) == "mpeg2video"
                else ["-c:v", "mpeg1video", "-qscale:v", "1", "-g", "15"]
            )

            final_cmd = base_cmd[:] + args
            tw, th = (src_w or 0, src_h or 0)
            sanitize_movflags_inplace(final_cmd)
            return TranscodePlan(
                input_path=input_path,
                target_container=target_container,
                preset_name=preset_name,
                codec_key=normalize_codec_key(codec_key) or "mpeg2video",
                safe_scale=None,
                vf_chain=None,
                src_w=src_w,
                src_h=src_h,
                src_fps=src_fps,
                target_w=tw,
                target_h=th,
                final_cmd_without_output=final_cmd,
                force_pix_fmt=None,
                skip_alignment_scale=True,
                force_key_at_start=bool(force_key_at_start),
            )

        # STRICT lossless möglich?
        if ll.status == "strict" and ll.recommended_encoder:
            choice = _choose_strict_lossless_encoder(
                target_container=target_container,
                src_codec=(probe_video_codec(input_path) or "").lower(),
                src_pix_fmt=src_pix_fmt,
                desired_codec_key=chosen_codec_key,
            )
            final_cmd += choice.args
            if choice.extra_args:
                final_cmd += choice.extra_args

            if src_pix_fmt and _encoder_supports_pix_fmt(choice.encoder, src_pix_fmt):
                force_pix_fmt = src_pix_fmt

            meta = probe_color_metadata(input_path)
            if normalize_codec_key(choice.encoder) == "h264":
                final_cmd = _apply_h264_vui_flags_if_needed(
                    final_cmd, meta, src_pix_fmt
                )
            else:
                final_cmd += _color_metadata_args(meta)

            if target_container.lower() in ("mp4", "m4v") and spec.get(
                "faststart", False
            ):
                final_cmd += ["-movflags", "+faststart"]

            chosen_codec_key = normalize_codec_key(choice.encoder) or "ffv1"
            tw, th = (src_w or 0, src_h or 0)

        else:
            # STRICT nicht möglich → best effort innerhalb der gleichen Codec-Familie
            want = normalize_codec_key(codec_key) or ""

            if want in {"mpeg4", "mpeg2video", "mpeg1video"}:
                meta = probe_color_metadata(input_path)
                vff = None
                if want == "mpeg4":
                    vff, extra = _build_mpeg4_near_lossless_chain(meta, src_w, src_h)
                    if extra:
                        final_cmd += extra
                    final_cmd += _mpeg4_near_lossless_args_mp4()
                    if vff:
                        final_cmd += ["-vf", vff]
                elif want == "mpeg2video":
                    final_cmd += _mpeg2_near_lossless_args()
                else:
                    final_cmd += _mpeg1_near_lossless_args()

                final_cmd += _color_metadata_args(meta)

                dedupe_vf_inplace(final_cmd)

                tw, th = (src_w or 0, src_h or 0)
                sanitize_movflags_inplace(final_cmd)
                return TranscodePlan(
                    input_path=input_path,
                    target_container=target_container,
                    preset_name=preset_name,
                    codec_key=want,
                    safe_scale=None,
                    vf_chain=vff if want == "mpeg4" else None,
                    src_w=src_w,
                    src_h=src_h,
                    src_fps=src_fps,
                    target_w=tw,
                    target_h=th,
                    final_cmd_without_output=final_cmd,
                    force_pix_fmt=None,
                    skip_alignment_scale=True,
                    force_key_at_start=bool(force_key_at_start),
                )

            enc_text = _run_ffmpeg_encoders("ffmpeg")
            avail = set(_parse_encoder_names(enc_text))
            candidates = _policy_candidates_same_family(want, preferred_encoder, avail)
            if not candidates:
                raise RuntimeError(
                    f"No encoder available for requested codec '{want}'."
                )

            enc_choice = candidates[0]
            final_cmd += _lossless_video_args_for_encoder(enc_choice)

            core_i, hw_i = _encoder_core_hw(enc_choice)
            if hw_i == "qsv":
                _inject_qsv_device(final_cmd, enc_choice)
            elif hw_i == "vaapi":
                _inject_vaapi_device(final_cmd)
                if "-vf" in final_cmd:
                    i = final_cmd.index("-vf")
                    vf = final_cmd[i + 1]
                    if "hwupload" not in vf:
                        final_cmd[i + 1] = "format=nv12,hwupload" + (
                            "," + vf if vf else ""
                        )
                else:
                    final_cmd += ["-vf", "format=nv12,hwupload"]

            if not _try_quick_encode(final_cmd + ["__probe__.mkv"]):
                _strip_hw_device_options(final_cmd)
                for cand in defin.CODEC_FALLBACK_POLICY.get(want, []):
                    if cand == enc_choice or cand not in avail:
                        continue
                    test_cmd = base_cmd[:]
                    test_cmd += _lossless_video_args_for_encoder(cand)
                    core_j, hw_j = _encoder_core_hw(cand)
                    if hw_j == "qsv":
                        _inject_qsv_device(test_cmd, cand)
                    elif hw_j == "vaapi":
                        _inject_vaapi_device(test_cmd)
                        if "-vf" in test_cmd:
                            i = test_cmd.index("-vf")
                            vf = test_cmd[i + 1]
                            if "hwupload" not in vf:
                                test_cmd[i + 1] = "format=nv12,hwupload" + (
                                    "," + vf if vf else ""
                                )
                        else:
                            test_cmd += ["-vf", "format=nv12,hwupload"]
                    if _try_quick_encode(test_cmd + ["__probe__.mkv"]):
                        final_cmd = test_cmd
                        enc_choice = cand
                        break

            chosen_codec_key = want

            # gleich nachdem chosen_codec_key feststeht:
            if normalize_codec_key(chosen_codec_key) == "qtrle":
                # Alpha erkennen (grob) und passendes RGB-Pixelformat wählen
                has_alpha = (src_pix_fmt or "").lower() in {
                    "argb",
                    "rgba",
                    "bgra",
                    "ya8",
                    "yuva420p",
                    "yuva422p",
                    "yuva444p",
                }
                force_pix_fmt = "argb" if has_alpha else "rgb24"

            meta = probe_color_metadata(input_path)
            if normalize_codec_key(enc_choice) == "h264":
                final_cmd = _apply_h264_vui_flags_if_needed(
                    final_cmd, meta, src_pix_fmt
                )
            else:
                final_cmd += _color_metadata_args(meta)

            if src_pix_fmt and _encoder_supports_pix_fmt(enc_choice, src_pix_fmt):
                force_pix_fmt = src_pix_fmt

            tw, th = (src_w or 0, src_h or 0)

        if target_container.lower() == "webm":
            final_cmd += ["-vsync", "0"]

        if normalize_codec_key(chosen_codec_key) in {"dnxhd", "dnxhr"}:
            tw, th = (src_w or 0, src_h or 0)
            final_cmd = _apply_dnx_rules_if_needed(
                final_cmd,
                codec_key=chosen_codec_key,
                container=target_container,
                target_w=tw,
                target_h=th,
                src_fps=src_fps,
                preset_name=preset_name,
            )

        # --- Only for MKV + MJPEG + Lossless: enforce full-range YUVJ ---
        if target_container.lower() in ("mkv", "matroska") and (
            normalize_codec_key(chosen_codec_key) == "mjpeg"
            or _encoder_family(
                _active_encoder(final_cmd, encoder_for_codec(chosen_codec_key))
            )
            == "mjpeg"
        ):
            _strip_options(final_cmd, {"-color_range"})
            final_cmd += ["-color_range", "pc"]
            force_pix_fmt = None

        # --- Alpha-fähige Codecs (Lossless-Pfad): zentrale Regeln anwenden ---
        final_cmd, force_pix_fmt = _apply_alpha_rules(
            final_cmd, chosen_codec_key, target_container, src_pix_fmt, force_pix_fmt
        )

        dedupe_vf_inplace(final_cmd)
        sanitize_movflags_inplace(final_cmd)
        return TranscodePlan(
            input_path=input_path,
            target_container=target_container,
            preset_name=preset_name,
            codec_key=chosen_codec_key,
            safe_scale=None,
            vf_chain=None,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            target_w=tw,
            target_h=th,
            final_cmd_without_output=final_cmd,
            force_pix_fmt=force_pix_fmt,
            skip_alignment_scale=skip_alignment_scale,
            force_key_at_start=bool(force_key_at_start),
        )

    # ===================== NICHT-LOSSLESS PFAD =====================
    # → Hier die vom User gewünschten Werte *wirklich* anwenden
    vf_parts: List[str] = []

    # 1) User-VF-Kette (enthält ggf. scale=…, setsar, setdar, format=…, fps=…)
    if user_vf_chain:
        vf_parts.append(user_vf_chain)
        safe_scale = user_safe_scale
        tw, th = (user_tgt_w or (src_w or 0), user_tgt_h or (src_h or 0))
    else:
        # Fallback: keine User-Vorgabe → nichts vorab in die VF-Kette
        safe_scale = None
        tw, th = (src_w or 0, src_h or 0)

    # 2) „Visual Sanity Layer“ nur ergänzen, wenn der User nichts skaliert hat
    meta = probe_color_metadata(input_path)
    _, _, src_pf_for_vsl = ffprobe_geometry(input_path)
    if not user_safe_scale:
        vsl_vf, vsl_extra = build_visual_chain_generic(
            meta=meta,
            src_w=src_w,
            src_h=src_h,
            container=target_container,
            codec_key=chosen_codec_key,
            src_pix_fmt=src_pf_for_vsl,
            scale_already_planned=bool(user_safe_scale),
        )
        if vsl_extra:
            base_cmd += vsl_extra
        if vsl_vf:
            vf_parts.append(vsl_vf)

    vf_chain = ",".join([p for p in vf_parts if p]) if vf_parts else None

    # container-spezifisches forced pix_fmt (nur falls NICHT schon in -vf format= gesetzt wird)
    pf_map = defin.CODEC_FORCED_PIX_FMT.get(
        normalize_codec_key(chosen_codec_key) or "", {}
    )
    forced_pf = pf_map.get(target_container.lower()) or pf_map.get("*")
    if forced_pf:
        force_pix_fmt = forced_pf

    # Encoder wählen/prüfen & Kommando bauen
    final_cmd = try_encode_with_fallbacks(
        base_cmd,
        codec_key=chosen_codec_key,
        container=target_container,
        preset_name=preset_name,
        vf_chain=vf_chain,
        ffmpeg_bin="ffmpeg",
        preferred_encoder=chosen_preferred_encoder,
    )

    # DNx-Sonderlogik anwenden (Profile/Bitraten/Pixelformat)
    final_cmd = _apply_dnx_rules_if_needed(
        final_cmd,
        codec_key=chosen_codec_key,
        container=target_container,
        target_w=tw,
        target_h=th,
        src_fps=src_fps,
        preset_name=preset_name,
    )
    final_cmd = apply_container_codec_quirks(
        final_cmd, container=target_container, codec_key=chosen_codec_key
    )

    # Farbraum-Signalisierung (VUI vs. Container)
    final_cmd = apply_color_signaling(
        final_cmd,
        container=target_container,
        codec_key=chosen_codec_key,
        meta=meta,
        src_pix_fmt=src_pf_for_vsl,
    )

    # JPEG2000 in toleranten Containern: adaptives Chroma/Pixelformat
    if normalize_codec_key(chosen_codec_key) == "jpeg2000" and (
        target_container.lower() in ("mkv", "matroska", "mov", "avi")
    ):
        _strip_options(final_cmd, {"-pix_fmt"})
        force_pix_fmt = None

        try:
            avail_encs = _available_encoders()
            if "jpeg2000" in avail_encs:
                if "-c:v" in final_cmd:
                    i_cv = final_cmd.index("-c:v")
                    if (
                        i_cv + 1 < len(final_cmd)
                        and final_cmd[i_cv + 1] == "libopenjpeg"
                    ):
                        final_cmd[i_cv + 1] = "jpeg2000"
                else:
                    final_cmd += ["-c:v", "jpeg2000"]
        except Exception:
            pass

    if normalize_codec_key(chosen_codec_key) == "mjpeg":
        if "-q:v" not in final_cmd and "-qscale:v" not in final_cmd:
            final_cmd += ["-q:v", "2"]

    # --- Alpha-fähige Codecs (Nicht-Lossless-Pfad): zentrale Regeln anwenden ---
    final_cmd, force_pix_fmt = _apply_alpha_rules(
        final_cmd, chosen_codec_key, target_container, src_pf_for_vsl, force_pix_fmt
    )

    # Falls oben noch nichts erzwungen wurde, setze eine sichere Default-Farbe (historische Kompatibilität)
    # am Ende von build_transcode_plan (NICHT-LOSSY Pfad), statt des festen yuvj422p:
    if not force_pix_fmt:
        enc = _active_encoder(
            final_cmd, encoder_for_codec(normalize_codec_key(chosen_codec_key) or "")
        )
        fam = _encoder_family(enc)
        # Nur YUV-Familien bekommen ein Default; RGB-/RLE-Codecs NICHT anrühren
        if fam in {
            "h264",
            "hevc",
            "vp9",
            "av1",
            "vp8",
            "mpeg4",
            "mpeg2video",
            "prores",
            "dnxhd",
            "mjpeg",
        }:
            # für MP4/WebM sorgt playback_pix_fmt_for ohnehin für yuv420p;
            # für MKV/MOV lassen wir es i.d.R. frei (None)
            force_pix_fmt = playback_pix_fmt_for(target_container)
        else:
            force_pix_fmt = None  # z. B. qtrle, png, rawvideo → Encoder-Default (RGB)

    if target_container.lower() == "webm":
        final_cmd += ["-vsync", "0"]

    dedupe_vf_inplace(final_cmd)
    sanitize_movflags_inplace(final_cmd)
    return TranscodePlan(
        input_path=input_path,
        target_container=target_container,
        preset_name=preset_name,
        codec_key=chosen_codec_key,
        safe_scale=safe_scale,  # ← enthält die vom User gewünschte Zielauflösung (Encoder-safe)
        vf_chain=vf_chain,  # ← enthält auch fps=..., wenn user_fps_rational gesetzt war
        src_w=src_w,
        src_h=src_h,
        src_fps=src_fps,
        target_w=tw,
        target_h=th,  # ← Zielabmessungen gemäß safe_scale / User-Wunsch
        final_cmd_without_output=final_cmd,
        force_pix_fmt=force_pix_fmt,
        skip_alignment_scale=skip_alignment_scale,
        force_key_at_start=bool(force_key_at_start),
    )


def assemble_ffmpeg_cmd(plan, output_path: Union[str, Path]) -> List[str]:
    """
    Nimmt einen TranscodePlan und liefert die finale ffmpeg-Commandliste.
    - Fügt plan.vf_chain hinzu, falls noch nicht vorhanden.
    - Erzwingt plan.force_pix_fmt per '-pix_fmt' und entfernt dann evtl. 'format='-Filter
      aus -vf/-filter:v (damit sich Pixelformat-Signalisierung nicht widerspricht).
    - Deduped 'format='-Duplikate in der Filterkette.
    - Hängt den Ausgabepfad an.
    """
    # defensiv kopieren
    cmd: List[str] = list(getattr(plan, "final_cmd_without_output", []))

    # Falls noch kein -vf gesetzt ist, aber plan.vf_chain existiert → einfügen
    vf_chain = getattr(plan, "vf_chain", None)
    if isinstance(vf_chain, str) and vf_chain:
        if "-vf" in cmd:
            # existierende -vf NICHT überschreiben (Plan hat i.d.R. bereits alles gesetzt)
            pass
        else:
            cmd += ["-vf", vf_chain]

    # Falls ein force_pix_fmt gefordert ist → per -pix_fmt setzen und 'format=' aus -vf entfernen
    force_pf = getattr(plan, "force_pix_fmt", None)
    if isinstance(force_pf, str) and force_pf:
        # 1) alte -pix_fmt entfernen (wir wollen nur eine, die *letzte* gewinnt)
        _strip_opt_with_param(cmd, {"-pix_fmt"})
        cmd += ["-pix_fmt", force_pf]

        # 2) 'format=' aus -vf/-filter:v tilgen, damit es keinen Konflikt gibt
        _strip_format_from_filter_arg(cmd, flag="-vf")
        _strip_format_from_filter_arg(cmd, flag="-filter:v")

    # OPTIONAL: dedupe innerhalb einer evtl. vorhandenen -vf-Kette (z. B. doppelte Filterteile)
    dedupe_vf_inplace(cmd)

    # Ausgabepfad anhängen
    out_str = str(output_path)
    cmd.append(out_str)
    ensure_pre_output_order(cmd)
    sanitize_movflags_inplace(cmd)
    return cmd


# --- kleine, lokale Helfer für assemble_ffmpeg_cmd --------------------------
def _strip_opt_with_param(cmd: List[str], names: set[str]) -> None:
    """
    Entfernt alle Vorkommen bekannter 2-Token-Optionen (z. B. '-pix_fmt <fmt>').
    """
    i = 0
    out: List[str] = []
    L = len(cmd)
    while i < L:
        tok = cmd[i]
        if tok in names and i + 1 < L:
            i += 2  # Option + Wert überspringen
            continue
        out.append(tok)
        i += 1
    cmd[:] = out


def _strip_format_from_vf_string(vf: str) -> str:
    """
    Entfernt *alle* 'format=…' Fragmente aus einer Filterkette.
    """
    parts = [p.strip() for p in vf.split(",") if p.strip()]
    parts = [p for p in parts if not p.lower().startswith("format=")]
    return ",".join(parts)


def _strip_format_from_filter_arg(cmd: List[str], flag: str) -> None:
    """
    Entfernt 'format='-Teile aus dem Wert zu -vf/-filter:v, falls vorhanden.
    """
    try:
        if flag in cmd:
            i = cmd.index(flag)
            if i + 1 < len(cmd) and isinstance(cmd[i + 1], str):
                cmd[i + 1] = _strip_format_from_vf_string(cmd[i + 1])
    except Exception:
        # defensiv: niemals hart failen
        pass


def _strip_vp9_libvpx_opts_inplace(cmd: list[str]) -> None:
    # Flags, die nur für libvpx-vp9 gelten – QSV/VAAPI mögen die nicht
    _strip_options(cmd, {"-row-mt", "-tile-columns", "-tile-rows", "-cpu-used"})
    # CRF ist libvpx-typisch; bei QSV nutzen wir eher -b:v / -global_quality
    _strip_options(cmd, {"-crf"})
