#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_pixfmt – Zentrale Pixelformat- und Farbsignalisierungs-Policy
-------------------------------------------------------------------
Dieses Modul entscheidet Policies (welches pix_fmt Ziel? welche Farbtags?),
mutiert aber selbst keine ffmpeg-Kommandoliste, mit Ausnahme der Farb-
signalisierung via apply_color_signaling(cmd, ...), die optional Container-
Tags/VUI ergänzt.

Öffentliche API:
- PixfmtContext
- PixfmtDecision
- decide_pix_fmt(ctx) -> PixfmtDecision
- apply_color_signaling(cmd, *, container, codec_key, meta, src_pix_fmt)
- infer_target_pix_fmt_from_plan(obj)
- playback_pix_fmt_for(container, encoder=None)

Design-Notizen:
- Alle Entscheidungswege laufen über decide_pix_fmt.
- Spezielle Codecs (ProRes/DNx/J2K/MJPEG/RAW/PNG/FFV1/UT/HuffYUV/CineForm/HAP)
  bekommen – falls nötig – ein *explizites* "-pix_fmt" (pix_fmt_flag).
  Für andere wird ein terminales format=… (vf_terminal_format) empfohlen.
- Hardware-Pfade (NVENC/QSV/VAAPI) legen Formate implizit fest (NV12/p010) –
  hier greifen wir *nicht* ein.
- Alpha-Erhalt wird zentral bewertet; MP4/M4V/WEBM/MPEG erzwingen 4:2:0 ohne Alpha.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

# ──────────────────────────────────────────────────────────────────────────────
#  Externe Helfer aus dem Hauptmodul (robuste Imports)
# ──────────────────────────────────────────────────────────────────────────────
from VideoEncodersCodecs_support import (
    _active_encoder,
    _active_encoder_or_guess,
    _apply_h264_vui_flags_if_needed,
    _cmd_contains_any,
    _container_allows_codec,
    _encoder_family,
    _encoder_for_codec,
    _encoder_supports_pix_fmt,
    _ffprobe_geometry,
    _get_preset_spec,
    _is_explicit_lossless,
    _is_stream_copy,
    _normalize_codec_key,
    _output_index,
    _set_kv_arg,
    _strip_options,
)  # type: ignore

import definitions as defin  # type: ignore
from i18n import _  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
#  Öffentliche Dataklassen
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PixfmtContext:
    # Kern
    container: str = ""
    codec_key: str = ""
    encoder_name: Optional[str] = None

    # Preset & Policy
    preset: Optional[str] = None  # Name des Presets (z.B. "web", "ultra", "lossless")
    prefer_10bit: Optional[bool] = None  # Qualität bevorzugen (10-bit, falls möglich)
    force_yuv420p: Optional[bool] = None  # maximale Kompatibilität 4:2:0 erzwingen

    # Quelle / Analyse
    src_pix_fmt: Optional[str] = None  # aus ffprobe
    color_meta: Dict[str, Optional[str]] = field(
        default_factory=dict
    )  # aus probe_color_metadata
    wants_alpha: Optional[bool] = None  # expliziter Wunsch, falls bekannt

    # Pipeline-Kontext
    has_filtergraph: bool = False  # gibt es -vf/-filter:v?
    is_stream_copy: bool = False  # -c:v copy?
    is_explicit_lossless: bool = False  # visual/strict lossless erwünscht
    hw_family: Optional[str] = None  # "nvenc" | "qsv" | "vaapi" | None


@dataclass
class PixfmtDecision:
    vf_terminal_format: Optional[str]  # z.B. "yuv420p" / "yuv422p10le" / "gbrap10le"
    pix_fmt_flag: Optional[str]  # falls -pix_fmt gesetzt werden soll
    alpha_preserved: bool
    reason: str  # kurze Begründung/Policy


@dataclass
class AlphaCheck:
    src_has_alpha: bool
    target_can_preserve: bool
    will_lose_alpha: bool
    reason: Optional[str]
    suggestions: list[tuple[str, str]]  # [(container, codec_key)] – Alternativen
    candidate_pix_fmts: list[str]  # alpha-fähige Ziel-Pixelformate, falls vorhanden


@dataclass
class CodecChoiceValidation:
    ok: bool
    errors: list[str]
    allowed_codecs_for_format: list[str]  # laut Formatspezifikation
    container_allowed: bool


__all__ = [
    "PixfmtContext",
    "PixfmtDecision",
    "decide_pix_fmt",
    "apply_color_signaling",
    "infer_target_pix_fmt_from_plan",
    "playback_pix_fmt_for",
    "probe_color_metadata",
]


# Container, in denen wir *keine* Alpha-Pfade verfolgen (Policy-konform zu _decide_best_target_pix_fmt)
_ALPHA_DISABLED_CONTAINERS: set[str] = {"mp4", "m4v", "webm", "mpeg"}

# Codecs, die Alpha sinnvoll tragen können (ggf. abhängig vom Pixelformat/Profil)
_ALPHA_OK_CODECS: set[str] = {
    "ffv1",
    "utvideo",
    "huffyuv",
    "rawvideo",
    "png",
    "qtrle",
    "magicyuv",
    "prores",
    "cineform",
    "hap",
}

# ──────────────────────────────────────────────────────────────────────────────
#  Zentraler Entscheider
# ──────────────────────────────────────────────────────────────────────────────


def decide_pix_fmt(ctx: PixfmtContext) -> PixfmtDecision:
    """
    Liefert *nur* die Entscheidung (Policy), mutiert kein Kommando.
    Berücksichtigt Container/Codec/HW/Alpha/HDR/Presets:
      - force_yuv420p → maximale Kompatibilität (MP4 strikt 8-bit 420; globales Gate)
      - prefer_10bit  → bestmögliche Qualität (10-bit), wenn sinnvoll & unterstützt
      - lossless      → Quell-pix_fmt beibehalten (falls möglich)
    """
    c = (ctx.container or "").lower()
    k = _normalize_codec_key(ctx.codec_key or "")
    enc = (ctx.encoder_name or "") or _encoder_for_codec(k)
    src_pf = (ctx.src_pix_fmt or "").lower()

    src_has_alpha = _pix_fmt_has_alpha(src_pf)
    wants_alpha = bool(
        ctx.wants_alpha if ctx.wants_alpha is not None else src_has_alpha
    )

    CODECS_REQUIRE_EXPLICIT_PIX_FMT = {
        "prores",
        "dnxhd",
        "dnxhr",
        "qtrle",
        "png",
        "rawvideo",
        "ffv1",
        "utvideo",
        "huffyuv",
        "magicyuv",
        "cineform",
        "jpeg2000",
        "mjpeg",
        "hap",
    }
    ALPHA_DISABLED_CONTAINERS = {"mp4", "m4v", "webm", "mpeg", "mpg"}

    # 0) Stream-Copy: nie eingreifen
    if ctx.is_stream_copy:
        alpha_ok = src_has_alpha and (c not in ALPHA_DISABLED_CONTAINERS)
        return PixfmtDecision(
            None, None, alpha_ok, "stream-copy: Quelle bleibt unverändert"
        )

    # 1) Hardware-Encoder: Formate werden implizit (NV12/p010) bestimmt – nicht eingreifen
    hw = (ctx.hw_family or "").lower()
    if hw in {"vaapi", "qsv", "nvenc"}:
        return PixfmtDecision(
            None,
            None,
            src_has_alpha and c not in ALPHA_DISABLED_CONTAINERS,
            f"hw-encode ({hw}): Format implizit (nv12/p010)",
        )

    # 2) Globales Kompatibilitäts-Gate (Preset): force_yuv420p ⇒ **immer** yuv420p
    #    (strikter als zuvor: kein Aufwerten auf 10-bit bei gesetztem force_yuv420p)
    if ctx.force_yuv420p:
        return PixfmtDecision(
            "yuv420p", "yuv420p", False, "preset: force_yuv420p (immer 8-bit 4:2:0)"
        )

    # 3) Explizit lossless? → Quelle beibehalten, falls Container + Encoder können
    if ctx.is_explicit_lossless and (src_pf or ""):
        keep, _prof = _should_preserve_src_pix_fmt_in_lossless(c, enc, src_pf)
        if keep and _encoder_supports_pix_fmt(enc, src_pf):
            needs_flag = k in CODECS_REQUIRE_EXPLICIT_PIX_FMT
            alpha_ok = _pix_fmt_has_alpha(src_pf) and (
                c not in ALPHA_DISABLED_CONTAINERS
            )
            return PixfmtDecision(
                None if not needs_flag else None,
                src_pf if needs_flag else None,
                alpha_ok,
                "lossless: Quell-pix_fmt beibehalten",
            )

    # 4) Alpha-Pfade priorisieren, wenn Container & Codec es realistisch tragen
    if wants_alpha and (c not in ALPHA_DISABLED_CONTAINERS) and (k in _ALPHA_OK_CODECS):
        pick: Optional[str] = None
        if k == "prores":  # nur 4444 trägt Alpha
            for pf in ("yuva444p10le", "yuva444p"):
                if _encoder_supports_pix_fmt(enc, pf):
                    pick = pf
                    break
        if not pick:
            pick = _pick_alpha_preserving_pix_fmt(enc, src_pf)
        if pick:
            needs_flag = k in CODECS_REQUIRE_EXPLICIT_PIX_FMT
            return PixfmtDecision(
                None if needs_flag else pick,
                pick if needs_flag else None,
                True,
                "alpha: alpha-fähiges pix_fmt gewählt",
            )

    # 5) Container-Policies
    # ── MP4/M4V: kein Alpha; feiner abgestufte 420-Entscheidung ───────────────
    if c in {"mp4", "m4v"}:
        # 5a) MPEG-4 Part 2 (mpeg4/libxvid) in MP4 → grundsätzlich 8-bit 4:2:0
        if _is_mpeg4_family(k) or _is_mpeg4_family(enc):
            return PixfmtDecision(
                "yuv420p", None, False, "mp4+mpeg4: strikt 8-bit 4:2:0"
            )

        # 5b) HDR-Signal → 10-bit 4:2:0, wenn Encoder das kann
        if _is_hdr_signal(ctx.color_meta, src_pf) and _encoder_supports_pix_fmt(
            enc, "yuv420p10le"
        ):
            return PixfmtDecision("yuv420p10le", None, False, "mp4: HDR → 10-bit 4:2:0")

        # 5c) ansonsten: prefer_10bit? Dann yuv420p10le wenn verfügbar, sonst yuv420p
        if _truthy(ctx.prefer_10bit) and _encoder_supports_pix_fmt(enc, "yuv420p10le"):
            return PixfmtDecision(
                "yuv420p10le", None, False, "mp4: prefer_10bit → 10-bit 4:2:0"
            )

        return PixfmtDecision("yuv420p", None, False, "mp4: Standard 8-bit 4:2:0")

    # ── WEBM: konservativ 420 ────────────────────────────────────────────────
    if c == "webm":
        return PixfmtDecision("yuv420p", None, False, "webm: Standard 4:2:0")

    # 5) Container-Policies
    # Vor den MOV/MKV/AVI-Fallbacks: Interframe-Codecs (vpx/h26x/mpeg) bekommen niemals RGBA/Alpha-Ausgabe
    fam = _encoder_family(enc)
    if fam in {"vp9", "vp8", "h264", "hevc", "av1", "mpeg4", "mpeg2video"}:
        # bevorzugt 10-bit 420, falls Encoder kann; sonst 8-bit 420
        if _truthy(ctx.prefer_10bit) and _encoder_supports_pix_fmt(enc, "yuv420p10le"):
            return PixfmtDecision(
                "yuv420p10le", None, False, "interframe: YUV 4:2:0 10-bit"
            )
        return PixfmtDecision("yuv420p", None, False, "interframe: YUV 4:2:0")

    # ── MOV/MKV/AVI: Quelle beibehalten oder in 10-bit-Variante upgraden ─────
    if src_pf and _encoder_supports_pix_fmt(enc, src_pf):
        if _truthy(ctx.prefer_10bit):
            up10 = _upgrade_to_10bit_like(src_pf)
            if up10 and _encoder_supports_pix_fmt(enc, up10):
                needs_flag = k in CODECS_REQUIRE_EXPLICIT_PIX_FMT
                return PixfmtDecision(
                    None if needs_flag else up10,
                    up10 if needs_flag else None,
                    _pix_fmt_has_alpha(up10),
                    "mov/mkv/avi: prefer_10bit (upgraded)",
                )
        needs_flag = k in CODECS_REQUIRE_EXPLICIT_PIX_FMT
        return PixfmtDecision(
            None if needs_flag else src_pf,
            src_pf if needs_flag else None,
            _pix_fmt_has_alpha(src_pf),
            "mov/mkv/avi: Quelle beibehalten",
        )

    # 6) Codec-Defaults (Intra-Codecs brauchen oft explizites -pix_fmt)
    if k in CODECS_REQUIRE_EXPLICIT_PIX_FMT:
        default_pf = _default_pix_fmt_for_intra_codec(
            k, prefer10=_truthy(ctx.prefer_10bit)
        )
        return PixfmtDecision(
            None, default_pf, "a" in (default_pf or ""), f"{k}: default -pix_fmt"
        )

    # 7) H.264/H.265/AV1/VPx generischer Fallback in MOV/MKV/AVI
    bit = _bit_depth_from_pix_fmt(src_pf)
    cands: List[str] = []
    if _truthy(ctx.prefer_10bit):
        cands += ["yuv422p10le", "yuv420p10le"]
    cands += (["yuv422p"] if bit >= 10 else []) + ["yuv422p", "yuv420p"]
    for cand in cands:
        if _encoder_supports_pix_fmt(enc, cand):
            return PixfmtDecision(cand, None, False, "fallback-Leiter (progressiv)")

    # 8) Keine sinnvolle Entscheidung möglich
    return PixfmtDecision(None, None, False, "keine Entscheidung (no-op)")


# ──────────────────────────────────────────────────────────────────────────────
#  Color-Metadaten & Signalisierung
# ──────────────────────────────────────────────────────────────────────────────


def probe_color_metadata(
    path: Path, ffprobe_bin: str = "ffprobe"
) -> Dict[str, Optional[str]]:
    try:
        out = subprocess.check_output(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_space,color_transfer,color_primaries,color_range,pix_fmt",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
        data = cast(Dict[str, Any], json.loads(out))
        streams = cast(List[Dict[str, Any]], data.get("streams") or [])
        st: Dict[str, Any] = streams[0] if streams else {}
        return {
            "colorspace": cast(Optional[str], st.get("color_space")),
            "color_trc": cast(Optional[str], st.get("color_transfer")),
            "color_primaries": cast(Optional[str], st.get("color_primaries")),
            "color_range": cast(Optional[str], st.get("color_range")),
            "pix_fmt": cast(Optional[str], st.get("pix_fmt")),
        }
    except Exception:
        return {
            "colorspace": None,
            "color_trc": None,
            "color_primaries": None,
            "color_range": None,
            "pix_fmt": None,
        }


def _color_metadata_args(meta: Dict[str, Optional[str]]) -> List[str]:
    args: List[str] = []
    if not meta:
        return args
    VALID_COLORSPACES = {
        "rgb",
        "bt709",
        "fcc",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "ycgco",
        "bt2020ncl",
        "bt2020cl",
        "smpte2085",
        "chroma-derived-ncl",
        "chroma-derived-cl",
        "ictcp",
    }
    prim = (meta.get("color_primaries") or "").strip()
    trc = (meta.get("color_trc") or "").strip()
    cs = (meta.get("colorspace") or "").strip().lower()
    rng = (meta.get("color_range") or "").strip().lower()
    if cs == "gbr":
        cs = "rgb"
    if prim:
        args += ["-color_primaries", prim]
    if trc:
        args += ["-color_trc", trc]
    if cs and cs in VALID_COLORSPACES:
        args += ["-colorspace", cs]
    if rng:
        args += ["-color_range", rng]
    return args


def _color_metadata_args_safe(
    container: str, codec_key: str, meta: Dict[str, Optional[str]]
) -> List[str]:
    c = (container or "").lower()
    k = _normalize_codec_key(codec_key or "") or ""
    # Keine Farbtags für MJPEG (JPEG-Range-Kollisionen)
    if k == "mjpeg":
        return []
    # Für H.264 in MP4/MOV/MKV → VUI über SPS statt Container-Tags
    if c in ("mp4", "m4v", "mov", "mkv", "matroska") and k == "h264":
        return []
    return _color_metadata_args(meta)


def apply_color_signaling(
    cmd: List[str],
    *,
    container: str,
    codec_key: str,
    meta: Dict[str, Optional[str]],
    src_pix_fmt: Optional[str],
):
    """Fügt *falls sinnvoll* Farbsignalisierung (Primaries/TRC/Matrix/Range) hinzu.
    RGB/Alpha-Formate sowie PNG/QTRLE/RAW werden nicht beschriftet.
    H.264 in MP4/MOV/MKV wird via _apply_h264_vui_flags_if_needed() behandelt.
    """
    if (src_pix_fmt or "").lower().startswith(("rgb", "gbr")) or _normalize_codec_key(
        codec_key
    ) in {"qtrle", "png", "rawvideo"}:
        return cmd
    k = _normalize_codec_key(codec_key or "")
    if k == "mjpeg":
        return cmd
    # Erzwinge Range je nach Codec-Policy (falls definiert)
    try:
        forced_range = getattr(defin, "CODEC_FORCE_COLOR_RANGE", {}).get(k)
    except Exception:
        forced_range = None
    if forced_range:
        _strip_options(
            cmd, {"-color_primaries", "-color_trc", "-colorspace", "-color_range"}
        )
        cmd += ["-color_range", forced_range]
        return cmd
    if (container or "").lower() in (
        "mp4",
        "m4v",
        "mov",
        "mkv",
        "matroska",
    ) and k == "h264":
        return _apply_h264_vui_flags_if_needed(cmd, meta, src_pix_fmt)
    cmd += _color_metadata_args_safe(container, codec_key, meta)
    return cmd


def _dedupe_color_tags(cmd: List[str]) -> None:
    """
    Behalte pro -color_* nur das letzte VOR dem Output.
    """
    if not cmd:
        return
    out_idx = _output_index(cmd)
    keys = ["-color_range", "-color_primaries", "-color_trc", "-colorspace"]
    for k in keys:
        idxs = [i for i in range(min(out_idx, len(cmd))) if cmd[i] == k]
        if len(idxs) > 1:
            # alle bis auf den letzten entfernen (inkl. Wert dahinter, falls vorhanden)
            for i in reversed(idxs[:-1]):
                del cmd[
                    i : (
                        i + 2
                        if i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("-")
                        else i + 1
                    )
                ]


# ──────────────────────────────────────────────────────────────────────────────
#  Diagnose/Komfort
# ──────────────────────────────────────────────────────────────────────────────


def playback_pix_fmt_for(
    container: Optional[str], encoder: Optional[str] = None
) -> Optional[str]:
    """
    Heuristik für ein "sicher abspielbares" Pixelformat:
    - MP4/M4V/WEBM → 8-bit 4:2:0
    - MOV/MKV/AVI → nicht festgelegt (None)
    Hinweis: Dies ist nur eine Anzeige-/Diagnosehilfe; die echte Entscheidung
    erfolgt in decide_pix_fmt(...).
    """
    enc = (encoder or "").lower()
    if enc.startswith("prores"):
        return "yuv422p10le"
    if enc == "dnxhd":
        return "yuv422p"
    c = (container or "").lower()
    if c in {"mp4", "m4v", "webm"}:
        return "yuv420p"
    if c in {"mkv", "matroska", "mov", "avi"}:
        return None
    return None


def infer_target_pix_fmt_from_plan(obj) -> Optional[str]:
    """Ermittelt aus Plan/Kommando das effektiv geplante Ziel-Pixelformat."""

    def _from_vf(vf: Optional[str]) -> Optional[str]:
        if not vf:
            return None
        m = re.search(r"(?:^|,)\s*format=([a-z0-9_]+)", vf.strip(), flags=re.IGNORECASE)
        return m.group(1) if m else None

    def _from_cmd(tokens: Iterable[Union[str, Path]]) -> Optional[str]:
        toks = [str(t) for t in tokens]
        if "-pix_fmt" in toks:
            i = toks.index("-pix_fmt")
            if i + 1 < len(toks):
                return toks[i + 1]
        for flag in ("-filter:v", "-vf"):
            if flag in toks:
                j = toks.index(flag)
                if j + 1 < len(toks):
                    pf = _from_vf(toks[j + 1])
                    if pf:
                        return pf
        return None

    if hasattr(obj, "__dict__") and hasattr(obj, "final_cmd_without_output"):
        force_pf = getattr(obj, "force_pix_fmt", None)
        if isinstance(force_pf, str) and force_pf:
            return force_pf
        vf_chain = getattr(obj, "vf_chain", None)
        pf_from_vf = _from_vf(vf_chain) if isinstance(vf_chain, str) else None
        if pf_from_vf:
            return pf_from_vf
        cmd = getattr(obj, "final_cmd_without_output", None)
        if isinstance(cmd, (list, tuple)):
            pf_cmd = _from_cmd(cmd)
            if pf_cmd:
                return pf_cmd
        return None
    if isinstance(obj, (list, tuple)):
        return _from_cmd(obj)
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Interne Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _is_mpeg4_family(name: Optional[str]) -> bool:
    """
    Erkenne MPEG-4 Part 2 Encoder-/Codecnamen (mpeg4/libxvid/xvid).
    Wird genutzt, um MP4+MPEG-4 strikt auf yuv420p festzunageln.
    """
    e = (name or "").lower()
    return e in {"mpeg4", "libxvid", "xvid"}


def _truthy(x: Optional[bool]) -> bool:
    return bool(x)


def _is_hdr_signal(meta: Dict[str, Optional[str]], src_pix_fmt: Optional[str]) -> bool:
    prim = (meta.get("color_primaries") or "").lower()
    trc = (meta.get("color_trc") or "").lower()
    bit = _bit_depth_from_pix_fmt(src_pix_fmt)
    is_bt2020 = ("2020" in prim) or (prim == "bt2020")
    is_hdr_trc = trc in {"smpte2084", "arib-std-b67", "pq", "hlg"}
    return is_bt2020 and is_hdr_trc and bit >= 10


def _bit_depth_from_pix_fmt(pf: Optional[str]) -> int:
    p = (pf or "").lower()
    if ("12" in p) or ("14" in p) or ("16" in p):
        return 12
    if ("10" in p) or ("p10" in p) or p.endswith("10le") or p.endswith("10be"):
        return 10
    return 8


def _pix_fmt_has_alpha(pf: Optional[str]) -> bool:
    p = (pf or "").lower()
    if not p:
        return False
    if p.startswith(("rgba", "bgra", "argb", "abgr", "gbrap", "yuva", "ayuv")):
        return True
    if p in {"ya8", "ya16be", "ya16le"}:
        return True
    if re.fullmatch(r"gbrap(?:10|12)?le", p):
        return True
    if re.fullmatch(r"yuva\d{3}p(?:10|12)?le", p):
        return True
    return False


def _alpha_candidates_for(src_pix_fmt: Optional[str]) -> List[str]:
    """Priorisierte Liste alpha-fähiger Formate (näher an der Quelle bevorzugt)."""
    s = (src_pix_fmt or "").lower()
    base_yuva = ["yuva420p", "yuva444p10le", "yuva444p"]
    if s.startswith("gbrap"):
        cands = [
            s,
            "gbrap12le",
            "gbrap10le",
            "gbrap",
            "rgba",
            "bgra",
            "argb",
        ] + base_yuva
    elif s.startswith(("rgba", "bgra", "argb")):
        base = (
            "rgba"
            if s.startswith("rgba")
            else ("bgra" if s.startswith("bgra") else "argb")
        )
        cands = [
            s,
            base,
            "bgra",
            "rgba",
            "argb",
            "gbrap12le",
            "gbrap10le",
            "gbrap",
        ] + base_yuva
    elif s.startswith("yuva"):
        cands = (
            [s]
            + [x for x in base_yuva if x != s]
            + ["gbrap12le", "gbrap10le", "gbrap", "rgba", "bgra", "argb"]
        )
    else:
        cands = [
            s,
            "gbrap12le",
            "gbrap10le",
            "gbrap",
            "rgba",
            "bgra",
            "argb",
        ] + base_yuva
    out: List[str] = []
    for fmt in cands:
        f = fmt.strip().lower()
        if f and f not in out:
            out.append(f)
    return out


def _pick_alpha_preserving_pix_fmt(
    encoder: str, src_pix_fmt: Optional[str]
) -> Optional[str]:
    if not src_pix_fmt:
        return None
    for fmt in _alpha_candidates_for(src_pix_fmt):
        if _encoder_supports_pix_fmt(encoder, fmt):
            return fmt
    return None


def _upgrade_to_10bit_like(pf: str) -> Optional[str]:
    """Versuche eine 10-bit-Variante mit gleicher Chroma-Subsampling-Familie zu finden."""
    p = (pf or "").lower()
    if p.startswith("gbrap"):
        return "gbrap10le"
    if "444" in p:
        return "yuv444p10le"
    if "422" in p:
        return "yuv422p10le"
    if "420" in p:
        return "yuv420p10le"
    return None


def _h264_profile_for_pix_fmt(pix_fmt: Optional[str]) -> Optional[str]:
    pf = (pix_fmt or "").lower()
    if pf.startswith("rgb"):
        return None
    if "444" in pf:
        return "high444"
    if "422" in pf:
        return "high422"
    if "420" in pf:
        return "high"
    return None


def _chroma_bucket(pix_fmt: Optional[str]) -> str:
    pf = (pix_fmt or "").lower()
    if pf.startswith("rgb") or pf == "gbr":
        return "rgb"
    if "444" in pf:
        return "444"
    if "422" in pf:
        return "422"
    if "420" in pf:
        return "420"
    return "other"


def _should_preserve_src_pix_fmt_in_lossless(
    container: str, encoder: str, src_pix_fmt: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """Entscheidet, ob wir bei *lossless* das Quell-Pixelformat 1:1 übernehmen."""
    if not src_pix_fmt:
        return (False, None)
    cont = (container or "").lower()
    fam = _encoder_family(encoder)
    buck = _chroma_bucket(src_pix_fmt)
    if not _encoder_supports_pix_fmt(encoder, src_pix_fmt):
        return (False, None)
    if cont in ("mkv", "matroska", "mov", "avi"):
        prof = _h264_profile_for_pix_fmt(src_pix_fmt) if fam == "h264" else None
        return (True, prof)
    if cont in ("mp4", "m4v"):
        if buck == "420":
            prof = _h264_profile_for_pix_fmt(src_pix_fmt) if fam == "h264" else None
            return (True, prof)
        return (False, None)
    if cont == "webm":
        return (buck == "420", None)
    prof = _h264_profile_for_pix_fmt(src_pix_fmt) if fam == "h264" else None
    return (True, prof)


def _default_pix_fmt_for_intra_codec(
    codec_key: str, *, prefer10: bool
) -> Optional[str]:
    k = _normalize_codec_key(codec_key or "")
    if k == "prores":
        return "yuv422p10le"  # 10-bit Standard
    if k in {"dnxhd", "dnxhr"}:
        return "yuv422p10le" if prefer10 else "yuv422p"
    if k == "jpeg2000":
        return "yuv444p10le" if prefer10 else "yuv444p"
    if k in {"png", "rawvideo"}:
        return "rgb24"  # ohne Alpha default
    if k in {"ffv1", "magicyuv"}:
        return "yuv422p10le" if prefer10 else "yuv422p"
    if k in {"huffyuv", "utvideo", "cineform"}:
        return "yuv422p"  # breit kompatibel
    if k == "mjpeg":
        return "yuvj422p"  # mjpeg-typisch
    if k == "hap":
        return "rgba"  # ohne Alpha-Flag wird später ggf. nachgeschoben
    return None


def _ensure_quality_defaults(cmd: List[str], container: str, codec_key: str) -> None:
    if _is_stream_copy(cmd) or _is_explicit_lossless(cmd):
        return
    enc = _active_encoder(
        cmd, _encoder_for_codec(_normalize_codec_key(codec_key) or codec_key)
    )
    fam = _encoder_family(enc)

    def _has(*flags) -> bool:
        return _cmd_contains_any(cmd, set(flags))

    if fam == "mpeg4":
        if not _has("-qscale:v", "-q:v", "-b:v", "-vb"):
            cmd += [
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
        if container.lower() in ("mp4", "m4v") and "-movflags" not in cmd:
            cmd += ["-movflags", "+faststart"]

    elif fam == "h264":
        if not _has("-crf", "-b:v", "-qp", "-x264-params"):
            cmd += ["-preset", "slow", "-crf", "18"]
        if not _has("-g"):
            cmd += ["-g", "240"]
        if not _has("-bf"):
            cmd += ["-bf", "2"]
        if not _has("-sc_threshold", "-sc-threshold", "-x264-params"):
            cmd += ["-sc_threshold", "0"]

    elif fam == "hevc":
        if not _has("-crf", "-b:v", "-qp", "-x265-params"):
            cmd += ["-preset", "medium", "-crf", "20"]
        if not _has("-g"):
            cmd += ["-g", "240"]
        if not _has("-bf"):
            cmd += ["-bf", "3"]

    elif fam == "vp9":
        if not _has("-lossless", "-crf", "-b:v"):
            cmd += ["-b:v", "0", "-crf", "30", "-row-mt", "1", "-cpu-used", "2"]

    elif fam == "vp8":
        if not _has("-crf", "-b:v"):
            enc2 = _active_encoder(cmd, "libvpx")
            if enc2 in ("libvpx", "libvpx-v8"):
                cmd += [
                    "-b:v",
                    "0",
                    "-crf",
                    "16",
                    "-deadline",
                    "good",
                    "-cpu-used",
                    "2",
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
                ]
            elif enc2.endswith("_vaapi"):
                cmd += ["-qp", "18", "-b:v", "0", "-g", "240"]

    elif fam == "av1":
        if not _has("-aom-params", "-crf", "-b:v", "-qp"):
            cmd += ["-b:v", "0", "-crf", "28", "-row-mt", "1"]

    elif fam == "mpeg2video":
        if not _has("-qscale:v", "-q:v", "-b:v", "-vb"):
            cmd += [
                "-qscale:v",
                "2",
                "-qmin",
                "1",
                "-qmax",
                "3",
                "-g",
                "15",
                "-bf",
                "2",
            ]

    elif fam == "jpeg2000":
        if not _has("-qscale:v", "-q:v"):
            qual = 20
            qscale = max(1, min(8, int(round((qual - 16) / 4.0)) + 1))
            cmd += ["-pred", "0", "-qscale:v", str(qscale)]
    elif fam == "prores":
        # Effektives Ziel-Pixelformat aus -pix_fmt oder -vf format=... lesen
        pf = (infer_target_pix_fmt_from_plan(cmd) or "").lower()

        # WICHTIG: Wenn das pix_fmt zu diesem Zeitpunkt noch nicht feststeht,
        #          KEIN Profil voreilig setzen (sonst blockieren wir 4444+Alpha).
        if not pf:
            return

        # Alpha-/4444-Indizien
        wants_4444 = pf.startswith(("yuva444", "gbrap")) or pf in {
            "rgba",
            "bgra",
            "argb",
        }

        if wants_4444:
            # Für 4444/Alpha zwingend prores_ks + Profil 4 (oder 5 für XQ)
            if enc != "prores_ks":
                _set_kv_arg(cmd, "-c:v", "prores_ks")

            # Falls -profile:v schon existiert und nicht 4/5 ist → auf 4 anheben
            toks = [str(t) for t in cmd]
            if "-profile:v" in toks:
                i = toks.index("-profile:v")
                if i + 1 < len(toks) and str(cmd[i + 1]) not in {"4", "5"}:
                    cmd[i + 1] = "4"
            else:
                _set_kv_arg(cmd, "-profile:v", "4")
        else:
            # Kein Alpha/4444 geplant → nur dann HQ (3) setzen,
            # wenn noch GAR KEINE Qualitätssteuerung existiert
            if not _has("-profile:v", "-qscale:v", "-b:v", "-vb"):
                _set_kv_arg(cmd, "-profile:v", "3")


# ──────────────────────────────────────────────────────────────────────────────
#  Komfort-Helfer für Aufrufer (optional)
# ──────────────────────────────────────────────────────────────────────────────


def _pixfmt_api_decision_for(
    *,
    cmd: List[str],
    container: str,
    codec_key: str,
    input_path: Path,
    encoder_hint: Optional[str] = None,
    preset_name: Optional[str] = None,
) -> Optional[PixfmtDecision]:
    """
    Baut einen PixfmtContext aus den vorhandenen Infos und ruft decide_pix_fmt(...).
    Fällt bei Problemen (Signature/Import) still auf None zurück – dann greift ein Fallback im Aufrufer.
    """
    try:
        # Quelle sondieren
        _w, _h, src_pf = _ffprobe_geometry(input_path)
        colors = probe_color_metadata(input_path) or {}

        # Aktiver Encoder (falls schon gesetzt), sonst Heuristik
        enc = encoder_hint or _active_encoder_or_guess(
            cmd, _normalize_codec_key(codec_key) or codec_key
        )

        # Pipeline-Kontext
        has_vf = any(flag in cmd for flag in ("-vf", "-filter:v"))
        scopy = _is_stream_copy(cmd)

        # Preset-Hinweise (falls vorhanden)
        prefer_10bit = False
        force_420 = False
        lossless = False
        try:
            spec = (
                _get_preset_spec(defin.CONVERT_PRESET, preset_name)
                if preset_name
                else {}
            )
            prefer_10bit = bool(spec.get("prefer_10bit", False))
            force_420 = bool(spec.get("force_yuv420p", False))
            lossless = bool(spec.get("lossless", False))
        except Exception:
            pass

        # HW-Familie aus Encoder ableiten
        hw_family = None
        e = (enc or "").lower()
        if e.endswith("_nvenc"):
            hw_family = "nvenc"
        elif "vaapi" in e:
            hw_family = "vaapi"
        elif "qsv" in e:
            hw_family = "qsv"

        ctx = PixfmtContext(
            container=(container or "").lower(),
            codec_key=_normalize_codec_key(codec_key or "") or (codec_key or ""),
            encoder_name=enc,
            preset=preset_name or None,
            src_pix_fmt=src_pf or "",
            color_meta=colors,
            wants_alpha=None,  # aus Quelle abgeleitet, falls vorhanden
            has_filtergraph=has_vf,
            is_stream_copy=scopy,
            is_explicit_lossless=lossless or _is_explicit_lossless(cmd),
            hw_family=hw_family,
            prefer_10bit=prefer_10bit,
            force_yuv420p=force_420,
        )
        return decide_pix_fmt(ctx)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  (Optionale) Alpha-Erzwingung für bestimmte Codecs – Legacy-Kompatibilität
# ──────────────────────────────────────────────────────────────────────────────


def _apply_alpha_rules(
    final_cmd: List[str],
    chosen_codec_key: str,
    container: str,
    src_pix_fmt: Optional[str],
    force_pix_fmt: Optional[str],
) -> Tuple[List[str], Optional[str]]:
    """
    Historischer Helper – in neuen Pipelines übernimmt die Pixfmt-API die Alpha-Entscheidung.
    Belassen als No-Op-Kompatibilitätsstub (greift nur bei klassischen Intraframe-Encodern).
    """
    # --- NEU: Alpha grundsätzlich unterbinden in Containern mit 4:2:0-Policy ---
    c = (container or "").lower()
    if c in {"mp4", "m4v", "webm", "mpeg"}:
        # evtl. zuvor gesetztes -pix_fmt (z. B. yuva***) sicher entfernen
        try:
            _strip_options(final_cmd, {"-pix_fmt"})
        except Exception:
            pass
        return final_cmd, force_pix_fmt

    # ---------------------------------------------------------------------------
    def _has_alpha(pf: Optional[str]) -> bool:
        p = (pf or "").lower()
        return any(tag in p for tag in ("rgba", "argb", "bgra", "gbrap", "ya8", "yuva"))

    if force_pix_fmt:
        return final_cmd, force_pix_fmt
    if not _has_alpha(src_pix_fmt):
        return final_cmd, force_pix_fmt

    key = _normalize_codec_key(chosen_codec_key or "") or ""
    enc_now = _active_encoder(final_cmd, _encoder_for_codec(key))

    # ProRes 4444
    if key == "prores":
        if _encoder_supports_pix_fmt(enc_now, "yuva444p10le"):
            # vorhandenes -profile:v auf "4" setzen (oder hinzufügen)
            try:
                j = final_cmd.index("-profile:v")
                if j + 1 < len(final_cmd):
                    final_cmd[j + 1] = "4"
                else:
                    final_cmd += ["-profile:v", "4"]
            except ValueError:
                # sauber vor dem Output einfügen
                out_idx = _output_index(final_cmd)
                final_cmd[out_idx:out_idx] = ["-profile:v", "4"]

            return final_cmd, "yuva444p10le"

    # HAP Alpha
    if key == "hap":
        if "-format" not in final_cmd:
            final_cmd += ["-format", "hap_alpha"]
        for pf in ("rgba", "bgra"):
            if _encoder_supports_pix_fmt(enc_now, pf):
                return final_cmd, pf

    # PNG / QTRLE / RAW / FFV1 / HuffYUV / UT / MagicYUV / CineForm
    if key in {
        "png",
        "qtrle",
        "rawvideo",
        "ffv1",
        "huffyuv",
        "utvideo",
        "magicyuv",
        "cineform",
    }:
        candidates_by_key: Dict[str, Tuple[str, ...]] = {
            "png": ("rgba", "bgra", "argb", "rgb24"),
            "qtrle": ("argb", "rgba", "bgra", "rgb24"),
            "rawvideo": ("rgba", "bgra", "argb"),
            "ffv1": ("gbrap10le", "rgba", "bgra", "argb"),
            "huffyuv": ("rgba", "bgra", "argb"),
            "utvideo": ("rgba", "bgra", "argb"),
            "magicyuv": ("gbrap10le", "rgba", "bgra", "argb"),
            "cineform": ("gbrap10le", "rgba", "bgra", "argb"),
        }
        for pf in candidates_by_key.get(key, ()):
            if _encoder_supports_pix_fmt(enc_now, pf):
                return final_cmd, pf

    # JPEG2000 Alpha
    if key == "jpeg2000":
        for pf in ("rgba", "yuva444p10le"):
            if _encoder_supports_pix_fmt(enc_now, pf):
                return final_cmd, pf

    if key == "av1":
        for pf in ("yuva444p10le", "yuva420p"):
            if _encoder_supports_pix_fmt(enc_now, pf):
                return final_cmd, pf

    return final_cmd, force_pix_fmt


def _codec_alpha_capability(codec_key: str) -> str:
    """
    'always'  → Codec ist grundsätzlich alpha-fähig (bei passendem pix_fmt/profil)
    'maybe'   → theoretisch möglich, aber nur in speziellen Profilen/Setups
    'no'      → praktisch kein Alpha
    """
    k = (_normalize_codec_key(codec_key or "") or "").lower()
    if k in _ALPHA_OK_CODECS:
        # ProRes nur mit 4444-Format tatsächlich  mit Alpha → 'maybe'
        return "maybe" if k == "prores" else "always"
    return "no"


def _container_alpha_capability(container: str) -> bool:
    return (container or "").lower() not in _ALPHA_DISABLED_CONTAINERS


def _detect_source_alpha(input_path: Path) -> tuple[bool, Optional[str]]:
    """Erkennt Alpha aus dem Quell-Pixelformat."""
    _w, _h, pf = _ffprobe_geometry(input_path)
    return (_pix_fmt_has_alpha(pf), pf)


def _assess_alpha_preservation(
    *,
    input_path: Path,
    target_container: str,
    codec_key: str,
    encoder_name: Optional[str] = None,
    stream_copy: bool = False,  # NEU: true bei -c:v copy
) -> AlphaCheck:
    """
    Prüft, ob Alpha realistisch erhalten werden kann.
    Bei stream_copy wird nur die Container-Fähigkeit geprüft; Encoder/Pixelformat entfallen.
    """
    src_has_alpha, src_pf = _detect_source_alpha(input_path)

    c_eff = (target_container or "").lower()
    cont_ok = _container_alpha_capability(c_eff)

    k_eff = _normalize_codec_key(codec_key or "") or ""
    cap = (
        _codec_alpha_capability(k_eff) if k_eff else "maybe"
    )  # unbekannt bei copy/keep ohne Src-Codec

    enc = (encoder_name or "") or _encoder_for_codec(k_eff) if k_eff else ""

    candidate_alpha_pfs: list[str] = (
        _alpha_candidates_for(src_pf) if src_has_alpha else []
    )

    # Encoder-Check nur relevant bei Re-Encode
    encoder_has_alpha_pf = False
    picked_pf: Optional[str] = None
    if src_has_alpha and candidate_alpha_pfs and not stream_copy and k_eff:
        for pf in candidate_alpha_pfs:
            if _encoder_supports_pix_fmt(enc, pf):
                encoder_has_alpha_pf = True
                picked_pf = pf
                break

    if stream_copy:
        # Bei Copy bleibt Pixelformat erhalten – Alpha hängt faktisch am Container (Policy)
        target_can_preserve = bool(src_has_alpha and cont_ok)
    else:
        target_can_preserve = bool(
            src_has_alpha and cont_ok and cap != "no" and encoder_has_alpha_pf
        )

    will_lose = bool(src_has_alpha and not target_can_preserve)

    reason: Optional[str] = None
    suggestions: list[tuple[str, str]] = []

    # --- Policy: bestimmte Codecs in MOV nicht vorschlagen ---
    _MOV_DENY = {"ffv1", "magicyuv"}

    def _policy_allows(container: str, codec: str) -> bool:
        return not (container == "mov" and codec in _MOV_DENY)

    seen: set[tuple[str, str]] = set()

    def add_suggestion(container: str, codec: str) -> None:
        key = (container, codec)
        if key in seen:
            return
        if _container_allows_codec(container, codec) and _policy_allows(
            container, codec
        ):
            suggestions.append(key)
            seen.add(key)

    # Preferred Alternativen (Reihenfolge beibehalten)
    alt_containers = ("mkv", "mov", "avi")
    alt_codecs = (
        "ffv1",
        "prores",
        "png",
        "qtrle",
        "utvideo",
        "magicyuv",
        "huffyuv",
        "rawvideo",
        "cineform",
        "hap",
    )

    if will_lose:
        if not cont_ok:
            reason = _("alpha_reason_container_no_alpha").format(container=c_eff)
            for alt_c in alt_containers:
                for alt_k in alt_codecs:
                    add_suggestion(alt_c, alt_k)

        elif not stream_copy and cap == "no":
            reason = _("alpha_reason_codec_no_alpha").format(codec=k_eff)
            for alt_k in alt_codecs:
                add_suggestion(c_eff, alt_k)

        elif not stream_copy and not encoder_has_alpha_pf:
            reason = _("alpha_reason_encoder_no_pf").format(
                encoder=enc, pix_fmt=src_pf or ""
            )
            for alt_c in alt_containers:
                for alt_k in alt_codecs:
                    add_suggestion(alt_c, alt_k)

    # ProRes: nur 4444 trägt Alpha – ggf. bevorzugtes Ziel-Pixelformat vorschlagen
    if (
        target_can_preserve
        and not stream_copy
        and k_eff == "prores"
        and picked_pf
        and not (picked_pf or "").startswith("yuva444")
    ):
        for pf in ("yuva444p10le", "yuva444p"):
            if _encoder_supports_pix_fmt(enc, pf):
                candidate_alpha_pfs = [pf] + [p for p in candidate_alpha_pfs if p != pf]
                break

    return AlphaCheck(
        src_has_alpha=src_has_alpha,
        target_can_preserve=target_can_preserve,
        will_lose_alpha=will_lose,
        reason=reason,
        suggestions=suggestions,
        candidate_pix_fmts=candidate_alpha_pfs if src_has_alpha else [],
    )
