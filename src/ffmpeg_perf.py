#!/usr/bin/env python3
# ffmpeg_perf.py
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import helpers as he

StrPath = Union[str, os.PathLike, Path]

# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PerfPolicy:
    # I/O
    min_thread_queue_size: int = 4096
    max_muxing_queue_size_per_input: int = 2048

    # AV1
    aom_cpu_used: Optional[int] = None
    prefer_svt_for_av1: bool = True

    # SVT
    svt_lp_from_threads: bool = True

    # HW-Filter Rewrites
    enable_cuda_filters: bool = True
    cuda_format: str = "nv12"

    # NEW: Auto-Promotion von CPU→GPU, wenn sinnvoll
    autopromote_gpu_encoders: bool = True
    # Profile/Modes, bei denen „Speed-first“ ist:
    autopromote_methods: tuple[str, ...] = (
        "messenger360p",
        "messenger720p",
        "web",
        "casual",
        "cinema",
        "studio",
    )
    # Lossless für h264/hevc via NVENC zulassen (NVENC kann lossless)
    autopromote_allow_lossless_h26x: bool = True
    autopromote_vp9: bool = False


# globaler Default (kannst du zur Laufzeit ersetzen)
_GLOBAL_POLICY = PerfPolicy()


def get_global_policy() -> PerfPolicy:
    return _GLOBAL_POLICY


# ──────────────────────────────────────────────────────────────────────────────
# System/FFmpeg Capabilities
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SysCaps:
    cores: int
    ram_gb: int
    has_cuda_hwaccel: bool
    has_nvenc: bool
    has_vaapi: bool
    has_qsv: bool
    encoders_text: str
    filters_text: str
    hwaccels_text: str


def _total_mem_gb() -> int:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // 1024 // 1024)
    except Exception:
        pass
    return 0


@lru_cache(maxsize=1)
def _ffmpeg_out(args: Tuple[str, ...]) -> str:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", *args], text=True, stderr=subprocess.STDOUT
        )
        return out
    except Exception:
        return ""


def _ff_has_token(txt: str, token: str) -> bool:
    return token in txt


def detect_syscaps() -> SysCaps:
    enc = _ffmpeg_out(("-encoders",))
    fil = _ffmpeg_out(("-filters",))
    hwa = _ffmpeg_out(("-hwaccels",))
    return SysCaps(
        cores=_cpu_affinity_count(),
        ram_gb=_total_mem_gb(),
        has_cuda_hwaccel=bool(re.search(r"(?m)^\s*cuda\s*$", hwa)),
        has_nvenc=("nvenc" in enc),
        has_vaapi=("vaapi" in (enc + fil + hwa)),
        has_qsv=("qsv" in (enc + fil + hwa)),
        encoders_text=enc,
        filters_text=fil,
        hwaccels_text=hwa,
    )


@lru_cache(maxsize=64)
def _ff_help_encoder(enc_name: str) -> str:
    """
    Gibt den Hilfetext des Encoders zurück (für Optionserkennung).
    """
    if not enc_name:
        return ""
    try:
        txt = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-h", f"encoder={enc_name}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return txt
    except Exception:
        return ""


def _encoder_supports_option(enc_name: Optional[str], opt_key: str) -> bool:
    """
    Prüft defensiv, ob im Encoder-Hilfetext eine Option (z. B. 'tile-columns', 'row-mt',
    'async_depth', 'surfaces') auftaucht.
    """
    if not enc_name or not opt_key:
        return False
    help_txt = _ff_help_encoder(enc_name)
    if not help_txt:
        return False
    # einfache Textsuche; ffmpeg listet " -option <desc>" oder "Options:" Aufzählungen
    patt = re.compile(rf"(?mi)^\s*{re.escape(opt_key)}\b")
    return bool(patt.search(help_txt))


# ──────────────────────────────────────────────────────────────────────────────
# Input/Command helpers
# ──────────────────────────────────────────────────────────────────────────────

_REMOTE_RE = re.compile(r"^(?:https?|rtmp|rtsp|srt|udp|tcp|s3|ftp)://", re.I)


def _extract_inputs(cmd: List[str]) -> List[str]:
    ins: List[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-i" and i + 1 < len(cmd):
            ins.append(str(cmd[i + 1]))
            i += 2
        else:
            i += 1
    return ins


def _any_remote(inputs: List[str]) -> bool:
    return any(_REMOTE_RE.match(x or "") for x in inputs)


def _vf_is_simple_scale(cmd: List[str]) -> bool:
    vf = _extract_vf(cmd)
    if not vf:
        return False
    return bool(_VF_SIMPLE_SCALE_RE.match(vf))


def _is_10bit_requested(cmd: List[str]) -> bool:
    # pix_fmt 10-bit?
    pf = _get_opt_value(cmd, "-pix_fmt")
    if pf and "10" in pf:
        return True
    # profile main10?
    prof = _get_opt_value(cmd, "-profile:v") or _get_opt_value(cmd, "-profile")
    if prof and "10" in prof:
        return True
    # x265-params profile=main10
    x265 = _get_opt_value(cmd, "-x265-params")
    if x265 and re.search(r"(?:^|:)profile\s*=\s*main10(?:$|:)", x265):
        return True
    # AV1 10-bit pix_fmt via -vf format=...?
    vf = _extract_vf(cmd) or ""
    if "p010" in vf or "yuv420p10" in vf:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Policy decision
# ──────────────────────────────────────────────────────────────────────────────


def decide_policy(
    cmd: List[str],
    *,
    method: Optional[
        str
    ] = None,  # z.B. "web", "casual", "archival", "lossless", "proxy", ...
    src_w: Optional[int] = None,
    src_h: Optional[int] = None,
) -> PerfPolicy:
    caps = detect_syscaps()
    inputs = _extract_inputs(cmd)
    remote = _any_remote(inputs)
    enc = _active_encoder_from_cmd(cmd)
    fam = _encoder_family(enc)
    simple_vf = _vf_is_simple_scale(cmd)
    tenbit = _is_10bit_requested(cmd)

    # 1) I/O-Queues
    if remote or len(inputs) >= 2 or "-filter_complex" in cmd:
        min_tqs = 8192
        mux_per_input = 4096
    else:
        min_tqs = 4096
        mux_per_input = 2048

    # 2) CUDA-Filter-Rewrite Eligibility
    has_scale_cuda = _ff_has_token(caps.filters_text, "scale_cuda")
    enable_cuda_filters = (
        simple_vf
        and has_scale_cuda
        and (
            _is_nvenc(enc)
            or (
                caps.has_cuda_hwaccel
                and _has_key_with_value(cmd, "-hwaccel")
                and (_get_opt_value(cmd, "-hwaccel") or "").lower() == "cuda"
            )
        )
    )
    cuda_fmt = (
        "p010le"
        if (enable_cuda_filters and tenbit)
        else (get_global_policy().cuda_format or "nv12")
    )

    # 3) AV1 Entscheidungen
    method_lc = (method or "").lower()
    if method_lc in {"messenger360p", "messenger720p", "web", "casual", "cinema"}:
        aom_cu = 6 if caps.cores >= 16 else (5 if caps.cores >= 8 else 4)
    elif method_lc in {"studio", "ultra", "lossless"}:
        aom_cu = 4
    else:
        aom_cu = 5 if caps.cores >= 8 else 4

    prefer_svt = True if caps.cores >= 8 else False

    return PerfPolicy(
        min_thread_queue_size=min_tqs,
        max_muxing_queue_size_per_input=mux_per_input,
        aom_cpu_used=aom_cu if fam == "av1" else None,
        prefer_svt_for_av1=prefer_svt,
        svt_lp_from_threads=True,
        enable_cuda_filters=enable_cuda_filters,
        cuda_format=cuda_fmt,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ffprobe helpers
# ──────────────────────────────────────────────────────────────────────────────


def probe_resolution(path: StrPath, *, apply_rotation: bool = True) -> Tuple[int, int]:
    p = str(Path(path))
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,side_data_list",
                "-of",
                "json",
                p,
            ],
            text=True,
        )
        data = json.loads(out)
        streams = data.get("streams") or []
        if not streams:
            return (0, 0)
        s = streams[0]
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        if apply_rotation:
            sdl = s.get("side_data_list") or []
            for sd in sdl:
                if sd.get(
                    "side_data_type", ""
                ).lower() == "displaymatrix" and isinstance(
                    sd.get("rotation"), (int, float)
                ):
                    rot = int(sd["rotation"]) % 360
                    if rot in (90, 270):
                        w, h = h, w
                    break
        return (max(0, w), max(0, h))
    except Exception:
        return (0, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Policy – Erweiterung
# ──────────────────────────────────────────────────────────────────────────────


def _encoders_available(caps: SysCaps) -> set[str]:
    txt = caps.encoders_text or ""
    names = set()
    for line in txt.splitlines():
        # Zeilenform: " V..... h264_nvenc ..." → Name = letztes Wort vor Beschreibung
        m = re.search(r"^\s*[A-Z\.]+\s+([a-z0-9_]+)\b", line)
        if m:
            names.add(m.group(1).strip().lower())
    return names


def _has_any_keys(cmd: List[str], keys: tuple[str, ...]) -> bool:
    s = set(cmd)
    return any(k in s for k in keys)


def _choose_gpu_encoder_for_family(fam: str, avail: set[str]) -> Optional[str]:
    # Bevorzugungsreihenfolge je Familie
    if fam == "h264":
        for cand in ("h264_nvenc", "h264_qsv", "h264_vaapi", "h264_amf"):
            if cand in avail:
                return cand
    if fam == "hevc":
        for cand in ("hevc_nvenc", "hevc_qsv", "hevc_vaapi", "hevc_amf"):
            if cand in avail:
                return cand
    if fam == "av1":
        for cand in ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi"):
            if cand in avail:
                return cand
    if fam == "vp9":
        # Keine echte "GPU"-Promotion für VP9. Wenn libvpx-vp9 vorhanden ist,
        # bleib dabei (No-op). QSV/VAAPI NICHT automatisch wählen.
        return "libvpx-vp9" if "libvpx-vp9" in avail else None
    return None


@lru_cache(maxsize=2)
def _nvenc_supported_presets_for(enc_name: str = "h264_nvenc") -> set[str]:
    try:
        txt = _ff_help_encoder(enc_name) or ""
        # ffmpeg zeigt oft sowas wie: "preset     <...>  (one of: default|p1|p2|...|slow|...|lossless)"
        m = re.search(r"(?mi)^\s*preset\s+.*?\(.*?:\s*([^)]+)\)", txt)
        if m:
            items = [x.strip().lower() for x in m.group(1).split("|") if x.strip()]
            return set(items)
    except Exception:
        pass
    # robuster Fallback – existiert praktisch überall
    return {
        "default",
        "slow",
        "medium",
        "fast",
        "hp",
        "hq",
        "bd",
        "ll",
        "llhq",
        "llhp",
        "lossless",
        "losslesshp",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
    }


_GENERIC_TO_NVENC_MAP: dict[str, list[str]] = {
    "ultrafast": ["p1", "llhp", "fast"],
    "superfast": ["p2", "fast"],
    "veryfast": ["p3", "fast"],
    "faster": ["p4", "medium"],
    "fast": ["p5", "hq"],
    "medium": ["p6", "hq"],
    "slow": ["p7", "slow"],
    "slower": ["p7", "slow"],
    "veryslow": ["p7", "slow"],
    "placebo": ["p7", "slow"],
    "lossless": ["lossless", "losslesshp", "p7"],
}


def _sanitize_nvenc_preset_inplace(cmd: list[str], enc_name: str) -> None:
    enc = (enc_name or "").lower()
    if not enc.endswith("_nvenc"):
        return
    if "-preset" not in cmd:
        return
    i = cmd.index("-preset")
    if i + 1 >= len(cmd):
        return
    raw = str(cmd[i + 1]).strip().lower()
    supp = _nvenc_supported_presets_for(enc)

    # bereits gültig → fertig
    if raw in supp:
        return

    # mappe x264-Style → NVENC
    for cand in _GENERIC_TO_NVENC_MAP.get(raw, []):
        if cand in supp:
            cmd[i + 1] = cand
            return

    # Fallbacks
    if "p4" in supp:
        cmd[i + 1] = "p4"
        return
    if "fast" in supp:
        cmd[i + 1] = "fast"
        return

    # letzten Ausweg: -preset entfernen → NVENC Default
    del cmd[i : i + 2]


def _sanitize_nvenc_ratecontrol_inplace(cmd: list[str]) -> None:
    """Optional: CRF → CQ für NVENC, wenn vorhanden."""
    # nur wenn *_nvenc aktiv
    try:
        j = cmd.index("-c:v")
        enc_now = str(cmd[j + 1]).lower() if j + 1 < len(cmd) else ""
        if not enc_now.endswith("_nvenc"):
            return
    except ValueError:
        return

    # -crf N → ersetze durch -cq N (NVENC) + setze rc falls nicht vorhanden
    if "-crf" in cmd:
        k = cmd.index("-crf")
        if k + 1 < len(cmd) and not str(cmd[k + 1]).startswith("-"):
            val = str(cmd[k + 1])
            # ersetze
            cmd[k : k + 2] = ["-cq", val]
            # rc setzen, falls nicht vorhanden (vorsichtig)
            if "-rc" not in cmd and "-rc:v" not in cmd:
                _insert_before_output(cmd, ["-rc", "vbr"])


def _maybe_promote_encoder_to_gpu(
    cmd: List[str], *, method: Optional[str], caps: SysCaps, pol: PerfPolicy
) -> None:
    if not pol.autopromote_gpu_encoders:
        return

    # Aktuellen Encoder ermitteln
    enc = _active_encoder_from_cmd(cmd)
    if not enc:
        return
    e_lc = enc.lower()

    # Schon HW-Encoder? → raus
    if any(e_lc.endswith(suf) for suf in ("_nvenc", "_qsv", "_vaapi", "_amf")):
        return

    fam = _encoder_family(enc)
    if fam not in ("h264", "hevc", "av1", "vp9"):
        return
    if fam == "vp9" and not pol.autopromote_vp9:
        return

    method_lc = (method or "").lower()

    # Nur bei "Speed-first"-Profilen automatisch anheben
    allowed = (not method) or (method_lc in pol.autopromote_methods)

    # Für AV1: lossless nicht automatisch anheben (Qualitäts-/Feature-Parität)
    if fam == "av1" and method_lc in ("lossless", "ultra", "studio"):
        allowed = False

    # Für H.264/HEVC: lossless erlaubt, aber toggelbar
    if (
        fam in ("h264", "hevc")
        and method_lc == "lossless"
        and not pol.autopromote_allow_lossless_h26x
    ):
        allowed = False

    if not allowed:
        return

    # Inkompatible Param-Flags → lieber nicht anfassen
    if fam == "hevc" and _has_any_keys(cmd, ("-x265-params",)):
        return
    if fam == "h264" and _has_any_keys(cmd, ("-x264-params",)):
        return
    if fam == "av1" and _has_any_keys(cmd, ("-svtav1-params",)):
        return

    avail = _encoders_available(caps)
    gpu_enc = _choose_gpu_encoder_for_family(fam, avail)
    if not gpu_enc:
        return

    # -c:v ersetzen (nur Wert austauschen)
    try:
        i = cmd.index("-c:v")
        if i + 1 < len(cmd):
            cmd[i + 1] = gpu_enc
    except ValueError:
        # Wenn kein -c:v da ist, vor Output setzen
        _insert_before_output(cmd, ["-c:v", gpu_enc])

    if gpu_enc.endswith("_nvenc"):
        _sanitize_nvenc_preset_inplace(cmd, gpu_enc)
        _sanitize_nvenc_ratecontrol_inplace(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# Mini-Utils
# ──────────────────────────────────────────────────────────────────────────────


def _output_index(cmd: List[str]) -> int:
    if not cmd:
        return 0
    last = str(cmd[-1])
    if not last.startswith("-"):
        # heuristische Erkennung eines "echten" Outputs (Pfad/Name)
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?[kKmM]?", last):
            return len(cmd)
        if re.fullmatch(r"\d+/\d+", last):
            return len(cmd)
        if ("/" in last) or ("\\" in last):
            return len(cmd) - 1
        if "." in last and not last.startswith("."):
            return len(cmd) - 1
    return len(cmd)


def _insert_before_output(cmd: List[str], tokens: List[str]) -> None:
    idx = _output_index(cmd)
    cmd[idx:idx] = tokens


def _insert_before_each(cmd: List[str], key: str, maker_tokens) -> None:
    i = 0
    while i < len(cmd):
        if cmd[i] == key:
            tokens = list(maker_tokens())
            if tokens and not (i > 0 and cmd[i - 1] == tokens[0]):
                cmd[i:i] = tokens
                i += len(tokens)
        i += 1


def _has_flag(cmd: List[str], key: str) -> bool:
    try:
        i = cmd.index(key)
        if i + 1 < len(cmd) and str(cmd[i + 1]).startswith("-"):
            return True
        return True
    except ValueError:
        return False


def _has_key_with_value(cmd: List[str], key: str) -> bool:
    try:
        i = cmd.index(key)
        return (i + 1 < len(cmd)) and (not str(cmd[i + 1]).startswith("-"))
    except ValueError:
        return False


def _get_opt_value(cmd: List[str], key: str) -> Optional[str]:
    try:
        i = cmd.index(key)
        if i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("-"):
            return str(cmd[i + 1])
    except ValueError:
        pass
    return None


def _count_inputs(cmd: List[str]) -> int:
    return sum(1 for x in cmd if x == "-i")


def _active_encoder_from_cmd(cmd: List[str]) -> Optional[str]:
    try:
        i = cmd.index("-c:v")
        return str(cmd[i + 1]).lower()
    except ValueError:
        return None


def _encoder_family(enc: Optional[str]) -> str:
    e = (enc or "").lower()
    if (
        e in ("libx264",)
        or e.startswith("h264_")
        or (e.endswith("_nvenc") and "h264" in e)
    ):
        return "h264"
    if (
        e in ("libx265",)
        or e.startswith("hevc_")
        or (e.endswith("_nvenc") and "hevc" in e)
    ):
        return "hevc"
    if (
        e in ("libaom-av1", "libsvtav1")
        or e.startswith("av1_")
        or e.endswith("av1_nvenc")
    ):
        return "av1"
    if e in ("libvpx-vp9", "vp9", "vp9_qsv", "vp9_vaapi"):
        return "vp9"
    if e in ("libvpx", "libvpx-v8", "vp8", "vp8_qsv", "vp8_vaapi"):
        return "vp8"
    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# Filter/Größe analysieren
# ──────────────────────────────────────────────────────────────────────────────


def _extract_vf(cmd: List[str]) -> Optional[str]:
    if "-vf" in cmd:
        try:
            return str(cmd[cmd.index("-vf") + 1])
        except Exception:
            return None
    if "-filter:v" in cmd:
        try:
            return str(cmd[cmd.index("-filter:v") + 1])
        except Exception:
            return None
    return None


def _eval_dim(expr: str, src_w: int, src_h: int) -> Optional[int]:
    s = (expr or "").strip().lower()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s == "iw":
        return src_w
    if s == "ih":
        return src_h
    m = re.match(r"trunc\(\s*i([wh])\s*/\s*(\d+)\s*\)\s*\*\s*(\d+)", s)
    if m:
        which = m.group(1)
        n1 = int(m.group(2))
        n2 = int(m.group(3))
        base = src_w if which == "w" else src_h
        n = n1 if n1 == n2 else n1
        if n > 0:
            return (base // n) * n
    return None


def _infer_output_wh(cmd: List[str], src_w: int, src_h: int) -> Tuple[int, int]:
    vf = _extract_vf(cmd)
    if not vf:
        return (src_w, src_h)

    m = (
        re.search(r"(?:^|,)scale\s*=\s*([^:,]+)\s*:\s*([^,]+)(?:,|$)", vf)
        or re.search(r"(?:^|,)scale_vaapi\s*=\s*([^:,]+)\s*:\s*([^,]+)(?:,|$)", vf)
        or re.search(r"(?:^|,)scale_cuda\s*=\s*([^:,]+)\s*:\s*([^,]+)(?:,|$)", vf)
    )
    if m:
        w = _eval_dim(m.group(1), src_w, src_h)
        h = _eval_dim(m.group(2), src_w, src_h)
        if w and h:
            return (w, h)

    m = re.search(r"(?:^|,)zscale\s*=\s*([^,]+)(?:,|$)", vf)
    if m:
        kw: Dict[str, str] = {}
        for part in m.group(1).split(":"):
            if "=" in part:
                k, v = part.split("=", 1)
                kw[k.strip().lower()] = v.strip()
        w = _eval_dim(kw.get("width", ""), src_w, src_h) if "width" in kw else None
        h = _eval_dim(kw.get("height", ""), src_w, src_h) if "height" in kw else None
        if w and h:
            return (w, h)

    return (src_w, src_h)


# ──────────────────────────────────────────────────────────────────────────────
# Ressourcen-/Tile-Heuristik
# ──────────────────────────────────────────────────────────────────────────────


def _cpu_affinity_count() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respektiert cgroups/affinity
    except Exception:
        return int(os.cpu_count() or 1)


def _free_cores_estimate(total: int) -> int:
    """sehr grob: freie Kerne ≈ max(1, total - load1)"""
    try:
        load1, _, _ = os.getloadavg()
        free = int(round(total - load1))
        return max(1, min(total, free))
    except Exception:
        return total


def _thread_budget() -> int:
    total = _cpu_affinity_count()
    free = _free_cores_estimate(total)
    # leichte Über-Subscription, hilft bei I/O/Filter-Stalls
    tgt = max(total, free * 2)
    return max(1, min(256, tgt))


def _choose_tiles_for_target(w: int, h: int, target: int, fam: str) -> Tuple[int, int]:
    max_c = 6
    if fam == "av1":
        max_r = 4
    elif fam == "vp9":
        max_r = 2
    else:
        max_r = 3

    col = max(0, min(max_c, int(math.log2(max(1, w // 320))) if w >= 320 else 0))
    row = max(0, min(max_r, int(math.log2(max(1, h // 180))) if h >= 180 else 0))

    def tiles(c, r):
        return (1 << c) * (1 << r)

    want = max(1, min(target, (1 << max_c) * (1 << max_r)))

    while tiles(col, row) < want and (col < max_c or row < max_r):
        if (col <= row and col < max_c) or row >= max_r:
            col += 1
        else:
            row += 1

    # an Auflösung clampen (Superblock-basiert)
    sb = 128 if fam == "av1" else 64

    def _cap_log2(pixels: int, sb: int) -> int:
        sbs = max(1, (pixels + sb - 1) // sb)  # ceil(pixels / sb)
        return max(0, int(math.floor(math.log2(sbs))))  # log2(#SB)

    col_cap = _cap_log2(w, sb)
    row_cap = _cap_log2(h, sb)

    col = min(col, col_cap)
    row = min(row, row_cap, max_r)

    return col, row


def _choose_tiles_hevc(w: int, h: int, target: int) -> Tuple[int, int]:
    # x265: analog, rows etwas großzügiger (bis 4)
    max_c, max_r = 6, 4
    col = max(0, min(max_c, int(math.log2(max(1, w // 320))) if w >= 320 else 0))
    row = max(0, min(max_r, int(math.log2(max(1, h // 180))) if h >= 180 else 0))

    def tiles(c, r):
        return (1 << c) * (1 << r)

    want = max(1, min(target, (1 << max_c) * (1 << max_r)))
    while tiles(col, row) < want and (col < max_c or row < max_r):
        if (col <= row and col < max_c) or row >= max_r:
            col += 1
        else:
            row += 1
    return col, row


# ──────────────────────────────────────────────────────────────────────────────
# Param-Merger / Flags
# ──────────────────────────────────────────────────────────────────────────────


def _append_param_kv(cmd: List[str], flag: str, kv: dict[str, str]) -> None:
    try:
        i = cmd.index(flag)
        cur = str(cmd[i + 1])
    except ValueError:
        i = -1
        cur = ""

    parsed: Dict[str, str] = {}
    if cur:
        for part in cur.split(":"):
            if "=" in part:
                k, v = part.split("=", 1)
                parsed[k.strip()] = v.strip()

    changed = False
    for k, v in kv.items():
        if k not in parsed:
            parsed[k] = v
            changed = True

    if not changed:
        return
    newval = ":".join(f"{k}={v}" for k, v in parsed.items())
    if i >= 0:
        cmd[i + 1] = newval
    else:
        _insert_before_output(cmd, [flag, newval])


def _is_qsv(enc: Optional[str]) -> bool:
    return bool(enc and enc.endswith("_qsv"))


def _is_vaapi(enc: Optional[str]) -> bool:
    return bool(enc and enc.endswith("_vaapi"))


def _is_libaom(enc: Optional[str]) -> bool:
    return (enc or "").lower() == "libaom-av1"


def _is_svtav1(enc: Optional[str]) -> bool:
    return (enc or "").lower() == "libsvtav1"


def _is_nvenc(enc: Optional[str]) -> bool:
    return bool(enc and enc.endswith("_nvenc"))


# ──────────────────────────────────────────────────────────────────────────────
# VF-Rewriter (CUDA/VAAPI/QSV)
# ──────────────────────────────────────────────────────────────────────────────

_VF_SIMPLE_SCALE_RE = re.compile(
    r"^\s*(?:format\s*=\s*[^,]+,)?\s*scale\s*=\s*([^:,]+)\s*:\s*([^,]+)\s*(?:,\s*format\s*=\s*[^,]+)?\s*$",
    re.IGNORECASE,
)


def _replace_vf(cmd: List[str], new_vf: str) -> None:
    for key in ("-vf", "-filter:v"):
        if key in cmd:
            i = cmd.index(key)
            if i + 1 < len(cmd):
                cmd[i + 1] = new_vf
            else:
                cmd[i + 1 : i + 1] = [new_vf]
            return
    _insert_before_output(cmd, ["-vf", new_vf])


def _maybe_rewrite_vf_for_cuda(
    cmd: List[str], enc: Optional[str], policy: PerfPolicy
) -> None:
    """
    Ersetzt *nur* sehr einfache Ketten (format?,scale,format?) → hwupload_cuda,scale_cuda,(format?)
    Greift nur bei NVENC und wenn policy.enable_cuda_filters=True.
    """
    if not policy.enable_cuda_filters or not _is_nvenc(enc):
        return
    vf = _extract_vf(cmd)
    if not vf:
        return
    m = _VF_SIMPLE_SCALE_RE.match(vf)
    if not m:
        return
    caps = detect_syscaps()
    if not _ff_has_token(caps.filters_text, "scale_cuda"):
        return

    scale_w, scale_h = m.group(1).strip(), m.group(2).strip()
    tenbit = _is_10bit_requested(cmd)
    cuda_fmt = (
        "p010le"
        if tenbit
        else (policy.cuda_format.strip() if policy.cuda_format else "nv12")
    )

    new_chain = f"hwupload_cuda,scale_cuda={scale_w}:{scale_h},format={cuda_fmt}"
    _replace_vf(cmd, new_chain)

    # Zero-Copy sicherstellen, falls nichts gesetzt
    if not _has_key_with_value(cmd, "-hwaccel"):

        def _cuda_hw_tokens():
            return [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-extra_hw_frames",
                "16",
            ]

        _insert_before_each(cmd, "-i", _cuda_hw_tokens)


def _maybe_rewrite_vf_for_vaapi(cmd: List[str], enc: Optional[str]) -> None:
    if not _is_vaapi(enc):
        return
    vf = _extract_vf(cmd)
    if not vf:
        return
    m = _VF_SIMPLE_SCALE_RE.match(vf)
    if not m:
        return
    caps = detect_syscaps()
    if not _ff_has_token(caps.filters_text, "scale_vaapi"):
        return

    scale_w, scale_h = m.group(1).strip(), m.group(2).strip()
    tenbit = _is_10bit_requested(cmd)
    fmt = "p010" if tenbit else "nv12"

    new_chain = f"hwupload,scale_vaapi={scale_w}:{scale_h},format={fmt}"
    _replace_vf(cmd, new_chain)

    if not _has_key_with_value(cmd, "-hwaccel"):

        def _vaapi_hw_tokens():
            return [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-extra_hw_frames",
                "16",
            ]

        _insert_before_each(cmd, "-i", _vaapi_hw_tokens)


def _maybe_rewrite_vf_for_qsv(cmd: List[str], enc: Optional[str]) -> None:
    if not _is_qsv(enc):
        return
    vf = _extract_vf(cmd)
    if not vf:
        return
    m = _VF_SIMPLE_SCALE_RE.match(vf)
    if not m:
        return
    caps = detect_syscaps()
    # prefer scale_qsv (alternativ ginge auch vpp_qsv)
    if not _ff_has_token(caps.filters_text, "scale_qsv"):
        return

    scale_w, scale_h = m.group(1).strip(), m.group(2).strip()
    tenbit = _is_10bit_requested(cmd)
    fmt = "p010" if tenbit else "nv12"

    new_chain = f"scale_qsv=w={scale_w}:h={scale_h}:format={fmt}"
    _replace_vf(cmd, new_chain)

    if not _has_key_with_value(cmd, "-hwaccel"):

        def _qsv_hw_tokens():
            return [
                "-hwaccel",
                "qsv",
                "-hwaccel_output_format",
                "qsv",
                "-extra_hw_frames",
                "16",
            ]

        _insert_before_each(cmd, "-i", _qsv_hw_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Finales Tuning
# ──────────────────────────────────────────────────────────────────────────────


def tune_final_cmd(
    cmd: List[str],
    *,
    src_w: Optional[int] = None,
    src_h: Optional[int] = None,
    policy: Optional[PerfPolicy] = None,
) -> List[str]:
    """
    Aggressive, aber defensive Parallelisierung:
      • threads/filter_threads/filter_complex_threads
      • Tiles/Row-MT (VP9/AV1) – nur wenn Encoder-Optionen vorhanden sind
      • x265: WPP/PME/PMODE + dynamische Tiles via -x265-params
      • libx264: sliced-threads=1
      • NVENC/VAAPI/QSV: Zero-Copy & Scale-Rewrites; NVENC surfaces, QSV async_depth
      • Muxing-Queue
    Bestehende Flags werden nie überschrieben – nur ergänzt.
    """
    pol = policy or get_global_policy()
    if not isinstance(cmd, list) or not cmd:
        return cmd

    # 0) Log/StdIn
    if not _has_key_with_value(cmd, "-loglevel"):
        _insert_before_output(cmd, ["-loglevel", "warning"])
    if not _has_flag(cmd, "-hide_banner"):
        _insert_before_output(cmd, ["-hide_banner"])
    if not _has_flag(cmd, "-nostdin"):
        _insert_before_output(cmd, ["-nostdin"])

    # 0.5) Input-Queues vor JEDEM -i
    n_inputs = _count_inputs(cmd)

    def _tq_tokens():
        size = str(min(16384, max(pol.min_thread_queue_size, 1024 * max(1, n_inputs))))
        return ["-thread_queue_size", size]

    if n_inputs > 0:
        _insert_before_each(cmd, "-i", _tq_tokens)

    # 0.6) OPTIONAL: Encoder CPU→GPU auto-promoten (sicher & konservativ)
    caps = detect_syscaps()
    _maybe_promote_encoder_to_gpu(
        cmd, method=os.environ.get("PF_METHOD"), caps=caps, pol=pol
    )

    # 1) Threads – bewusst nicht drosseln
    T = _thread_budget()
    if not _has_key_with_value(cmd, "-threads"):
        _insert_before_output(cmd, ["-threads", str(T)])
    if not _has_key_with_value(cmd, "-filter_threads"):
        _insert_before_output(cmd, ["-filter_threads", str(T)])
    if "-filter_complex" in cmd and not _has_key_with_value(
        cmd, "-filter_complex_threads"
    ):
        _insert_before_output(cmd, ["-filter_complex_threads", str(min(T, 64))])

    # 2) Codec/Familie & Zielgröße
    enc = _active_encoder_from_cmd(cmd)
    fam = _encoder_family(enc)

    W = int(src_w or 1920)
    H = int(src_h or 1080)
    W, H = _infer_output_wh(cmd, W, H)

    # 2a) VP9: Row-MT + Frame-Parallel + Tiles (falls Optionen existieren)
    if fam == "vp9":
        # Manche Builds erwarten -row-mt, andere setzen it implizit → nur setzen, wenn vorhanden
        if _encoder_supports_option(enc, "row-mt") and not _has_key_with_value(
            cmd, "-row-mt"
        ):
            _insert_before_output(cmd, ["-row-mt", "1"])
        cols, rows = _choose_tiles_for_target(W, H, T, fam)
        if (
            cols > 0
            and not _has_key_with_value(cmd, "-tile-columns")
            and not _has_key_with_value(cmd, "-tile-columns")
        ):
            _insert_before_output(cmd, ["-tile-columns", str(cols)])
        if (
            rows > 0
            and not _has_key_with_value(cmd, "-tile-rows")
            and not _has_key_with_value(cmd, "-til-rows")
        ):
            _insert_before_output(cmd, ["-tile-rows", str(rows)])
        if _encoder_supports_option(enc, "frame-parallel") and not _has_key_with_value(
            cmd, "-frame-parallel"
        ):
            _insert_before_output(cmd, ["-frame-parallel", "1"])

    # 2b) AV1 – libaom: Tiles + (optional) row-mt
    if _is_libaom(enc):
        cols, rows = _choose_tiles_for_target(W, H, T, "av1")
        if cols > 0 and not _has_key_with_value(cmd, "-tile-columns"):
            _insert_before_output(cmd, ["-tile-columns", str(cols)])
        if rows > 0 and not _has_key_with_value(cmd, "-tile-rows"):
            _insert_before_output(cmd, ["-tile-rows", str(rows)])
        # cpu-used aus Policy
        if pol.aom_cpu_used is not None and not _has_key_with_value(cmd, "-cpu-used"):
            _insert_before_output(cmd, ["-cpu-used", str(pol.aom_cpu_used)])
        # row-mt (nur wenn vom Build unterstützt)
        if _encoder_supports_option("libaom-av1", "row-mt") and not _has_key_with_value(
            cmd, "-row-mt"
        ):
            _insert_before_output(cmd, ["-row-mt", "1"])

    # 2c) SVT-AV1 – -lp = Threads (falls nicht gesetzt)
    if (
        _is_svtav1(enc)
        and pol.svt_lp_from_threads
        and not _has_key_with_value(cmd, "-lp")
    ):
        _insert_before_output(cmd, ["-lp", str(T)])
        # (optional) Tiles für SVT-AV1 – nur wenn Option existiert
        if _encoder_supports_option(
            "libsvtav1", "tile-columns"
        ) and not _has_key_with_value(cmd, "-tile-columns"):
            c, r = _choose_tiles_for_target(W, H, max(1, T // 2), "av1")
            _insert_before_output(cmd, ["-tile-columns", str(c)])
            if _encoder_supports_option(
                "libsvtav1", "tile-rows"
            ) and not _has_key_with_value(cmd, "-tile-rows"):
                _insert_before_output(cmd, ["-tile-rows", str(r)])

    # 2d) x265 – parallele Defaults + Tiles
    if (enc or "").lower() == "libx265":
        ft = max(1, min(16, max(1, T // 2)))
        params = {
            "pools": str(T),
            "frame-threads": str(ft),
            "pmode": "1",
            "pme": "1",
            "wpp": "1",
        }
        want_tiles = max(1, T // max(1, ft))
        c, r = _choose_tiles_hevc(W, H, want_tiles)
        if c > 0:
            params["tile-columns"] = str(c)
        if r > 0:
            params["tile-rows"] = str(r)
        _append_param_kv(cmd, "-x265-params", params)

    # 2e) x264 – sliced-threads ergänzen
    if (enc or "").lower() == "libx264":
        _append_param_kv(cmd, "-x264-params", {"sliced-threads": "1"})

    # 2f) GPU Encoder – Zero-Copy + Backend-Spezialitäten
    if _is_nvenc(enc):
        has_vf = ("-vf" in cmd) or ("-filter:v" in cmd) or ("-filter_complex" in cmd)
        if not has_vf and not _has_key_with_value(cmd, "-hwaccel"):

            def _cuda_hw_tokens():
                return [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_output_format",
                    "cuda",
                    "-extra_hw_frames",
                    "16",
                ]

            _insert_before_each(cmd, "-i", _cuda_hw_tokens)
        _maybe_rewrite_vf_for_cuda(cmd, enc, pol)

        if _encoder_supports_option(enc, "surfaces") and not _has_key_with_value(
            cmd, "-surfaces"
        ):
            surf = str(
                min(
                    64,
                    max(16, (_thread_budget() // 2) + (8 if max(W, H) >= 2160 else 0)),
                )
            )
            _insert_before_output(cmd, ["-surfaces", surf])

    if _is_vaapi(enc):
        _maybe_rewrite_vf_for_vaapi(cmd, enc)
        # VAAPI hat keine einheitliche "async_depth"-Option im Encoder – skip

    if _is_qsv(enc):
        _maybe_rewrite_vf_for_qsv(cmd, enc)
        # QSV async_depth → Pipeline-Parallelität; nur setzen, wenn unterstützt
        if _encoder_supports_option(enc, "async_depth") and not _has_key_with_value(
            cmd, "-async_depth"
        ):
            depth = str(min(16, max(4, T // 2)))
            _insert_before_output(cmd, ["-async_depth", depth])

    # AMF: kein spezielles Rewrite notwendig; Threads/Queues greifen global

    # 3) Muxing-Queue großzügig
    if not _has_key_with_value(cmd, "-max_muxing_queue_size"):
        mq = str(min(16384, pol.max_muxing_queue_size_per_input * max(1, n_inputs)))
        _insert_before_output(cmd, ["-max_muxing_queue_size", mq])

    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Autotune & Runner
# ──────────────────────────────────────────────────────────────────────────────


def autotune_final_cmd(input_file, cmd: List[str]) -> List[str]:
    # Breite Default-Heuristik, 'web' bewusst fest
    try:
        W, H = he.probe_resolution(input_file)  # nutzt dein helpers.he
    except Exception:
        W, H = (0, 0)
    auto_pol = decide_policy(cmd, method="web", src_w=W, src_h=H)
    return tune_final_cmd(list(cmd), src_w=W, src_h=H, policy=auto_pol)
