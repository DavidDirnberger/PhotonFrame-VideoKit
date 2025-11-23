#!/usr/bin/env python3
# ai_helpers.py
# ─────────────────────────────────────────────────────────────────────────────
# Zweck:
#   High-level Hilfen rund um ESRGAN/RealCUGAN: Capability-Probe, Backend-Wahl,
#   Orchestrierung der Upscaling-Pipelines (PyTorch/NCNN), Preflight/Diagnose,
#   sowie robuste CLI-Argument-Erkennung und Ausführung.
#
# Public API (von anderen Modulen gedacht aufzurufen):
#   - SourceSignature (dataclass)
#   - probe_system_caps(venv_python: Optional[Path], esr_script: Optional[Path]) -> SystemCaps
#   - format_model_status_line(model_key: str, backend: str, caps: SystemCaps, *, lang: Optional[str]=None) -> str
#   - compute_available_models_for_caps(caps: SystemCaps) -> List[str]
#   - pick_backend_for_model_runtime(model_key: str, caps: SystemCaps, *, prefer_vulkan: bool | None = None) -> Optional[str]
#   - compute_preflight_info(venv_python: Path, model: str, raw_dir: Path, ...) -> PreflightInfo
#   - preflight_table_payload(info: PreflightInfo) -> (selected, labels, groups)
#   - run_esrgan_for_chunk(venv_python: Path, esr_script: Path, esrgan_root: Path, model: str, raw_dir: Path, up_dir: Path, ...) -> bool
#
# Struktur:
#   1) Imports (Stdlib / Third-Party / Projekt)
#   2) Datenmodelle (SourceSignature, PreflightInfo)
#   3) GFPGAN Warm-Up (optional)
#   4) Capability-Probing & Backend-Wahl
#   5) Preflight-Hilfen (Tabellen/Labels)
#   6) CLI-Arg-Builder & Spawn-Utilities
#   7) NCNN-Wege (per-Frame / Directory)
#   8) Python-Wege (per-Frame / Directory) inkl. TTA-Emulation & Pool
#   9) Normalisierung & Utilities (Collect/Relocate/Ensure PNG)
#  10) Orchestrierung (Kaskade)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import platform
import shutil

# ========== 1) IMPORTS =======================================================
# --- Standardbibliothek ---
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ai_backend as ab
import ai_ncnn_backend as an
import ai_pytorch_backend as ap
import ai_tta as at
import consoleOutput as co
import definitions as defin
import graphic_helpers as gh
import helpers as he
import image_helper as ih
import mem_guard as mg
from ai_backend import SystemCaps

# import configManager as cm
from configManager import ConfigManager

# --- Projekt-Module (lokal) ---
from i18n import _
from loghandler import print_log

# --- Globale Cancel-Handler & ESC-Listener aktivieren (einmalig) ---
mg.install_global_cancel_handlers()
mg.enable_escape_cancel()


# ========== RUNTIME-CONFIG (PUBLIC) ==========================================
# Public API: set_config / get_config
#   - force_backend: 'pytorch' | 'ncnn' | 'auto'/None
#   - disable_gpu: True = Torch auf CPU, NCNN wird übersprungen (kein Vulkan)
#   - prefer_vulkan: True bevorzugt NCNN bei automatischer Wahl
_FORCE_BACKEND: Optional[str] = None  # 'pytorch' | 'ncnn' | None ('auto')
_DISABLE_GPU: bool = False  # True => GPU aus (Torch->CPU, NCNN skip)
_PREFER_VULKAN: Optional[bool] = None  # Bevorzugung in Auto-Pick


def set_config(
    *,
    force_backend: Optional[str] = None,
    disable_gpu: Optional[bool] = None,
    prefer_vulkan: Optional[bool] = None,
) -> None:
    """Setzt Laufzeit-Overrides für Backend/Device-Auswahl."""
    global _FORCE_BACKEND, _DISABLE_GPU, _PREFER_VULKAN
    if force_backend is not None:
        fb = str(force_backend).strip().lower() or None
        if fb in ("auto", ""):
            fb = None
        if fb not in (None, "pytorch", "ncnn"):
            co.print_warning(
                f"[ai_helpers.set_config] unknown force_backend={force_backend!r}; "
                f"expected 'pytorch'|'ncnn'|'auto'/None"
            )
        else:
            _FORCE_BACKEND = fb
    if disable_gpu is not None:
        _DISABLE_GPU = bool(disable_gpu)
    if prefer_vulkan is not None:
        _PREFER_VULKAN = bool(prefer_vulkan)
    print_log(
        f"[ai_helpers.cfg] force_backend={_FORCE_BACKEND or 'auto'}; "
        f"disable_gpu={'yes' if _DISABLE_GPU else 'no'}; "
        f"prefer_vulkan={_PREFER_VULKAN}"
    )


# ========== 4) CAPABILITY-PROBING & BACKEND-WAHL =============================

# ==== RUNTIME-CAPS & BACKEND-WAHL ============================================


def probe_system_caps(
    *, venv_python: Optional[Path] = None, esr_script: Optional[Path] = None
) -> SystemCaps:
    """Ermittelt Torch/NCNN/TTA-Fähigkeiten. NCNN ist nur OK, wenn das Binary existiert."""
    os_name = platform.system().lower()

    # Torch-Device (cuda/mps/cpu/None)
    dev, cu_ok, mps_ok = ab.probe_torch_device_in_venv(venv_python)

    # NCNN binaries
    b1 = shutil.which("realesrgan-ncnn-vulkan")
    b2 = shutil.which("realcugan-ncnn-vulkan")
    bins: List[str] = [p for p in (b1, b2) if p]
    ncnn_ok = bool(bins)

    # TTA-Fähigkeiten
    py_tta = bool(
        esr_script and venv_python and at.python_supports_tta(venv_python, esr_script)
    )
    n_tta = any(at.ncnn_supports_tta(b) for b in bins)

    caps = SystemCaps(
        os_name=os_name,
        torch_device=dev,
        torch_cuda_ok=cu_ok,
        torch_mps_ok=mps_ok,
        ncnn_ok=ncnn_ok,
        ncnn_bins=bins,
        realcugan_ok=bool(b2),
        python_tta_ok=py_tta,
        ncnn_tta_ok=n_tta,
    )

    # „GPU aus“-Override → NCNN & CUDA/MPS aus
    if _DISABLE_GPU:
        caps = SystemCaps(
            os_name=os_name,
            torch_device="cpu",
            torch_cuda_ok=False,
            torch_mps_ok=False,
            ncnn_ok=False,
            ncnn_bins=[],
            realcugan_ok=False,
            python_tta_ok=py_tta,
            ncnn_tta_ok=False,
        )
        print_log("[caps] override: disable_gpu=True → Torch=CPU, NCNN=off")

    print_log(
        "[caps] torch_dev=%s cuda=%s mps=%s ncnn_ok=%s realcugan_ok=%s ncnn_bins=%s"
        % (
            caps.torch_device,
            caps.torch_cuda_ok,
            caps.torch_mps_ok,
            caps.ncnn_ok,
            caps.realcugan_ok,
            caps.ncnn_bins,
        )
    )
    return caps


def compute_available_models_for_caps(caps: SystemCaps) -> List[str]:
    """Filtert alle Models, für die mind. ein lauffähiges Backend existiert."""
    meta_all = ab.get_model_meta()
    out: List[str] = []
    for key, meta in meta_all.items():
        av = [str(x).lower() for x in (meta.get("available_backends") or [])]
        if not av and meta.get("backend"):
            av = [str(meta["backend"]).lower()]
        if not av:
            av = ["pytorch", "ncnn"]
        ok_any = any(ab.backend_runtime_available(bk, caps, key) for bk in av)
        if ok_any:
            out.append(key)
    return out


def available_backends(model: str) -> List[str]:
    meta = _model_meta(model)
    if "available_backends" in meta and isinstance(meta["available_backends"], list):
        return [str(x).lower() for x in meta["available_backends"] if x]
    if "backend" in meta and isinstance(meta["backend"], str):
        return [meta["backend"].lower()]
    return ["pytorch", "ncnn"]


def _pick_backend_for_model(
    model_key: str,
    meta: Dict[str, Any],
    caps: SystemCaps,
    *,
    prefer_vulkan: bool | None = None,
) -> Optional[str]:
    """Heuristik zur Backend-Wahl je Modell & Runtime-Caps."""
    av = [str(x).lower() for x in (meta.get("available_backends") or [])]
    if not av:
        av = (
            [str(meta.get("backend", "")).lower()]
            if meta.get("backend")
            else ["pytorch", "ncnn"]
        )

    ordered = list(av)
    if bool(prefer_vulkan):
        ordered = sorted(av, key=lambda x: 0 if x == "ncnn" else 1)
    else:
        # Torch-GPU > Torch-MPS > NCNN > Torch-CPU
        def _score(bk: str) -> int:
            if bk == "pytorch":
                if caps.torch_device == "cuda":
                    return 0
                if caps.torch_device == "mps":
                    return 1
                if caps.torch_device == "cpu":
                    return 3
                return 99
            if bk == "ncnn":
                return 2
            return 50

        ordered = sorted(av, key=_score)

    for bk in ordered:
        if ab.backend_runtime_available(bk, caps, model_key):
            return bk
    return None


def pick_backend_for_model_runtime(
    model_key: str, caps: SystemCaps, *, prefer_vulkan: bool | None = None
) -> Optional[str]:
    """Fassade mit Runtime-Overrides (force_backend / prefer_vulkan)."""
    if prefer_vulkan is None and _PREFER_VULKAN is not None:
        prefer_vulkan = _PREFER_VULKAN

    if _FORCE_BACKEND in ("pytorch", "ncnn"):
        fb = _FORCE_BACKEND
        if fb == "ncnn" and _DISABLE_GPU:
            co.print_warning(
                "[ai_helpers] force_backend='ncnn' aber disable_gpu=True → NCNN benötigt Vulkan. Fallback PyTorch."
            )
            fb = "pytorch"
        if ab.backend_runtime_available(fb, caps, model_key):
            print_log(f"[backend-pick] forced: {fb}")
            return fb
        co.print_warning(
            f"[backend-pick] forced backend '{_FORCE_BACKEND}' nicht verfügbar → Auto-Heuristik."
        )

    meta = ab.get_model_meta().get(model_key, {})
    return _pick_backend_for_model(model_key, meta, caps, prefer_vulkan=prefer_vulkan)


def _resolve_entry_backend_for_chunk(
    *, model: str, caps: SystemCaps, entry_backend: Optional[str]
) -> Optional[str]:
    """
    Ermittelt das Start-Backend für run_esrgan_for_chunk:
      - Nimmt 'entry_backend' (von aienhance) als Vorgabe, wenn gültig.
      - Sonst verwendet exakt deine vorhandene Laufzeit-Heuristik
        pick_backend_for_model_runtime(...) (gleiche Logik wie in aienhance).
    Gibt 'pytorch' | 'ncnn' | None zurück.
    """
    be = (entry_backend or "").strip().lower() if entry_backend else None
    if be in {"pytorch", "ncnn"}:
        return be
    # gleiche Logik wie aienhance via vorhandener Hilfsfunktion
    picked = pick_backend_for_model_runtime(model, caps)
    return (picked or "").lower() if picked else None


def format_model_status_line(
    model_key: str, backend: str, caps: SystemCaps, *, lang: Optional[str] = None
) -> str:
    """UI-Zeile: „Beschreibung — [Backend]“."""
    meta = ab.get_model_meta().get(model_key, {})
    base = ab.desc_from_meta(meta)
    be = backend.lower()
    be_tag = "PyTorch"
    if be == "pytorch":
        if caps.torch_device == "cuda":
            be_tag = "PyTorch • CUDA"
        elif caps.torch_device == "mps":
            be_tag = "PyTorch • MPS"
        elif caps.torch_device == "cpu":
            be_tag = "PyTorch • CPU"
    elif be == "ncnn":
        be_tag = "NCNN • Vulkan"
    return f"{base}  —  [{be_tag}]"


def _model_meta(model: str) -> Dict[str, Any]:
    try:
        meta = getattr(defin, "MODEL_META", {})
        if isinstance(meta, dict):
            return dict(meta.get(model, {}))
    except Exception:
        pass
    return {}


# ========== 5) PREFLIGHT-HILFEN =============================================


@dataclass
class PreflightInfo:
    cuda_ok: bool
    gpu: str
    vram_mb: int
    torch: str
    torch_cuda: str
    cuda_visible_devices: str
    workers: int
    tile: Optional[int]
    fp32: bool
    frames: int
    scale: int


def compute_preflight_info(
    venv_python: Path,
    model: str,
    raw_dir: Path,
    total_frames: Optional[int] = None,
    prefer_fp32: Optional[bool] = None,
    initial_tile: Optional[int] = None,
    face_enhance: bool = False,  # rückwärtskompatibel
    user_profile: Optional[str] = None,
    tta: bool = False,
) -> PreflightInfo:
    """Liest GPU/VRAM/Torch, schätzt Tile/Workers/Scale und liefert UI-freundliche Preflight-Infos."""
    print_log(f"[preflight] start model={model} raw_dir={raw_dir}")
    cuda_ok, dev_name, vram_mb = ab.detect_gpu_info(venv_python)
    torch_v, cuda_v = ab.torch_versions(venv_python)
    scale = ab.infer_model_scale(model)

    if prefer_fp32 is None:
        prefer_fp32 = ab.prefer_fp32_default()

    tile_size = (
        initial_tile
        if (isinstance(initial_tile, int) and initial_tile >= 0)
        else ab.pick_tile_size(vram_mb, bool(prefer_fp32), scale)
    )
    frames = (
        int(total_frames)
        if (total_frames is not None)
        else ih.count_raw_frames(raw_dir)
    )
    mp = ih.estimate_input_megapixels(raw_dir)
    norm = ab.normalize_worker_profile_name(user_profile or "auto")
    workers = ab.suggest_parallel_workers(
        vram_mb=vram_mb,
        tile_size=int(tile_size or 0),
        prefer_fp32=bool(prefer_fp32),
        total_frames=frames,
        input_megapixels=mp,
        scale=scale,
        model=model,
        face_enhance=face_enhance,
        user_profile=norm,
    )

    # Profil hart respektieren – auch in der Vorschau:
    if norm in ("serial", "no_parallelisation", "min", "minimal"):
        workers = 1
    # TTA-Vorschau realistisch einknicken
    if tta:
        try:
            workers = at.adjust_workers_for_emulated_tta(
                workers,
                vram_mb,
                honor_min=(norm in ("min", "minimal", "serial", "no_parallelisation")),
                user_profile=norm,
            )
        except Exception:
            pass

    print_log(
        f"[preflight] cuda_ok={he.yesno(cuda_ok)} gpu='{dev_name}' vram={vram_mb}MB torch={torch_v} cuda={cuda_v} "
        f"tile={ab.tile_label(tile_size)} fp32={he.yesno(prefer_fp32)} frames={frames} scale={scale} workers={workers} "
        f"CUDA_VISIBLE_DEVICES={ab.env_cuda_mask()}"
    )

    return PreflightInfo(
        cuda_ok=cuda_ok,
        gpu=dev_name or "-",
        vram_mb=int(vram_mb or 0),
        torch=torch_v,
        torch_cuda=cuda_v,
        cuda_visible_devices=ab.env_cuda_mask(),
        workers=workers,
        tile=(int(tile_size) if tile_size else None),
        fp32=bool(prefer_fp32),
        frames=frames,
        scale=scale,
    )


def preflight_table_payload(
    info: PreflightInfo,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, str]], Dict[str, set]]:
    """Gibt (selected, labels, groups) für eine zweisprachige Preflight-Tabelle zurück."""
    selected = {
        "cuda": _(he.yesno(info.cuda_ok)),
        "gpu": info.gpu,
        "vram_mb": info.vram_mb,
        "cuda_visible_devices": info.cuda_visible_devices,
        "torch": info.torch,
        "torch_cuda": info.torch_cuda,
        "workers": info.workers,
        "tile": ab.tile_label(info.tile),
        "fp32": _(he.yesno(info.fp32)),
        "frames": info.frames,
    }
    labels = {
        "cuda": {"de": "CUDA", "en": "CUDA"},
        "gpu": {"de": "GPU", "en": "GPU"},
        "vram_mb": {"de": "VRAM (MB)", "en": "VRAM (MB)"},
        "cuda_visible_devices": {
            "de": "CUDA_VISIBLE_DEVICES",
            "en": "CUDA_VISIBLE_DEVICES",
        },
        "torch": {"de": "Torch", "en": "Torch"},
        "torch_cuda": {"de": "torch.cuda", "en": "torch.cuda"},
        "workers": {"de": "Workers", "en": "Workers"},
        "tile": {"de": "Tile", "en": "Tile"},
        "fp32": {"de": "FP32", "en": "FP32"},
        "frames": {"de": "Frames", "en": "Frames"},
    }
    groups = {
        "env": {
            "cuda",
            "gpu",
            "vram_mb",
            "cuda_visible_devices",
            "torch",
            "torch_cuda",
        },
        "plan": {"workers", "tile", "fp32", "frames"},
    }
    return selected, labels, groups


# ========== 10) ORCHESTRIERUNG (KASKADE) ====================================


def _effective_cuda_mask_from_env(use_gpu: bool) -> Optional[str]:
    """
    Liefert eine EIN-GPU-taugliche Maske aus der Umgebung.
      - use_gpu=False → "cpu" (erzwingt CPU)
      - use_gpu=True  → None (auto), außer eine Maske ist gesetzt:
           * "" (leer) → "cpu"
           * "0,1,..." → erster Eintrag ("0")
           * "0"       → "0"
    Hintergrund: Wir vermeiden explizit Multi-GPU-Betrieb.
    """
    if not use_gpu:
        return "cpu"
    val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if val is None:
        return None  # auto
    m = str(val).strip()
    if m == "":
        return "cpu"
    # nur das erste Device zulassen
    if "," in m:
        first = m.split(",")[0].strip()
        return first or None
    return m


def _parse_tri_state_env(name: str) -> Optional[bool]:
    """
    Liest ein Tri-State-Flag aus einer Umgebungsvariable:

        off/false/no/0  -> False
        on/true/yes/1   -> True
        auto/leer/sonst -> None  (kein Override)
    """
    try:
        v = os.environ.get(name, "")
    except Exception:
        return None
    s = str(v).strip().lower()
    if not s or s == "auto":
        return None
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return None


def _decide_gpu_usage(venv_python: Path, esr_script: Path) -> Tuple[bool, str]:
    """
    Entscheidet final, ob GPU genutzt wird.

    Priorität:
      1) _DISABLE_GPU (runtime override, z. B. via set_config)
      2) AI_GPU_ENABLED (ENV, tri-state)
      3) config [hardware].gpu_enabled (tri-state)
      4) Runtime-Caps (CUDA/MPS/NCNN verfügbar?)

    Rückgabe:
        (use_gpu: bool, reason_tag: str)
    """
    # 1) Globaler Runtime-Override (z. B. aienhance -G off)
    if bool(globals().get("_DISABLE_GPU")):
        return (False, "env_DISABLE_GPU")

    # 2) Config laden (kann 'auto', true/false, leer, etc. sein)
    cfg_raw: str = "auto"
    cfg_tri: Optional[bool] = None
    try:
        proj_root = getattr(defin, "PROJECT_ROOT", None) or getattr(
            defin, "SCRIPT_DIR", None
        )
        cm = ConfigManager(project_root=proj_root) if proj_root else ConfigManager()
        cm.load()
        cfg_raw = (
            (cm.get_str("hardware", "gpu_enabled", fallback="auto") or "auto")
            .strip()
            .lower()
        )
        # get_bool gibt jetzt für 'auto' / unbekannt → fallback (None) zurück
        cfg_tri = cm.get_bool(
            "hardware", "gpu_enabled", fallback=None, auto_resolve=False
        )
    except Exception as e:
        print_log(f"[gpu.decide] config load failed: {e!r}")
        cfg_raw = "auto"
        cfg_tri = None

    # 3) Runtime-Caps ermitteln (unabhängig vom Tri-State)
    caps = probe_system_caps(venv_python=venv_python, esr_script=esr_script)
    hw_ok = bool(caps.torch_cuda_ok or caps.torch_mps_ok or caps.ncnn_ok)

    # 4) ENV-Override: AI_GPU_ENABLED (tri-state)
    env_tri = _parse_tri_state_env("AI_GPU_ENABLED")
    if env_tri is not None:
        # ENV gewinnt vor Config
        if env_tri is False:
            return (False, "env_AI_GPU_ENABLED_off")
        # env_tri True → GPU erlauben, aber nur wenn HW ok
        return (
            hw_ok,
            "env_AI_GPU_ENABLED_on_hw_ok" if hw_ok else "env_AI_GPU_ENABLED_on_hw_fail",
        )

    # 5) Config-Auswertung als Tri-State
    #    cfg_tri ist True/False/None, cfg_raw der String („auto“, „true“, …)
    if cfg_tri is False or cfg_raw in {"false", "off", "no", "0"}:
        # explizit deaktiviert
        return (False, "cfg_off")

    if cfg_tri is True or cfg_raw in {"true", "on", "yes", "1"}:
        # explizit erlaubt → nur nutzen, wenn HW-Caps da sind
        return (hw_ok, "cfg_on_hw_ok" if hw_ok else "cfg_on_hw_fail")

    # 6) auto / leer / irgendwas anderes → KEIN Override → Runtime entscheidet
    #    „Runtime entscheidet“ heißt hier: GPU, wenn irgendein Backend HW nutzt.
    return (hw_ok, "auto_hw_ok" if hw_ok else "auto_hw_fail")


# ==== ORCHESTRIERUNG =========================================================


def run_ai_for_chunk(
    venv_python: Path,
    esr_script: Path,
    esrgan_root: Path,
    model: str,
    raw_dir: Path,
    up_dir: Path,
    *,
    face_enhance: bool,
    denoise: Optional[float],
    chunk_idx: Optional[int] = 0,
    chunks_total: Optional[int] = 0,
    outscale: float = 4.0,
    tta: bool = False,
    realcugan_noise: Optional[int] = None,
    user_worker_profile: Optional[str] = None,
    entry_backend: Optional[str] = None,
) -> bool:
    """
    CANCEL-HARDENED Orchestrierung:
      - ESC/Cancel durchschlägt sofort (kein Fallback).
      - Nach *jedem* Leiter-Schritt wird auf Cancel geprüft.
      - Directory-Mode wird nie gestartet, wenn ein Cancel gesehen wurde.
      - Rückgabe ist bei Cancel stets False.
    """
    try:
        # ---- Globale Cancel-Guards -------------------------------------------
        try:
            mg.CANCEL.clear()
        except Exception:
            pass
        mg.install_global_cancel_handlers()
        mg.enable_escape_cancel()

        # kleine Hilfsfunktion: Cancel prüfen + hart aussteigen
        def _cancelled(tag: str) -> bool:
            if mg.is_cancelled():
                print_log(
                    f"[orchestrate] CANCEL @ {tag} → stoppe sofort, kein Fallback."
                )
                try:
                    mg.kill_all()
                except Exception:
                    pass
                return True
            return False

        if _cancelled("begin"):
            return False

        forced = globals().get("_FORCE_BACKEND")

        # ---- GPU-Entscheidung (ENV > Config > Probe) --------------------------
        use_gpu, reason = _decide_gpu_usage(venv_python, esr_script)
        cuda_mask = _effective_cuda_mask_from_env(use_gpu)
        if not use_gpu:
            print_log(
                f"[orchestrate] GPU disabled ({reason}) → force Torch CPU, disable NCNN"
            )
            print_log(
                "[PT-orch] GPU disabled → forcing CUDA=cpu, skipping NCNN", "_pytorch"
            )
        else:
            print_log(
                f"[orchestrate] GPU enabled ({reason}); cuda_mask={cuda_mask or '(auto)'}"
            )

        print_log(
            f"[orchestrate] begin model={model} outscale={outscale} tta_req={'yes' if tta else 'no'} "
            f"raw_dir={raw_dir} up_dir={up_dir} forced_backend={forced or 'auto'} "
            f"gpu={'on' if use_gpu else 'off'} entry_backend={entry_backend or 'auto'}"
        )

        if _cancelled("after-begin-log"):
            return False

        # ---- CAPS & Laufzeitinfos ---------------------------------------------
        caps_rt = probe_system_caps(venv_python=venv_python, esr_script=esr_script)
        is_cugan = str(model).lower().startswith(("realcugan", "models-"))
        ncnn_bin = (
            None
            if (not use_gpu)
            else (an.which_realcugan() if is_cugan else an.which_ncnn())
        )

        py_tta_ok = at.python_supports_tta(venv_python, esr_script)
        ncnn_tta_ok = bool(ncnn_bin and at.ncnn_supports_tta(ncnn_bin))

        cuda_ok, dev, vram_mb = ab.detect_gpu_info(venv_python)
        prefer_fp32 = ab.prefer_fp32_default()
        scale_int = int(max(1, min(8, round(float(outscale)))))
        total_frames = ih.count_raw_frames(raw_dir)
        input_mp = ih.estimate_input_megapixels(raw_dir)
        gpu_id = ab.parse_gpu_id_for_ncnn(cuda_mask)

        print_log(
            f"[orchestrate] caps: torch_dev={caps_rt.torch_device} ncnn_ok={'yes' if caps_rt.ncnn_ok else 'no'} "
            f"realcugan_ok={'yes' if caps_rt.realcugan_ok else 'no'} py_tta_ok={'yes' if py_tta_ok else 'no'} "
            f"ncnn_tta_ok={'yes' if ncnn_tta_ok else 'no'} ncnn_bin={ncnn_bin}"
        )
        print_log(
            f"[orchestrate] gpu='{dev}' vram={vram_mb}MB fp32={'yes' if prefer_fp32 else 'no'} "
            f"frames={total_frames} mp={input_mp:.3f} CUDA_MASK={ab.env_cuda_mask()} gpu_id_ncnn={gpu_id}"
        )

        ab.persist_context(raw_dir, up_dir, tag="orchestrate-start")

        if _cancelled("after-caps"):
            return False

        # ---- PyTorch Leiter (Pool → Serial → Directory) -----------------------
        def _torch_per_frame_parallel() -> bool:
            if _cancelled("torch-parallel-enter"):
                return False
            prof = user_worker_profile
            norm_prof = ab.normalize_worker_profile_name(prof or "auto") or "auto"
            if norm_prof in ("serial", "no_parallelisation", "min", "minimal"):
                print_log(
                    f"[PT-orch] skip torch per-frame PARALLEL due to profile={norm_prof}",
                    "_pytorch",
                )
                return False
            print_log(
                f"[PT-orch] try torch per-frame PARALLEL profile={prof or norm_prof}",
                "_pytorch",
            )
            phase = gh.ui_new_phase()
            ok = ap.run_esrgan_per_frame_python(
                venv_python=venv_python,
                esr_script=esr_script,
                esrgan_root=esrgan_root,
                model=model,
                raw_dir=raw_dir,
                up_dir=up_dir,
                face_enhance=face_enhance,
                denoise=denoise,
                outscale=float(outscale),
                tta=bool(tta),
                prefer_fp32=prefer_fp32,
                initial_tile=None,
                cuda_mask=cuda_mask,
                chunk_idx=chunk_idx,
                chunks_total=chunks_total,
                user_worker_profile=prof or norm_prof or "auto",
                ui_phase_id=phase,
            )
            if _cancelled("torch-parallel-exit"):
                return False
            if not ok:
                ab.persist_context(raw_dir, up_dir, tag="torch-per-frame-parallel-fail")
            return ok

        def _torch_per_frame_serial() -> bool:
            if _cancelled("torch-serial-enter"):
                return False
            print_log("[PT-orch] try torch per-frame SERIAL", "_pytorch")
            phase = gh.ui_new_phase()
            ok = ap.run_esrgan_per_frame_python(
                venv_python=venv_python,
                esr_script=esr_script,
                esrgan_root=esrgan_root,
                model=model,
                raw_dir=raw_dir,
                up_dir=up_dir,
                face_enhance=face_enhance,
                denoise=denoise,
                outscale=float(outscale),
                tta=bool(tta),
                prefer_fp32=prefer_fp32,
                initial_tile=None,
                cuda_mask=cuda_mask,
                chunk_idx=chunk_idx,
                chunks_total=chunks_total,
                user_worker_profile="serial",
                ui_phase_id=phase,
            )
            if _cancelled("torch-serial-exit"):
                return False
            if not ok:
                ab.persist_context(raw_dir, up_dir, tag="torch-per-frame-serial-fail")
            return ok

        def _torch_directory() -> bool:
            # Directory-Mode *niemals* bei Cancel starten
            if _cancelled("torch-dir-enter"):
                return False
            produced = len(list(up_dir.glob("frame_*.png"))) + len(
                list(up_dir.glob("*_out.png"))
            )
            if produced >= max(1, total_frames):
                # Selbst wenn „already complete“, bei Cancel KEIN Erfolg.
                if _cancelled("torch-dir-skip-already-complete"):
                    return False
                print_log(
                    "[orchestrate] up_dir bereits vollständig – skip torch directory"
                )
                print_log(
                    "[PT-orch] skip torch directory (already complete)", "_pytorch"
                )
                return True
            print_log("[orchestrate] torch directory …")
            print_log("[PT-orch] try torch DIRECTORY mode", "_pytorch")
            phase = gh.ui_new_phase()
            raw_for_dir = ab.ensure_png_inputs_for_tool(
                raw_dir, tmp_root=raw_dir.parent
            )
            ok = ap.run_esrgan_python_dir(
                venv_python=venv_python,
                esr_script=esr_script,
                esrgan_root=esrgan_root,
                model=model,
                raw_dir=raw_for_dir,
                up_dir=up_dir,
                face_enhance=face_enhance,
                denoise=denoise,
                tile_size=0,
                force_fp32=prefer_fp32,
                outscale=float(outscale),
                tta=bool(tta),
                cuda_mask=cuda_mask,
                ui_phase_id=phase,
            )
            if _cancelled("torch-dir-exit"):
                return False
            if not ok:
                ab.persist_context(raw_for_dir, up_dir, tag="torch-dir-fail")
            return ok

        def _try_torch_ladder() -> bool:
            if _cancelled("torch-ladder-enter"):
                return False
            print_log("[PT-orch] START torch ladder", "_pytorch")
            if _torch_per_frame_parallel():
                print_log(
                    "[PT-orch] ladder stop at torch per-frame PARALLEL", "_pytorch"
                )
                return True
            if _cancelled("after-parallel"):
                return False
            if _torch_per_frame_serial():
                print_log("[PT-orch] ladder stop at torch per-frame SERIAL", "_pytorch")
                return True
            if _cancelled("after-serial"):
                return False
            res = _torch_directory()
            if _cancelled("after-dir"):
                return False
            print_log(f"[PT-orch] ladder end torch DIRECTORY res={res}", "_pytorch")
            return res

        # ---- NCNN Leiter (per-frame K→1 → Directory) --------------------------
        def _ncnn_per_frame(workers: int) -> bool:
            if _cancelled("ncnn-per-frame-enter"):
                return False
            if not ncnn_bin:
                ab.warn_once(
                    "ncnn.missing",
                    f"{'realcugan-ncnn-vulkan' if is_cugan else 'realesrgan-ncnn-vulkan'} nicht gefunden oder deaktiviert.",
                )
                return False
            raw_for_ncnn = ab.ensure_png_inputs_for_tool(
                raw_dir, tmp_root=raw_dir.parent
            )
            ab.persist_context(
                raw_for_ncnn, up_dir, tag=f"ncnn-per-frame-workers{workers}-before"
            )
            phase = gh.ui_new_phase()
            ok = an.run_esrgan_per_frame_ncnn(
                ncnn_bin=ncnn_bin,
                model=model,
                raw_dir=raw_for_ncnn,
                up_dir=up_dir,
                scale=scale_int,
                workers=max(1, int(workers)),
                tta=bool(tta and ncnn_tta_ok),
                gpu_id=gpu_id,
                chunk_idx=chunk_idx,
                chunks_total=chunks_total,
                ui_phase_id=phase,
                noise_level=(
                    int(realcugan_noise)
                    if is_cugan and realcugan_noise is not None
                    else None
                ),
                is_cugan=bool(is_cugan),
            )
            if _cancelled("ncnn-per-frame-exit"):
                return False
            if ok and (
                ab.normalize_upscaled_sequence(up_dir, raw_for_ncnn, esrgan_root) > 0
            ):
                return True
            ab.persist_context(
                raw_for_ncnn, up_dir, tag=f"ncnn-per-frame-workers{workers}-fail"
            )
            return False

        def _ncnn_directory() -> bool:
            if _cancelled("ncnn-dir-enter"):
                return False
            if not ncnn_bin:
                ab.warn_once(
                    "ncnn.missing",
                    f"{'realcugan-ncnn-vulkan' if is_cugan else 'realesrgan-ncnn-vulkan'} nicht gefunden oder deaktiviert.",
                )
                return False
            raw_for_ncnn = ab.ensure_png_inputs_for_tool(
                raw_dir, tmp_root=raw_dir.parent
            )
            ret = an.run_esrgan_ncnn_dir(
                ncnn_bin,
                raw_for_ncnn,
                up_dir,
                model,
                scale_int,
                bool(tta and ncnn_tta_ok),
                gpu_id,
                timeout_sec=0,
                noise_level=(
                    int(realcugan_noise)
                    if is_cugan and realcugan_noise is not None
                    else None
                ),
                is_cugan=bool(is_cugan),
            )
            if _cancelled("ncnn-dir-exit"):
                return False
            if ret == 0 and (
                ab.normalize_upscaled_sequence(up_dir, raw_for_ncnn, esrgan_root) > 0
            ):
                return True
            ab.persist_context(raw_for_ncnn, up_dir, tag="ncnn-dir-fail")
            return False

        def _try_ncnn_ladder() -> bool:
            if _cancelled("ncnn-ladder-enter"):
                return False
            if not ncnn_bin:
                print_log("[orchestrate] NCNN nicht verfügbar – Leiter übersprungen.")
                return False
            norm_profile = ab.normalize_worker_profile_name(
                user_worker_profile or "auto"
            )
            workers_eff = ab.suggest_parallel_workers(
                vram_mb=vram_mb,
                tile_size=0,
                prefer_fp32=prefer_fp32,
                total_frames=total_frames,
                input_megapixels=input_mp,
                scale=scale_int,
                model=model,
                user_profile=norm_profile,
            )
            if norm_profile in ("serial", "no_parallelisation", "min", "minimal"):
                workers_eff = 1
            if workers_eff > 1 and _ncnn_per_frame(workers_eff):
                return True
            if _cancelled("after-ncnn-K"):
                return False
            if _ncnn_per_frame(1):
                return True
            if _cancelled("after-ncnn-1"):
                return False
            return _ncnn_directory()

        # ---- Start-Backend bestimmen ------------------------------------------
        start_backend = _resolve_entry_backend_for_chunk(
            model=model, caps=caps_rt, entry_backend=entry_backend
        )
        if forced in {"pytorch", "ncnn"}:
            start_backend = forced
        print_log(
            f"[orchestrate] start-backend={start_backend or 'none'} (forced={forced or 'no'})"
        )

        if _cancelled("before-exec"):
            return False

        # ---- Ausführung nach Richtungslogik -----------------------------------
        if start_backend == "pytorch":
            res_torch = _try_torch_ladder()
            if _cancelled("after-torch-ladder"):
                return False
            if res_torch:
                print_log("[PT-orch] EXIT result=True (torch ladder)", "_pytorch")
                return True

            # kein Fallback nach Cancel
            if mg.is_cancelled():
                print_log("[orchestrate] Cancel erkannt → KEIN NCNN-Fallback.")
                return False

            ncnn_avail = ab.backend_runtime_available("ncnn", caps_rt, model)
            if not ncnn_avail or (not use_gpu):
                print_log(
                    "[orchestrate] Torch-Leiter fehlgeschlagen und NCNN nicht verfügbar → Abbruch."
                )
                print_log(
                    "[PT-orch] EXIT result=False (torch failed, no ncnn)", "_pytorch"
                )
                return False

            print_log(
                "[orchestrate] Torch-Leiter fehlgeschlagen → NCNN-Leiter (einmalig) …"
            )
            res = _try_ncnn_ladder()
            print_log(f"[PT-orch] EXIT result={res} (torch→ncnn)", "_pytorch")
            return res

        elif start_backend == "ncnn":
            res = _try_ncnn_ladder()
            if _cancelled("after-ncnn-ladder"):
                return False
            print_log(f"[PT-orch] EXIT result={res} (ncnn only)", "_pytorch")
            return res

        else:
            print_log("[orchestrate] Kein lauffähiges Start-Backend erkannt → Abbruch.")
            print_log("[PT-orch] EXIT result=False (no backend)", "_pytorch")
            return False

    except KeyboardInterrupt:
        try:
            mg.CANCEL.set()
        except Exception:
            pass
        try:
            mg.kill_all()
        except Exception:
            pass
        # co.print_info("Abgebrochen (ESC/Strg+C).")
        print_log("[PT-orch] KeyboardInterrupt → cancelled", "_pytorch")
        return False
