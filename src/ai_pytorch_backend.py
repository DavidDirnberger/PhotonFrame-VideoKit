##!/usr/bin/env python3
# ai_pytorch_backend.py

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

# ========== 1) IMPORTS =======================================================
# --- Standardbibliothek ---
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ai_backend as ab
import ai_tta as at
import consoleOutput as co
import definitions as defin
import graphic_helpers as gh
import helpers as he
import image_helper as ih
import mem_guard as mg
from ai_weights import resolve_gfpgan_weight

# --- Projekt-Module (lokal) ---
from i18n import _
from loghandler import print_log
from pil_image import Image
from pool_workers import (
    run_sharded_dir_job_with_retries,
)  # persistente CLI-Worker (Directory-Shards)

# --- Globale Cancel-Handler & ESC-Listener aktivieren (einmalig) ---
mg.install_global_cancel_handlers()
mg.enable_escape_cancel()

_ESRGAN_FILE2NAME: dict[str, str] = {
    "realesr-general-x4v3.pth": "realesr-general-x4v3",
    "realesr-general-wdn-x4v3.pth": "realesr-general-x4v3",
    "realesrgan-x4plus.pth": "realesrgan-x4plus",
    "realesrgan-x4plus-anime-6b.pth": "realesrgan-x4plus-anime-6b",
    "realesrgan-x2plus.pth": "realesrgan-x2plus",
}


# ─────────────────────────────────────────────────────────────────────────────
# GFPGAN: optionales einmaliges Warm-Up (Download/Lazy-Load verhindern)
# ─────────────────────────────────────────────────────────────────────────────
def _gfpgan_warmup_once(
    *,
    venv_python: Path,
    esr_script: Path,
    esrgan_root: Path,
    model: str,
    help_text: Optional[str] = None,
    cuda_mask: Optional[str] = None,
    timeout_s: int = 60,
) -> bool:
    """
    Führt *einmalig* ein kleines Face-Enhance (CLI) aus, damit GFPGAN/Facexlib-
    Gewichte geladen/ggf. heruntergeladen sind, bevor mehrere Prozesse starten.
    Concurrency-safe via Lock-Directory; bevorzugt CPU, um die GPU frei zu halten.
    """
    try:
        if str(os.environ.get("AI_GFPGAN_WARMUP", "1")).strip().lower() in (
            "0",
            "false",
            "off",
            "no",
        ):
            return False
        stamp_dir = esrgan_root / ".cache"
        warm_dir = esrgan_root / "__warmup__"
        lock_dir = stamp_dir / ".gfpgan_warm.lock"
        done_file = stamp_dir / "gfpgan_warmup.ok"
        stamp_dir.mkdir(parents=True, exist_ok=True)
        if done_file.exists():
            return True
        have_lock = False
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            have_lock = True
        except Exception:
            pass
        if not have_lock:
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                if done_file.exists() or (not lock_dir.exists()):
                    return done_file.exists()
                time.sleep(0.25)
            return False
        try:
            warm_dir.mkdir(parents=True, exist_ok=True)
            tiny = warm_dir / "tiny.png"
            outd = warm_dir / "out"
            try:
                Image.new("RGB", (16, 16), (128, 128, 128)).save(
                    str(tiny), format="PNG"
                )
            except Exception:
                tiny.write_bytes(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90wS\xde"
                    b"\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd\xa7\x93\x00\x00\x00\x0cIDATx\x9cc`\x18\x05\xa3"
                    b"\x60\x14\x8c\x02\x00\x06\xfd\x01\xfe\xa3\xa3\x8b\x1b\x00\x00\x00\x00IEND\xaeB`\x82"
                )
            ht = help_text or ab.probe_realesrgan_help(venv_python, esr_script)
            in_flag = ab.pick_flag(ht, ["-i", "--input"], "-i")
            out_flag = ab.pick_flag(
                ht, ["--output", "--outdir", "--save-dir", "-o"], "--output"
            )
            ext_flag = ab.pick_flag(ht, ["--ext"], "--ext")
            sfx_flag = ab.pick_flag(ht, ["--suffix"], "--suffix")
            fe_flag = ab.pick_flag(ht, ["--face_enhance"], "--face_enhance")
            tile_flag = ab.pick_flag(ht, ["--tile", "-t"], "--tile")
            tpad_flag = ab.pick_flag(
                ht, ["--tile_pad", "--tile-pad", "-tp"], "--tile_pad"
            )
            scale_fl = ab.pick_flag(ht, ["-s", "--outscale"], "-s")
            args = []
            if in_flag:
                args += [in_flag, str(tiny)]
            args += ["-n", model]
            if out_flag:
                args += [out_flag, str(outd)]
            if ext_flag:
                args += [ext_flag, "png"]
            if sfx_flag:
                args += [sfx_flag, "out"]
            if scale_fl:
                args += [scale_fl, "1"]
            if tile_flag:
                args += [tile_flag, "64"]
            if tpad_flag:
                args += [tpad_flag, "10"]
            if fe_flag:
                args += [fe_flag]
            env = _inject_face_env(
                ab.build_esrgan_env(ab.build_env_for_gpu("cpu")), esrgan_root
            )

            cp = mg.run(
                [str(venv_python), str(esr_script), *map(str, args)],
                cwd=str(esrgan_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
                timeout=timeout_s,
            )
            out = cp.stdout or ""
            if ab.debug_enabled() or (int(cp.returncode or 0) != 0):
                ab.log_head_tail("GFPGAN-WARMUP", out, head_lines=60, tail_lines=30)
            try:
                done_file.write_text(
                    json.dumps(
                        {"ts": time.time(), "rc": int(cp.returncode or 0)}, indent=2
                    )
                )
            except Exception:
                pass
            return True
        finally:
            try:
                lock_dir.rmdir()
            except Exception:
                pass
            try:
                shutil.rmtree(warm_dir, ignore_errors=True)
            except Exception:
                pass
    except Exception as e:
        co.print_warning(f"[GFPGAN] Warm-Up übersprungen: {e}")
        return False


def _run_with_cudnn_completed(
    venv_python: Path,
    esr_script: Path,
    args_list: List[str],
    cwd: Path,
    *,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    argv = [str(venv_python), str(esr_script), *map(str, args_list)]
    ab.log_cmd("SPAWN", argv)
    cp = mg.run(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=ab.build_esrgan_env(env),
        check=False,
    )
    rc = int(cp.returncode or 0)
    out = cp.stdout or ""
    print_log(f"[SPAWN-RET] rc={rc} bytes={len(out)}")
    if rc != 0 or ab.debug_enabled():
        ab.log_head_tail("SPAWN-OUT", out, head_lines=80, tail_lines=40)
    return cp


def _popen_with_cudnn(
    venv_python: Path,
    esr_script: Path,
    args_list: List[str],
    cwd: Path,
    *,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> subprocess.Popen[str]:
    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print_log(f"[LOG] capturing → {log_path}")
        except Exception as e:
            print_log(f"[LOG] open failed: {e!r}")
            log_path = None
    argv = [str(venv_python), str(esr_script), *map(str, args_list)]
    ab.log_cmd("SPAWN-bg", argv)
    proc = mg.popen(
        argv,
        cwd=str(cwd),
        env=ab.build_esrgan_env(env),
        text=True,
        log_to=(str(log_path) if log_path else None),
    )
    return proc


def _site_pkg_weights_dir() -> Optional[Path]:
    """Versucht realesrgan/weights in site-packages zu finden (pip-Install)."""
    try:
        import importlib

        m = importlib.import_module("realesrgan")
        p = Path(getattr(m, "__file__", "")).parent / "weights"
        return p if p.is_dir() else None
    except Exception:
        return None


def _resolve_esrgan_pytorch_weights(
    vm_base: Optional[Path] = None,
) -> tuple[str, Path] | None:
    """
    Liefert (model_name, model_path) für PyTorch-ESRGAN – ohne Download.
    Reihenfolge:
      1) ESRGAN_MODEL_PATH | ESRGAN_DEFAULT_MODEL_PATH (Env)
      2) $VM_BASE/real-esrgan/weights, $VM_BASE/weights, $VM_BASE/models
      3) site-packages realesrgan/weights
      4) ~/.cache/{realesrgan, torch/hub/checkpoints}
    Priorität: general-x4v3  > x4plus > anime6B > x2plus
    """
    priority = [
        "realesr-general-x4v3.pth",
        "realesrgan-x4plus.pth",
        "realesrgan-x4plus-anime-6b.pth",
        "realesrgan-x2plus.pth",
        "realesr-animevideov3",
    ]

    # 1) Env
    for env_key in ("ESRGAN_MODEL_PATH", "ESRGAN_DEFAULT_MODEL_PATH"):
        p = Path(os.environ.get(env_key, "")).expanduser()
        if p.is_file():
            name = _ESRGAN_FILE2NAME.get(p.name.lower(), "realesr-general-x4v3")
            print_log(f"[PT-weights] via ENV {env_key} -> {p}", "_pytorch")
            return name, p

    # 2–4) bekannte Orte
    bases: list[Path] = []
    if vm_base:
        bases += [
            vm_base / "real-esrgan" / "weights",
            vm_base / "weights",
            vm_base / "models",
        ]
    sp = _site_pkg_weights_dir()
    if sp:
        bases.append(sp)
    bases += [
        Path.home() / ".cache/realesrgan",
        Path.home() / ".cache/torch/hub/checkpoints",
    ]

    for fname in priority:
        for b in bases:
            p = b / fname
            if p.is_file():
                print_log(f"[PT-weights] found {fname} in {b}", "_pytorch")
                return _ESRGAN_FILE2NAME[fname], p
    print_log("[PT-weights] NOT FOUND (return None)", "_pytorch")
    return None


# ==== NEW helper =============================================================


def _gpu_flag_if_supported(
    venv_python: Path, esr_script: Path, help_text: str
) -> Optional[str]:
    """
    Liefert einen GPU-Flag-String, der *wirklich* von der CLI akzeptiert wird.
    NEU: Default = deaktiviert. Nur wenn ESRGAN_USE_GPU_FLAG=1 gesetzt ist,
    wird das Flag probeweise aktiviert. Sonst verwenden wir ausschließlich ENV
    (CUDA_VISIBLE_DEVICES), was deiner Präferenz entspricht.
    """
    try:
        use_flag = str(os.environ.get("ESRGAN_USE_GPU_FLAG", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not use_flag:
            print_log(
                "[PT-util] GPU flag disabled by default (set ESRGAN_USE_GPU_FLAG=1 to enable).",
                "_pytorch",
            )
            return None

        # Nur wenn die Hilfe den String überhaupt erwähnt
        if "--gpu-id" not in help_text and "--gpu_id" not in help_text:
            print_log(
                "[PT-util] GPU flag not present in help → use ENV only", "_pytorch"
            )
            return None

        # Probe '--gpu-id'
        cp = mg.run(
            [str(venv_python), str(esr_script), "--gpu-id", "0", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=ab.build_esrgan_env(),
            check=False,
            timeout=5,
        )
        if int(cp.returncode or 0) == 0:
            return "--gpu-id"

        # Fallback: '--gpu_id'
        cp = mg.run(
            [str(venv_python), str(esr_script), "--gpu_id", "0", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=ab.build_esrgan_env(),
            check=False,
            timeout=5,
        )
        if int(cp.returncode or 0) == 0:
            return "--gpu_id"

        print_log("[PT-util] no working GPU flag → use ENV only", "_pytorch")
        return None
    except Exception as e:
        print_log(f"[PT-util] gpu-flag probe error: {e!r} → use ENV only", "_pytorch")
        return None


def _dump_pool_shard_logs(up_dir: Path, max_chars: int = 4000) -> None:
    """
    Liest die Shard-Logs nach Pool-Fehlschlag und loggt Head/Tail in print_log.
    """
    try:
        pool_tmp = up_dir / "__pool_tmp__"
        if not pool_tmp.is_dir():
            print_log("[PT-pf] no __pool_tmp__ found for shard log dump", "_pytorch")
            return
        for d in sorted(pool_tmp.glob("shard_*")):
            out_dir = d / "out"
            if not out_dir.is_dir():
                continue
            for lf in sorted(out_dir.glob("shard_*.log")):
                try:
                    txt = lf.read_text(errors="replace")
                except Exception as e:
                    print_log(f"[PT-pf] shard log read failed {lf}: {e!r}", "_pytorch")
                    continue
                head = txt[: max_chars // 2]
                tail = txt[-max_chars // 2 :] if len(txt) > max_chars // 2 else ""
                print_log(f"[PT-pf] ===== {lf} (HEAD) =====\n{head}", "_pytorch")
                if tail:
                    print_log(f"[PT-pf] ===== {lf} (TAIL) =====\n{tail}", "_pytorch")
    except Exception as e:
        print_log(f"[PT-pf] shard log dump error: {e!r}", "_pytorch")


# ==== DROP-IN: run_esrgan_per_frame_python ===================================


# ─────────────────────────────────────────────────────────────────────────────
# Runner bevorzugen: vm_realesrgan.py (Fallback: esr_script)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_runner(esr_script: Path) -> Path:
    """
    Nutzt bevorzugt den projektlokalen Runner 'vm_realesrgan.py'.
    Fällt auf das übergebene esr_script zurück, falls der Runner fehlt.
    """
    try:
        here = Path(__file__).resolve().parent
        cand = here / "vm_realesrgan.py"
        if cand.is_file():
            return cand
    except Exception:
        pass
    return esr_script


# ─────────────────────────────────────────────────────────────────────────────
# CUDA/Torch Diagnose – ausgelagert für schlankere Hauptfunktion
# ─────────────────────────────────────────────────────────────────────────────
def _cuda_diag_for_log(venv_python: Path) -> None:
    code = r"""
import json
d = {"ok": False}
try:
    import torch
    d["torch"] = getattr(torch, "__version__", None)
    d["cuda_version"] = getattr(getattr(torch,"version",object()), "cuda", None)
    d["cuda_available"] = bool(getattr(getattr(torch,"cuda",object()), "is_available", lambda: False)())
    d["device_count"] = int(getattr(getattr(torch,"cuda",object()), "device_count", lambda: 0)())
    if d["cuda_available"]:
        i = getattr(getattr(torch,"cuda",object()), "current_device", lambda: 0)()
        props = getattr(getattr(torch,"cuda",object()), "get_device_properties", lambda *_: None)(i)
        d["current_device"] = int(i)
        if props:
            d["name"] = getattr(props, "name", "")
            d["cc"] = f"{getattr(props,'major',0)}.{getattr(props,'minor',0)}"
            d["total_mb"] = int(getattr(props, "total_memory", 0) / 1024 / 1024)
            d["mp_count"] = int(getattr(props, "multi_processor_count", 0))
    try:
        import torch.backends.cudnn as cudnn
        d["cudnn_ok"] = bool(getattr(cudnn, "is_available", lambda: False)())
        try:
            d["cudnn_ver"] = int(getattr(cudnn, "version", lambda: 0)() or 0)
        except Exception:
            d["cudnn_ver"] = None
    except Exception as e:
        d["cudnn_err"] = str(e)
    d["ok"] = True
except Exception as e:
    d["err"] = str(e)
print(json.dumps(d))
""".strip()
    try:
        cp = mg.run(
            [str(venv_python), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=ab.build_esrgan_env(),
        )
        txt = (cp.stdout or "").strip()
        print_log(f"[PT-pf] CUDA-DIAG {txt}", "_pytorch")
    except Exception as e:
        print_log(f"[PT-pf] CUDA-DIAG error: {e!r}", "_pytorch")


def _inject_face_env(env: Dict[str, str], esrgan_root: Path) -> Dict[str, str]:
    """Setzt GFPGAN_MODEL_PATH falls nicht gesetzt, bevorzugt lokale Gewichte."""
    try:
        if not env.get("GFPGAN_MODEL_PATH"):
            vm_base = (
                esrgan_root.parent
            )  # dein Layout: …/videoManagerApp/real-esrgan → parent = videoManagerApp
            p = resolve_gfpgan_weight(vm_base)
            if p:
                env["GFPGAN_MODEL_PATH"] = str(p)
                print_log(f"[face] GFPGAN_MODEL_PATH = {p}", "_pytorch")
    except Exception as e:
        print_log(f"[face] inject env failed: {e!r}", "_pytorch")
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Title/Progress-String – UI Helfer
# ─────────────────────────────────────────────────────────────────────────────
def _pf_phase_title(
    *,
    chunk_idx: int,
    chunks_total: int,
    phase: str,
    backend: str,
    model: str,
    scale: float,
    workers: int,
    tile: int,
    fp32: bool,
    tta: bool,
    finished: int,
    total: int,
) -> str:
    return ab.fmt_phase_title(
        chunk_idx=chunk_idx,
        chunks_total=chunks_total,
        phase=phase,
        backend=backend,
        model=model,
        scale=scale,
        workers=workers,
        tile=tile,
        fp32=fp32,
        tta=tta,
        finished=finished,
        total=total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Args für EINEN Frame bauen (neutral: -n nur falls CLI es kann; Weights injizieren)
# ─────────────────────────────────────────────────────────────────────────────
def _pf_build_single_frame_args(
    *,
    help_text: str,
    model: str,
    input_path: Path,
    in_flag: Optional[str],
    out_dir: Path,
    face_enhance: bool,
    denoise: Optional[float],
    scale: float,
    force_fp32: bool,
    tile_size: int,
    tile_pad: int,
    tta: bool,
    gpu_flag: Optional[str],
    cuda_mask: Optional[str],
    weights_path: Optional[Path],
) -> List[str]:
    args_common = ab.build_realesrgan_cli_common(
        help_text,
        model,
        out_dir=out_dir,
        face_enhance=face_enhance,
        denoise=denoise,
        scale=scale,
        include_ext_and_suffix=True,
        force_fp32=force_fp32,
        tile_size=tile_size,
        tile_pad=tile_pad,
        tta=tta,  # für per-Frame immer False; TTA(emul) läuft separat
    )
    args = ([in_flag, str(input_path)] if in_flag else []) + args_common

    # GPU-Flag nur, wenn verifiziert & nicht CPU-Maske
    if gpu_flag and (cuda_mask is None or str(cuda_mask).strip().lower() != "cpu"):
        args += [gpu_flag, "0"]

    # Gewichtspfad neutral injizieren (Runner bevorzugt --weights)
    if weights_path and weights_path.is_file():
        args = ensure_cli_has_weightish(
            help_text, args, model_name=model, model_path=weights_path
        )

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Spawn eines einzelnen Frame-Prozesses (Hintergrund, mit Logfile)
# ─────────────────────────────────────────────────────────────────────────────
def _pf_spawn_single_frame(
    *,
    venv_python: Path,
    runner: Path,
    args: List[str],
    esrgan_root: Path,
    cuda_mask: Optional[str],
    log_path: Optional[Path],
) -> subprocess.Popen[str]:
    return _popen_with_cudnn(
        venv_python,
        runner,
        args,
        esrgan_root,
        env=ab.build_env_for_gpu(cuda_mask),
        log_path=log_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retry eines einzelnen Frames (blocking) – ENV-only, ohne explizites GPU-Flag
# ─────────────────────────────────────────────────────────────────────────────
def _pf_retry_single_frame(
    *,
    venv_python: Path,
    runner: Path,
    esrgan_root: Path,
    help_text: str,
    model: str,
    src_path: Path,
    in_flag: Optional[str],
    out_dir: Path,
    face_enhance: bool,
    denoise: Optional[float],
    scale: float,
    tile_try: int,
    fp32_try: bool,
    tile_pad: int,
    cuda_mask: Optional[str],
    weights_path: Optional[Path],
) -> int:
    args = _pf_build_single_frame_args(
        help_text=help_text,
        model=model,
        input_path=src_path,
        in_flag=in_flag,
        out_dir=out_dir,
        face_enhance=face_enhance,
        denoise=denoise,
        scale=scale,
        force_fp32=bool(fp32_try),
        tile_size=int(tile_try or 0),
        tile_pad=tile_pad,
        tta=False,
        gpu_flag=None,  # wichtig: Retries ohne explizites GPU-Flag, nur ENV
        cuda_mask=cuda_mask,
        weights_path=weights_path,
    )
    rc = _run_with_cudnn_completed(
        venv_python, runner, args, esrgan_root, env=ab.build_env_for_gpu(cuda_mask)
    )
    return int(rc.returncode or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Factory: make_args(in_dir, out_dir) für den Pool (Directory-Shards)
# ─────────────────────────────────────────────────────────────────────────────
def _pf_pool_make_args_factory(
    *,
    help_text: str,
    model: str,
    out_dir: Path,
    face_enhance: bool,
    denoise: Optional[float],
    scale: float,
    prefer_fp32: bool,
    tile_size: int,
    tile_pad: int,
    tta: bool,
    gpu_flag: Optional[str],
    cuda_mask: Optional[str],
    weights_path: Optional[Path],
    in_flag: Optional[str],
):
    import shlex

    global_up_dir = out_dir  # LEGACY-KONTRAKT: alle Worker schreiben direkt hierher

    def _make_args(in_dir: Path, _ignored_out_dir_local: Path) -> List[str]:
        common = ab.build_realesrgan_cli_common(
            help_text,
            model,
            out_dir=global_up_dir,  # ← FIX: gemeinsame Out-Dir
            face_enhance=face_enhance,
            denoise=denoise,
            scale=scale,
            include_ext_and_suffix=True,
            force_fp32=bool(prefer_fp32),
            tile_size=int(tile_size or 0),
            tile_pad=int(tile_pad),
            tta=bool(tta),
        )

        # Eingabe-Flag nur, wenn die CLI eins verlangt
        args = ([in_flag, str(in_dir)] if in_flag else []) + common

        # Optionales GPU-Flag (nur wenn gültig + nicht CPU-Maske)
        if gpu_flag and (cuda_mask is None or str(cuda_mask).strip().lower() != "cpu"):
            args += [gpu_flag, "0"]

        # Gewichte robust injizieren
        if weights_path and weights_path.is_file():
            args = ensure_cli_has_weightish(
                help_text, args, model_name=model, model_path=weights_path
            )

        print_log(
            f"[PT-pf] POOL make_args in='{in_dir}' out='{global_up_dir}' → args={shlex.join(map(str, args))}",
            "_pytorch",
        )
        return args

    return _make_args


# ─────────────────────────────────────────────────────────────────────────────
# Neutral: Weights/Model-Pfad sicher anfügen
# (Signatur exakt wie von dir vorgegeben)
# ─────────────────────────────────────────────────────────────────────────────
def ensure_cli_has_weightish(
    help_text: str,
    argv: List[str],
    *,
    model_name: Optional[str] = None,
    model_path: Optional[Path] = None,
) -> List[str]:
    """
    Injezieren eines Weight-Arguments NUR wenn:
      - CLI es unterstützt, und
      - model_path existiert, und
      - der Pfad *zum Modell passt* (Dateiname enthält den Modellnamen
        bzw. ist in einer erlaubten Kompatibilitätstabelle).
    Bevorzugt '--weights' (Runner), sonst '--model_path' (Upstream).
    """
    ht = help_text or ""
    s = " ".join(map(str, argv))

    has_weights_flag = "--weights" in ht
    has_model_path = ("--model_path" in ht) or ("--model-path" in ht)

    # Schon gesetzt?
    if (
        (" --weights " in f" {s} ")
        or (" --model_path " in f" {s} ")
        or (" --model-path " in f" {s} ")
    ):
        return argv

    # Kein Ziel?
    if not model_path or not model_path.is_file():
        return argv

    # Modell-Kompatibilität prüfen
    def _norm(x: str) -> str:
        return x.lower().replace("_", "-")

    mp = _norm(model_path.name)
    mn = _norm(model_name or "")

    # erlaubte Paare (z. B. x4v3 kann auch wdn-x4v3 enthalten, da DNI möglich)
    compat_ok = False
    if mn:
        if mn in mp:
            compat_ok = True
        else:
            # kleine Kompatibilitätstabelle
            allowed: dict[str, list[str]] = {
                "realesr-general-x4v3": [
                    "realesr-general-x4v3",
                    "realesr-general-wdn-x4v3",
                ],
                "realesrgan-x4plus": ["realesrgan-x4plus", "real-esrgan-x4plus"],
                "realesrgan-x2plus": ["realesrgan-x2plus", "real-esrgan-x2plus"],
                "realesrgan-x4plus-anime-6b": ["anime-6b"],
                "realesr-animevideov3": ["realesr-animevideov3"],
            }
            for key, subs in allowed.items():
                if key in mn and any(sub in mp for sub in subs):
                    compat_ok = True
                    break

    if mn and not compat_ok:
        print_log(
            f"[PT-util] skip weight injection: model='{model_name}' vs file='{model_path.name}'",
            "_pytorch",
        )
        return argv

    if has_weights_flag:
        print_log(f"[PT-util] inject --weights {model_path}", "_pytorch")
        return [*argv, "--weights", str(model_path)]
    if has_model_path:
        print_log(f"[PT-util] inject --model_path {model_path}", "_pytorch")
        return [*argv, "--model_path", str(model_path)]

    print_log("[PT-util] no supported weight flag in help → skip injection", "_pytorch")
    return argv


def _resolve_weights_for_model(
    model: str, vm_base: Optional[Path] = None
) -> Optional[Path]:
    """
    Sucht gezielt nach dem *passenden* .pth zum gewünschten Modell.
    Durchsucht u. a.: $VM_BASE/real-esrgan/weights, $VM_BASE/weights, $VM_BASE/models,
    site-packages/realesrgan/weights, ~/.cache/{realesrgan,torch/hub/checkpoints}.
    """
    want = {
        "realesr-animevideov3": ["realesr-animevideov3.pth"],
        "realesr-general-x4v3": [
            "realesr-general-x4v3.pth",
            "realesr-general-wdn-x4v3.pth",
        ],  # wir liefern *einen* Pfad zurück (Runner kann DNI selber finden)
        "realesrgan-x4plus": ["RealESRGAN_x4plus.pth", "realesrgan-x4plus.pth"],
        "realesrgan-x2plus": ["RealESRGAN_x2plus.pth", "realesrgan-x2plus.pth"],
        "realesrgan-x4plus-anime-6b": [
            "RealESRGAN_x4plus_anime_6B.pth",
            "realesrgan-x4plus-anime-6b.pth",
        ],
    }
    model = model.strip()
    targets = want.get(model, [])
    if not targets:
        return None

    bases: list[Path] = []
    if vm_base:
        bases += [
            vm_base / "real-esrgan" / "weights",
            vm_base / "weights",
            vm_base / "models",
        ]
    sp = _site_pkg_weights_dir()
    if sp:
        bases.append(sp)
    bases += [
        Path.home() / ".cache/realesrgan",
        Path.home() / ".cache/torch/hub/checkpoints",
    ]

    for b in bases:
        for name in targets:
            p = b / name
            if p.is_file():
                print_log(f"[PT-weights] {model} → {p}", "_pytorch")
                return p

    print_log(f"[PT-weights] {model}: no local weights found", "_pytorch")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HAUPTFUNKTION (repariert & verschlankt)
# ─────────────────────────────────────────────────────────────────────────────
def run_esrgan_per_frame_python(
    venv_python: Path,
    esr_script: Path,
    esrgan_root: Path,
    model: str,
    raw_dir: Path,
    up_dir: Path,
    *,
    face_enhance: bool,
    denoise: Optional[float],
    outscale: float = 4.0,
    tta: bool = False,
    prefer_fp32: Optional[bool] = None,
    initial_tile: Optional[int] = None,
    cuda_mask: Optional[str] = None,
    chunk_idx: Optional[int] = 0,
    chunks_total: Optional[int] = None,
    user_worker_profile: Optional[str] = None,
    ui_phase_id: Optional[int] = None,
    hooks: Optional[Dict[str, Callable[..., None]]] = None,
) -> bool:
    try:
        import shutil
        from collections import deque

        mg.install_global_cancel_handlers()

        def _hk(name: str, **payload) -> None:
            try:
                if hooks and callable(hooks.get(name)):
                    hooks[name](**payload)
            except Exception:
                pass

        # Bevorzugt den Projekt-Runner
        runner = _resolve_runner(esr_script)

        retries = int(getattr(defin, "AI_PER_FRAME_RETRIES", 6) or 6)
        print_log(
            f"[PT-pf] ENTER model={model} scale={outscale} face={face_enhance} denoise={denoise} "
            f"tta_req={tta} init_tile={initial_tile} chunk={chunk_idx}/{chunks_total} "
            f"cuda_mask={cuda_mask} retries={retries} pid={os.getpid()} cwd={os.getcwd()} runner={runner}",
            "_pytorch",
        )

        # Sichtbares Env (Kurz-Dump)
        env_probe_keys = [
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_MODULE_LOADING",
            "TORCH_ALLOW_TF32",
            "NVIDIA_TF32_OVERRIDE",
            "AI_POOL_DISABLE",
            "AI_POOL_FORCE",
            "AI_POOL_TTA",
            "AI_WORKERS",
            "AI_WORKERS_MIN",
            "AI_WORKERS_MAX",
            "AI_AUTO_HEADROOM",
            "ESRGAN_FORCE_FP32",
            "ESRGAN_MODEL_PATH",
            "ESRGAN_DEFAULT_MODEL_PATH",
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
        ]
        env_dump = {k: os.environ.get(k, None) for k in env_probe_keys}
        print_log(f"[PT-pf] ENV {json.dumps(env_dump, indent=2)}", "_pytorch")
        print_log(
            f"[PT-pf] BINARIES venv_python={venv_python} runner={runner}", "_pytorch"
        )

        # CUDA/Torch Diagnose
        _cuda_diag_for_log(venv_python)

        if mg.is_cancelled():
            _hk("cancelled", where="begin")
            print_log("[PT-pf] early cancel before setup", "_pytorch")
            return False

        t0 = time.time()
        input_dir = ab.ensure_png_inputs_for_tool(raw_dir, tmp_root=raw_dir.parent)
        ab.log_paths_state(
            title="torch-per-frame: pre-run",
            raw_dir=raw_dir,
            input_dir=input_dir,
            up_dir=up_dir,
            esrgan_root=esrgan_root,
        )

        help_text = ab.probe_realesrgan_help(venv_python, runner)
        ab.dump_flag_diagnostics("torch-per-frame", help_text)

        # Optional: GFPGAN Warmup (CPU), um Lazy-Downloads vor Parallelstart zu triggern
        if face_enhance:
            try:
                _gfpgan_warmup_once(
                    venv_python=venv_python,
                    esr_script=runner,
                    esrgan_root=esrgan_root,
                    model=model,
                    help_text=help_text,
                )
                print_log("[PT-pf] GFPGAN warmup done (cpu)", "_pytorch")
            except Exception as _e:
                co.print_warning(f"[GFPGAN] Warm-Up skipped: {_e}")
                print_log(f"[PT-pf] GFPGAN warmup skipped: {_e!r}", "_pytorch")

        inputs = sorted(input_dir.glob("frame_*.png"), key=ih.frame_index_from_name)
        total = len(inputs)
        if total == 0:
            print_log("[PT-pf] no inputs found → abort", "_pytorch")
            _hk(
                "end",
                success=False,
                produced=0,
                total=0,
                duration=0.0,
                reason="no_inputs",
            )
            return False

        cuda_ok, dev_name, vram_mb = ab.detect_gpu_info(venv_python)
        torch_v, cuda_v = ab.torch_versions(venv_python)
        print_log(
            f"[PT-pf] DEVICE cuda_ok={cuda_ok} dev='{dev_name}' vram={vram_mb}MB torch={torch_v} torch.cuda={cuda_v} "
            f"CUDA_VISIBLE_DEVICES={ab.env_cuda_mask()} cuda_mask={cuda_mask}",
            "_pytorch",
        )

        scale = max(1.0, float(outscale))
        if prefer_fp32 is None:
            prefer_fp32 = ab.prefer_fp32_default()
        tile_size = (
            initial_tile
            if (isinstance(initial_tile, int) and initial_tile >= 0)
            else ab.pick_tile_size(vram_mb, bool(prefer_fp32), int(round(scale)))
        )
        tile_pad = 10
        input_mp = ih.estimate_input_megapixels(input_dir)

        norm_profile = ab.normalize_worker_profile_name(user_worker_profile or "auto")
        base_workers = ab.suggest_parallel_workers(
            vram_mb=vram_mb,
            tile_size=int(tile_size or 0),
            prefer_fp32=bool(prefer_fp32),
            total_frames=total,
            input_megapixels=input_mp,
            scale=int(round(scale)),
            model=model,
            face_enhance=face_enhance,
            user_profile=user_worker_profile,
        )
        emulate_tta = bool(tta and at.can_emulate_tta_x8())
        workers = (
            base_workers
            if not emulate_tta
            else at.adjust_workers_for_emulated_tta(
                base_workers,
                vram_mb,
                honor_min=(norm_profile in ("min", "minimal", "serial")),
                user_profile=norm_profile,
            )
        )
        if norm_profile in ("serial", "min", "minimal", "no_parallelisation"):
            workers = 1

        # Flags aus Help
        in_flag = ab.pick_flag(help_text, ["-i", "--input"], "-i")
        gpu_flag = _gpu_flag_if_supported(
            venv_python, runner, help_text
        )  # nur wenn ESRGAN_USE_GPU_FLAG=1

        up_dir.mkdir(parents=True, exist_ok=True)
        weights = _resolve_esrgan_pytorch_weights(esrgan_root.parent)
        weights_path = weights[1] if weights else None

        if weights:
            print_log(
                f"[PT-pf] weights resolved name={weights[0]} path={weights[1]}",
                "_pytorch",
            )
        else:
            print_log(
                "[PT-pf] WARNING no weights resolved (will rely on defaults)",
                "_pytorch",
            )

        # ─────────────────────────────────────────────────────────────────────
        # NEU: PERSISTENTER POOL + TTA (emuliert, x8)
        # ─────────────────────────────────────────────────────────────────────
        shard_min = int(os.environ.get("AI_POOL_SHARD_MIN", "80") or 80)
        use_pool_env_off = str(os.environ.get("AI_POOL_DISABLE", "")).strip() == "1"
        use_pool_env_force = str(os.environ.get("AI_POOL_FORCE", "")).strip() == "1"

        use_tta_pool = bool(
            emulate_tta
            and (not use_pool_env_off)
            and (use_pool_env_force or (workers >= 2 and total >= shard_min))
        )

        if use_tta_pool:
            print_log(
                f"[PT-pf] TTA-POOL start workers={workers} total={total} shard_min={shard_min}",
                "_pytorch",
            )

            # 1) TTA-Varianten vorbereiten (Identität via Hardlinks, andere via Transform)
            tta_tmp_root = up_dir / "__tta_tmp__"
            variants = at.prepare_tta_variant_dirs(
                input_dir=input_dir, tmp_root=tta_tmp_root
            )
            if not variants:
                print_log(
                    "[PT-pf] TTA-POOL: keine Varianten erzeugt → Fallback per-frame",
                    "_pytorch",
                )
            else:
                # UI: zwei Balken falls mehrere Chunks, Fortschritt über (var_idx/8) × Frames
                try:
                    term_cols = shutil.get_terminal_size((80, 20)).columns
                except Exception:
                    term_cols = 80
                bar_len = max(20, min(80, max(20, term_cols - len(" 100% []"))))
                two_bars = isinstance(chunks_total, int) and (chunks_total or 0) > 1
                cur_chunk = (chunk_idx or 0) + 1
                nvars = len(variants)

                def _draw_var_progress(var_idx: int, done: int, tot_local: int) -> None:
                    frac_vars = (
                        var_idx
                        - 1
                        + (0 if tot_local <= 0 else done / max(1, tot_local))
                    ) / max(1, nvars)
                    title = ab.fmt_phase_title(
                        chunk_idx=cur_chunk,
                        chunks_total=(chunks_total or 1),
                        phase=f"Upscaling (Pool • TTA {var_idx}/{nvars})",
                        backend="PyTorch",
                        model=model,
                        scale=scale,
                        workers=workers,
                        tile=int(tile_size or 0),
                        fp32=bool(prefer_fp32),
                        tta=True,
                        finished=min(done, tot_local),
                        total=tot_local,
                    )
                    if two_bars:
                        top, _t = gh.make_bar(
                            max(
                                0.0,
                                min(
                                    (cur_chunk - 1 + frac_vars)
                                    / float(chunks_total or 1),
                                    1.0,
                                ),
                            ),
                            bar_len,
                        )
                    else:
                        top = None
                    bot, _t = gh.make_bar(frac_vars, bar_len)
                    gh.draw_chunk_block_cond(
                        two_bars=two_bars,
                        title=title,
                        top_bar=top,
                        bot_bar=bot,
                        hint=_("cancel_hint"),
                        ui_phase_id=ui_phase_id,
                    )

                # Gemeinsames ENV; Guard gegen geerbtes '' für CUDA
                env = _inject_face_env(
                    ab.build_esrgan_env(ab.build_env_for_gpu(cuda_mask)), esrgan_root
                )
                if cuda_mask is None and env.get("CUDA_VISIBLE_DEVICES", "") == "":
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                    print_log(
                        "[PT-pf] TTA-POOL env fix: removed CUDA_VISIBLE_DEVICES='' (GPU sichtbar)",
                        "_pytorch",
                    )

                for i, var in enumerate(variants, 1):
                    if mg.is_cancelled():
                        break

                    # args-Factory: Ausgabe **in die Varianten-out/**, nicht direkt ins up_dir!
                    make_args = _pf_pool_make_args_factory(
                        help_text=help_text,
                        model=model,
                        out_dir=var.out_dir,
                        face_enhance=face_enhance,
                        denoise=denoise,
                        scale=scale,
                        prefer_fp32=bool(prefer_fp32),
                        tile_size=int(tile_size or 0),
                        tile_pad=20,
                        tta=False,  # wichtig: KEIN --tta (wir emulieren)
                        gpu_flag=gpu_flag,
                        cuda_mask=cuda_mask,
                        weights_path=weights_path,
                        in_flag=in_flag,
                    )

                    def _progress_cb_tta(done: int, total_local: int, _vi=i):
                        # leicht gedrosseltes Log + UI-Update
                        if total_local > 0:
                            p = int(100 * done / max(1, total_local))
                            if (
                                p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
                                or done == total_local
                            ):
                                print_log(
                                    f"[PT-pf] TTA-POOL var {_vi}/{nvars}: {done}/{total_local} ({p}%)",
                                    "_pytorch",
                                )
                        _draw_var_progress(_vi, done, total_local)

                    ok_var = run_sharded_dir_job_with_retries(
                        venv_python=venv_python,
                        esr_script=runner,
                        esrgan_root=esrgan_root,
                        in_dir=var.in_dir,
                        out_dir=var.out_dir,
                        make_args=make_args,
                        total_frames=total,
                        workers=max(2, int(workers)),
                        env=env,
                        progress_cb=_progress_cb_tta,
                    )
                    print_log(
                        f"[PT-pf] TTA-POOL var {i}/{nvars} result={ok_var}", "_pytorch"
                    )

                # 3) Fuse: Varianten mitteln → up_dir/stem_out.png
                fuse_title = ab.fmt_phase_title(
                    chunk_idx=cur_chunk,
                    chunks_total=(chunks_total or 1),
                    phase="Upscaling (Pool • TTA Fuse)",
                    backend="PyTorch",
                    model=model,
                    scale=scale,
                    workers=workers,
                    tile=int(tile_size or 0),
                    fp32=bool(prefer_fp32),
                    tta=True,
                )
                at.fuse_tta_pool_variants_to_updir(
                    variants=variants,
                    up_dir=up_dir,
                    input_dir=input_dir,
                    esrgan_root=esrgan_root,
                    ui_title=fuse_title,
                    two_bars=two_bars,
                    cur_chunk=cur_chunk,
                    chunks_total=(chunks_total or 1),
                    ui_phase_id=ui_phase_id,
                )  # → Anzahl erfolgreich gefuseter Frames

                # 4) Normalisieren + Erfolg prüfen
                normalized = ab.normalize_upscaled_sequence(
                    up_dir,
                    input_dir,
                    esrgan_root,
                    expected_total=total,
                    created_after=t0,
                )
                seq_count = len(list(up_dir.glob("frame_*.png")))
                success = (normalized >= total) or (seq_count >= total)

                # Optional: TTA-Tmp aufräumen
                try:
                    shutil.rmtree(tta_tmp_root, ignore_errors=True)
                except Exception:
                    pass

                if success:
                    ab.persist_context(input_dir, up_dir, tag="torch-tta-pool-ok")
                    _hk(
                        "end",
                        success=True,
                        produced=max(normalized, seq_count),
                        total=total,
                        duration=(time.time() - t0),
                        mode="tta_pool",
                    )
                    return True

                # Diagnose + Fallback per-frame
                print_log(
                    "[PT-pf] TTA-POOL normalization insufficient → Fallback per-frame …",
                    "_pytorch",
                )
                ab.debug_dump_pool_failure(
                    up_dir=up_dir,
                    input_dir=input_dir,
                    esrgan_root=esrgan_root,
                    tag="torch-tta-pool-fail",
                    also_cat_logs=True,
                )
                _dump_pool_shard_logs(up_dir)
                ab.persist_context(input_dir, up_dir, tag="torch-tta-pool-fail")
            # Ende use_tta_pool
        # ─────────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────────
        # PERSISTENTER POOL (ohne TTA-Emulation)
        # ─────────────────────────────────────────────────────────────────────
        if (
            (not emulate_tta)
            and not use_pool_env_off
            and (use_pool_env_force or (workers >= 2 and total >= shard_min))
        ):
            print_log(
                f"[PT-pf] POOL start workers={workers} total={total} shard_min={shard_min}",
                "_pytorch",
            )

            # UI Setup für den Pool-Fortschritt
            try:
                term_cols = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                term_cols = 80
            bar_len = max(20, min(80, max(20, term_cols - len(" 100% []"))))
            two_bars = isinstance(chunks_total, int) and (chunks_total or 0) > 1
            cur_chunk = (chunk_idx or 0) + 1

            def _draw_pool_progress(done: int, total_local: int) -> None:
                # Titel & Balken zeichnen
                title = ab.fmt_phase_title(
                    chunk_idx=cur_chunk,
                    chunks_total=(chunks_total or 1),
                    phase="Upscaling (Pool)",
                    backend="PyTorch",
                    model=model,
                    scale=scale,
                    workers=workers,
                    tile=int(tile_size or 0),
                    fp32=bool(prefer_fp32),
                    tta=False,
                    finished=min(done, total_local),
                    total=total_local,
                )
                hint = _("cancel_hint")
                if two_bars:
                    top0, _t = gh.make_bar(
                        max(0.0, min((cur_chunk - 1) / float(chunks_total or 1), 1.0)),
                        bar_len,
                    )
                else:
                    top0 = None
                p = 0.0 if total_local <= 0 else done / max(1, total_local)
                bot, _t = gh.make_bar(p, bar_len)
                gh.draw_chunk_block_cond(
                    two_bars=two_bars,
                    title=title,
                    top_bar=top0,
                    bot_bar=bot,
                    hint=hint,
                    ui_phase_id=ui_phase_id,
                )

            try:
                make_args = _pf_pool_make_args_factory(
                    help_text=help_text,
                    model=model,
                    out_dir=up_dir,
                    face_enhance=face_enhance,
                    denoise=denoise,
                    scale=scale,
                    prefer_fp32=bool(prefer_fp32),
                    tile_size=int(tile_size or 0),
                    tile_pad=20,
                    tta=False,
                    gpu_flag=gpu_flag,
                    cuda_mask=cuda_mask,
                    weights_path=weights_path,
                    in_flag=in_flag,
                )

                env = _inject_face_env(
                    ab.build_esrgan_env(ab.build_env_for_gpu(cuda_mask)), esrgan_root
                )

                # Guard gegen geerbtes leeres ''
                if cuda_mask is None and env.get("CUDA_VISIBLE_DEVICES", "") == "":
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                    print_log(
                        "[PT-pf] POOL env fix: removed inherited CUDA_VISIBLE_DEVICES='' (GPU visible again)",
                        "_pytorch",
                    )

                ab.persist_context(input_dir, up_dir, tag="torch-pool-begin")

                def _progress_cb(done: int, total_local: int) -> None:
                    # Log + UI
                    if total_local > 0:
                        p = int(100 * done / max(1, total_local))
                        if (
                            p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
                            or done == total_local
                        ):
                            print_log(
                                f"[PT-pf] POOL progress {done}/{total_local} ({p}%)",
                                "_pytorch",
                            )
                    _draw_pool_progress(done, total_local)

                ok_pool = run_sharded_dir_job_with_retries(
                    venv_python=venv_python,
                    esr_script=runner,
                    esrgan_root=esrgan_root,
                    in_dir=input_dir,
                    out_dir=up_dir,
                    make_args=make_args,
                    total_frames=total,
                    workers=max(2, int(workers)),
                    env=env,
                    progress_cb=_progress_cb,
                )
                print_log(f"[PT-pf] POOL result={ok_pool}", "_pytorch")

                # Direkt normalisieren (nur *_out.png im gemeinsamen up_dir)
                if ok_pool:
                    normalized = ab.normalize_upscaled_sequence(
                        up_dir,
                        input_dir,
                        esrgan_root,
                        expected_total=total,
                        created_after=t0,
                    )
                    seq_count = len(list(up_dir.glob("frame_*.png")))
                    success = (normalized >= total) or (seq_count >= total)

                    if success:
                        ab.persist_context(input_dir, up_dir, tag="torch-pool-ok")
                        _hk(
                            "end",
                            success=True,
                            produced=max(normalized, seq_count),
                            total=total,
                            duration=(time.time() - t0),
                            mode="pool",
                        )
                        return True

                    # Falls nicht ausreichend: Diagnose + Fallback
                    print_log(
                        "[PT-pf] POOL normalization insufficient → fallback per-frame …",
                        "_pytorch",
                    )
                    ab.debug_dump_pool_failure(
                        up_dir=up_dir,
                        input_dir=input_dir,
                        esrgan_root=esrgan_root,
                        tag="torch-pool-fail",
                        also_cat_logs=True,
                    )
                    _dump_pool_shard_logs(up_dir)
                    ab.persist_context(input_dir, up_dir, tag="torch-pool-fail")

            except Exception as e:
                co.print_warning(f"[Py-pf] POOL exception → fallback: {e!r}")
                print_log(
                    f"[PT-pf] POOL exception → fallback: {type(e).__name__}: {e!r}",
                    "_pytorch",
                )

        # ─────────────────────────────────────────────────────────────────────
        # PER-FRAME (Fallback oder wenn Pool nicht genutzt werden soll)
        # ─────────────────────────────────────────────────────────────────────
        try:
            term_cols = shutil.get_terminal_size((80, 20)).columns
        except Exception:
            term_cols = 80
        bar_len = max(20, min(80, max(20, term_cols - len(" 100% []"))))
        two_bars = isinstance(chunks_total, int) and (chunks_total or 0) > 1
        cur_chunk = (chunk_idx or 0) + 1
        tta_lbl = " + TTA(emul)" if emulate_tta else ""

        def _draw_progress(finished: int) -> None:
            total_done = min(total, finished)
            title = _pf_phase_title(
                chunk_idx=cur_chunk,
                chunks_total=(chunks_total or 1),
                phase="Upscaling (Per-Frame)",
                backend=f"PyTorch{tta_lbl}",
                model=model,
                scale=scale,
                workers=workers,
                tile=int(tile_size or 0),
                fp32=bool(prefer_fp32),
                tta=emulate_tta,
                finished=total_done,
                total=total,
            )
            hint = _("cancel_hint")
            if two_bars:
                top0, _t = gh.make_bar(
                    max(0.0, min((cur_chunk - 1) / float(chunks_total or 1), 1.0)),
                    bar_len,
                )
            else:
                top0 = None
            bot, _t = gh.make_bar(total_done / max(1, total), bar_len)
            gh.draw_chunk_block_cond(
                two_bars=two_bars,
                title=title,
                top_bar=top0,
                bot_bar=bot,
                hint=hint,
                ui_phase_id=ui_phase_id,
            )

        print_log(
            (
                "[Py-pf] running without TTA"
                if not emulate_tta
                else "[Py-pf] using emulated TTA x8"
            ),
            "_pytorch",
        )

        queue = deque(inputs)
        running: List[Tuple[Path, subprocess.Popen[str], Path]] = []
        attempts: Dict[str, int] = {}
        ok_outputs = 0
        abort_now = False

        # initiale Spawns
        while queue and len(running) < max(1, int(workers)):
            p = queue.popleft()
            log_path = up_dir / "__logs__" / f"{p.stem}.log"
            args = _pf_build_single_frame_args(
                help_text=help_text,
                model=model,
                input_path=p,
                in_flag=in_flag,
                out_dir=up_dir,
                face_enhance=face_enhance,
                denoise=denoise,
                scale=scale,
                force_fp32=bool(prefer_fp32),
                tile_size=int(tile_size or 0),
                tile_pad=int(tile_pad),
                tta=False,
                gpu_flag=gpu_flag,
                cuda_mask=cuda_mask,
                weights_path=weights_path,
            )
            _hk("frame_start", stem=p.stem, path=str(p), mode="per_frame")
            print_log(
                f"[PT-pf] spawn {p.name} args={' '.join(map(str, args))}", "_pytorch"
            )
            proc = _pf_spawn_single_frame(
                venv_python=venv_python,
                runner=runner,
                args=args,
                esrgan_root=esrgan_root,
                cuda_mask=cuda_mask,
                log_path=log_path,
            )
            running.append((p, proc, log_path))

        finished = 0
        with mg.escape_cancel_guard(nonintrusive=True):
            while (running or queue) and not abort_now:
                if mg.is_cancelled():
                    abort_now = True
                    mg.kill_all()
                    _hk("cancelled", where="per_frame_loop")
                    print_log("[PT-pf] cancel inside per-frame loop", "_pytorch")
                    break

                new_running: List[Tuple[Path, subprocess.Popen[str], Path]] = []
                for src_path, proc, lp in running:
                    ret = proc.poll()
                    if ret is None:
                        new_running.append((src_path, proc, lp))
                        continue

                    ok_here = False
                    out_file = up_dir / f"{src_path.stem}_out.png"
                    if out_file.exists():
                        ok_here = True
                    else:
                        relocated = ab.relocate_output_for_frame(
                            stem=src_path.stem,
                            up_dir=up_dir,
                            input_dir=input_dir,
                            esrgan_root=esrgan_root,
                            raw_dir=raw_dir,
                        )
                        if relocated and relocated.exists():
                            ok_here = True

                    if not ok_here:
                        # Retry-Leiter (ENV-only; ohne explizites GPU-Flag)
                        tile_candidates = [int(tile_size or 0)]
                        if tile_candidates[0] <= 0:
                            tile_candidates = [640, 512, 384, 320, 256, 192, 128]
                        else:
                            tile_candidates += [
                                max(64, tile_candidates[0] // 2),
                                256,
                                192,
                                128,
                            ]

                        tried = set()
                        for t_try in tile_candidates:
                            for fp_try in (bool(prefer_fp32), True):
                                key = (t_try, fp_try)
                                if key in tried:
                                    continue
                                tried.add(key)
                                co.print_info(
                                    f"[PER-FRAME] Retry {src_path.name} mit fp32={'yes' if fp_try else 'no'} tile={ab.tile_label(t_try)} …"
                                )
                                rc = _pf_retry_single_frame(
                                    venv_python=venv_python,
                                    runner=runner,
                                    esrgan_root=esrgan_root,
                                    help_text=help_text,
                                    model=model,
                                    src_path=src_path,
                                    in_flag=in_flag,
                                    out_dir=up_dir,
                                    face_enhance=face_enhance,
                                    denoise=denoise,
                                    scale=scale,
                                    tile_try=int(t_try),
                                    fp32_try=bool(fp_try),
                                    tile_pad=int(tile_pad),
                                    cuda_mask=cuda_mask,
                                    weights_path=weights_path,
                                )
                                print_log(
                                    f"[PT-pf] RETRY rc={rc} for {src_path.name}",
                                    "_pytorch",
                                )
                                if (up_dir / f"{src_path.stem}_out.png").exists():
                                    ok_here = True
                                    break
                                relocated = ab.relocate_output_for_frame(
                                    stem=src_path.stem,
                                    up_dir=up_dir,
                                    input_dir=input_dir,
                                    esrgan_root=esrgan_root,
                                    raw_dir=raw_dir,
                                )
                                if relocated and relocated.exists():
                                    ok_here = True
                                    break
                            if ok_here:
                                break

                    if not ok_here:
                        key = src_path.stem
                        attempts[key] = attempts.get(key, 0) + 1
                        if attempts[key] < retries:
                            co.print_warning(
                                f"[PER-FRAME] erneuter Versuch {src_path.name} ({attempts[key]}/{retries}) …"
                            )
                            log_path = up_dir / "__logs__" / f"{src_path.stem}.log"
                            args = _pf_build_single_frame_args(
                                help_text=help_text,
                                model=model,
                                input_path=src_path,
                                in_flag=in_flag,
                                out_dir=up_dir,
                                face_enhance=face_enhance,
                                denoise=denoise,
                                scale=scale,
                                force_fp32=bool(prefer_fp32),
                                tile_size=int(tile_size or 0),
                                tile_pad=int(tile_pad),
                                tta=False,
                                gpu_flag=gpu_flag,
                                cuda_mask=cuda_mask,
                                weights_path=weights_path,
                            )
                            proc = _pf_spawn_single_frame(
                                venv_python=venv_python,
                                runner=runner,
                                args=args,
                                esrgan_root=esrgan_root,
                                cuda_mask=cuda_mask,
                                log_path=log_path,
                            )
                            new_running.append((src_path, proc, log_path))
                            continue
                        co.print_error(
                            f"[PER-FRAME] {src_path.name} nach {attempts[key]} Versuchen fehlgeschlagen → Abbruch."
                        )
                        _hk(
                            "frame_done",
                            stem=src_path.stem,
                            ok=False,
                            path=str(out_file),
                            attempts=attempts[key],
                            mode="per_frame",
                        )
                        abort_now = True
                        break
                    else:
                        ok_outputs += 1
                        _hk(
                            "frame_done",
                            stem=src_path.stem,
                            ok=True,
                            path=str(out_file),
                            mode="per_frame",
                        )

                    finished += 1

                running = new_running
                while queue and len(running) < max(1, int(workers)) and not abort_now:
                    p = queue.popleft()
                    log_path = up_dir / "__logs__" / f"{p.stem}.log"
                    args = _pf_build_single_frame_args(
                        help_text=help_text,
                        model=model,
                        input_path=p,
                        in_flag=in_flag,
                        out_dir=up_dir,
                        face_enhance=face_enhance,
                        denoise=denoise,
                        scale=scale,
                        force_fp32=bool(prefer_fp32),
                        tile_size=int(tile_size or 0),
                        tile_pad=int(tile_pad),
                        tta=False,
                        gpu_flag=gpu_flag,
                        cuda_mask=cuda_mask,
                        weights_path=weights_path,
                    )
                    proc = _pf_spawn_single_frame(
                        venv_python=venv_python,
                        runner=runner,
                        args=args,
                        esrgan_root=esrgan_root,
                        cuda_mask=cuda_mask,
                        log_path=log_path,
                    )
                    running.append((p, proc, log_path))

                _draw_progress(finished)
                time.sleep(0.02)

        if abort_now:
            mg.kill_all()
            _hk(
                "end",
                success=False,
                produced=ok_outputs,
                total=total,
                duration=(time.time() - t0),
                mode="per_frame",
            )
            print_log(f"[PT-pf] EXIT fail produced={ok_outputs}/{total}", "_pytorch")
            return False

        if mg.is_cancelled():
            _hk("cancelled", where="finalize")
            print_log("[PT-pf] cancel at finalize", "_pytorch")
            return False

        produced_any = ih.count_produced_frames(up_dir, input_dir, esrgan_root)
        print_log(
            f"[PT-pf] finalize: produced_any={produced_any} expected={total}",
            "_pytorch",
        )
        if produced_any >= total:
            ab.normalize_upscaled_sequence(
                up_dir, input_dir, esrgan_root, expected_total=total, created_after=t0
            )
            success = True
        else:
            normalized = ab.normalize_upscaled_sequence(
                up_dir, input_dir, esrgan_root, expected_total=total, created_after=t0
            )
            seq_count = len(list((up_dir).glob("frame_*.png")))
            success = (
                (normalized >= total) or (seq_count >= total) or (produced_any >= total)
            )
            print_log(
                f"[PT-pf] finalize-check normalized={normalized} seq_count={seq_count} produced_any={produced_any} "
                f"→ success={success}",
                "_pytorch",
            )

        _hk(
            "end",
            success=bool(success),
            produced=max(produced_any, 0),
            total=total,
            duration=(time.time() - t0),
            mode=("tta_emul" if emulate_tta else "per_frame"),
        )
        print_log(
            f"[PT-pf] EXIT success={success} produced={max(produced_any, 0)}/{total}",
            "_pytorch",
        )
        return success

    except KeyboardInterrupt:
        try:
            mg.CANCEL.set()
        except Exception:
            pass
        try:
            mg.kill_all()
        except Exception:
            pass
        co.print_info(_("aborted_by_user"))
        print_log("[PT-pf] KeyboardInterrupt → cancelled", "_pytorch")
        return False


# ==== DROP-IN: run_esrgan_python_dir =========================================


def run_esrgan_python_dir(
    venv_python: Path,
    esr_script: Path,
    esrgan_root: Path,
    model: str,
    raw_dir: Path,
    up_dir: Path,
    *,
    face_enhance: bool,
    denoise: Optional[float],
    tile_size: int = 0,
    force_fp32: bool = False,
    outscale: float = 4.0,
    tta: bool = False,
    cuda_mask: Optional[str] = None,
    ui_phase_id: Optional[int] = None,
) -> bool:
    import shutil

    mg.install_global_cancel_handlers()

    try:
        help_text = ab.probe_realesrgan_help(venv_python, esr_script)
        ab.dump_flag_diagnostics("torch-directory", help_text)
        in_flag = ab.pick_flag(help_text, ["-i", "--input"], "-i")

        # GPU-Flag hier *nicht* erzwingen; Directory-Mode läuft ohnehin mit ENV
        gpu_flag = _gpu_flag_if_supported(venv_python, esr_script, help_text)
        print_log(
            f"[torch-directory] gpu_flag={gpu_flag!r} (ENV preferred)", "_pytorch"
        )

        total = ih.count_raw_frames(raw_dir)
        already = len(list(up_dir.glob("frame_*.png")))
        if already >= max(1, total):
            print_log("[torch-directory] up_dir bereits vollständig – skip")
            try:
                term_cols = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                term_cols = 80
            bar_len = max(20, min(80, max(20, term_cols - len(" 100% []"))))
            title_phase = ab.fmt_phase_title(
                chunk_idx=1,
                chunks_total=1,
                phase="Upscaling (Directory)",
                backend=f"PyTorch{' + TTA' if tta and at.python_supports_tta(venv_python, esr_script) else ''}",
                model=model,
                scale=outscale,
                workers=None,
                tile=tile_size,
                fp32=force_fp32,
                tta=bool(tta),
            )
            bot2, _t = gh.make_bar(1.0, bar_len)
            gh.draw_chunk_block(
                two_bars=False,
                title=title_phase + " • Upscaling abgeschlossen",
                top_bar=None,
                bot_bar=bot2,
                hint=_("cancel_hint"),
            )
            return True

        def _args(_force_fp32: bool, _tile: int) -> List[str]:
            common = ab.build_realesrgan_cli_common(
                help_text,
                model,
                out_dir=up_dir,
                face_enhance=face_enhance,
                denoise=denoise,
                scale=outscale,
                include_ext_and_suffix=True,
                force_fp32=_force_fp32,
                tile_size=_tile,
                tile_pad=20,
                tta=bool(tta),
            )
            args = ([in_flag, str(raw_dir)] if in_flag else []) + common

            weights_path = _resolve_weights_for_model(model, esrgan_root.parent)
            if weights_path:
                args = ensure_cli_has_weightish(
                    help_text, args, model_name=model, model_path=weights_path
                )

            # GPU-Flag *optional*, nur wenn validiert (meist unnötig)
            if gpu_flag and (
                cuda_mask is None or str(cuda_mask).strip().lower() != "cpu"
            ):
                args += [gpu_flag, "0"]
            return args

        print_log(
            f"BackendUsed (plan): torch-directory; fp32={he.yesno(force_fp32)}; tile={ab.tile_label(tile_size)}; outscale={outscale:.2f}; frames≈{total}; tta_req={'yes' if tta else 'no'}"
        )
        print_log(
            f"torch-directory: args_base={' '.join(_args(force_fp32, tile_size))}"
        )

        ab.log_paths_state(
            title="torch-directory: pre-run",
            raw_dir=raw_dir,
            input_dir=raw_dir,
            up_dir=up_dir,
            esrgan_root=esrgan_root,
        )

        try:
            term_cols = shutil.get_terminal_size((80, 20)).columns
        except Exception:
            term_cols = 80
        bar_len = max(20, min(80, max(20, term_cols - len(" 100% []"))))

        title_phase = ab.fmt_phase_title(
            chunk_idx=1,
            chunks_total=1,
            phase="Upscaling (Directory)",
            backend=f"PyTorch{' + TTA' if tta and at.python_supports_tta(venv_python, esr_script) else ''}",
            model=model,
            scale=outscale,
            workers=None,
            tile=tile_size,
            fp32=force_fp32,
            tta=bool(tta),
        )
        bot, _t = gh.make_bar(0.0, bar_len)
        gh.draw_chunk_block(
            two_bars=False,
            title=title_phase,
            top_bar=None,
            bot_bar=bot,
            hint=_("cancel_hint"),
        )

        log_path = (
            up_dir / "__logs__" / "dir_cli.log" if ab.capture_logs_enabled() else None
        )

        with mg.escape_cancel_guard(nonintrusive=True):
            proc = mg.popen(
                [
                    str(venv_python),
                    str(esr_script),
                    *map(str, _args(force_fp32, tile_size)),
                ],
                cwd=esrgan_root,
                text=True,
                env=ab.build_esrgan_env(ab.build_env_for_gpu(cuda_mask)),
                log_to=(str(log_path) if log_path else None),
            )

            results_dir = esrgan_root / "results"
            while proc.poll() is None:
                if mg.is_cancelled():
                    mg.kill_all()
                    break
                try:
                    finished = 0
                    finished += len(list((up_dir).glob("frame_*.png"))) + len(
                        list((up_dir).glob("*_out.png"))
                    )
                    if results_dir.exists():
                        finished += len(list(results_dir.rglob("*_out.png")))
                        finished += len(list(results_dir.rglob("frame_*.png")))
                except Exception:
                    finished = 0
                p = min(finished / max(1, total or 1), 1.0)
                bot, _t = gh.make_bar(p, bar_len)
                gh.draw_chunk_block(
                    two_bars=False,
                    title=title_phase + f" • {finished}/{total}",
                    top_bar=None,
                    bot_bar=bot,
                    hint=_("cancel_hint"),
                )
                time.sleep(0.25)

            try:
                proc.wait(timeout=5)
            except Exception:
                pass

            success = ab.normalize_upscaled_sequence(up_dir, raw_dir, esrgan_root) > 0
            if not success and not mg.is_cancelled():
                print_log(
                    "ESRGAN Directory: erster Versuch fehlgeschlagen → Retry mit FP32/Tiles …"
                )
                proc2 = mg.popen(
                    [
                        str(venv_python),
                        str(esr_script),
                        *map(str, _args(True, tile_size or 256)),
                    ],
                    cwd=esrgan_root,
                    text=True,
                    env=ab.build_esrgan_env(ab.build_env_for_gpu(cuda_mask)),
                    log_to=(str(log_path) if log_path else None),
                )
                while proc2.poll() is None:
                    if mg.is_cancelled():
                        mg.kill_all()
                        break
                    try:
                        finished = len(list((up_dir).glob("frame_*.png")))
                    except Exception:
                        finished = 0
                    p = min(finished / max(1, total or 1), 1.0)
                    bot, _t = gh.make_bar(p, bar_len)
                    gh.draw_chunk_block(
                        two_bars=False,
                        title=title_phase + " • RETRY",
                        top_bar=None,
                        bot_bar=bot,
                        hint=_("cancel_hint"),
                    )
                    time.sleep(0.25)
                success = (
                    ab.normalize_upscaled_sequence(up_dir, raw_dir, esrgan_root) > 0
                )

        bot2, _t = gh.make_bar(1.0 if success else 0.0, bar_len)
        gh.draw_chunk_block(
            two_bars=False,
            title=title_phase
            + f" • {'Upscaling abgeschlossen' if success else 'Upscaling FEHLGESCHLAGEN'}",
            top_bar=None,
            bot_bar=bot2,
            hint=_("cancel_hint"),
        )
        return success

    except KeyboardInterrupt:
        try:
            mg.CANCEL.set()
        except Exception:
            pass
        try:
            mg.kill_all()
        except Exception:
            pass
        co.print_info(_("aborted_by_user"))
        return False
