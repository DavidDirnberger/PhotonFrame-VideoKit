#!/usr/bin/env python3
# ai_ncnn_backend.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# ========== 1) IMPORTS =======================================================
# --- Standardbibliothek ---
from typing import Optional, Tuple

import ai_backend as ab
import consoleOutput as co
import definitions as defin
import fileSystem as fs
import graphic_helpers as gh
import helpers as he
import image_helper as ih
import mem_guard as mg

# --- Projekt-Module (lokal) ---
from i18n import _
from loghandler import print_log

# --- Globale Cancel-Handler & ESC-Listener aktivieren (einmalig) ---
mg.install_global_cancel_handlers()
mg.enable_escape_cancel()


def which_ncnn() -> Optional[str]:
    """Pfad zu realesrgan-ncnn-vulkan (falls im PATH)."""
    return shutil.which("realesrgan-ncnn-vulkan")


def which_realcugan() -> Optional[str]:
    """Pfad zu realcugan-ncnn-vulkan (falls im PATH)."""
    return shutil.which("realcugan-ncnn-vulkan")


def pick_ncnn_gpu_flag(ncnn_bin: str) -> Optional[str]:
    """
    Erkennt das korrekte GPU-Flag des NCNN-Binaries (-g/--gpu-id oder -d/--device).
    Fallback: '-g' falls die Help nicht geparst werden kann.
    """
    try:
        cp = mg.run(
            [ncnn_bin, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        ht = (cp.stdout or "").lower()
        if re.search(r"\-\-gpu\-id|\s\-g[\s,]", ht):
            return "-g"
        if re.search(r"\-\-device|\s\-d[\s,]", ht):
            return "-d"
    except Exception:
        pass
    return "-g"


def probe_ncnn_help(ncnn_bin: str) -> str:
    """NCNN -h (mit kurzem Timeout) – Ergebnis wird geloggt (Head/Tail)."""
    try:
        cp = subprocess.run(
            [ncnn_bin, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        out = cp.stdout or ""
        head = "\n".join(out.splitlines()[:20])
        tail = "\n".join(out.splitlines()[-10:]) if len(out.splitlines()) > 20 else ""
        print_log(f"[NCNN -h] {ncnn_bin}\n--- head ---\n{head}\n--- tail ---\n{tail}")
        return out
    except Exception as e:
        print_log(f"[NCNN -h] failed: {e!r}")
        return ""


def ncnn_format_flag(ncnn_bin: str) -> str:
    """Wähle korrektes Format-Flag für NCNN (manche Builds haben --format statt -f)."""
    ht = probe_ncnn_help(ncnn_bin)
    if re.search(r"(^|\s)--format(\s|,|=|$)", ht):
        print_log("[ncnn] detected format flag '--format'")
        return "--format"
    print_log("[ncnn] using default format flag '-f'")
    return "-f"


# --- Helper: RealCUGAN models sanity ----------------------------------------


def _realcugan_models_ok(dirp: Path, scale: Optional[int] = None) -> Tuple[bool, str]:
    """
    Verifiziert, dass der RealCUGAN-Model-Ordner *brauchbar* ist:
      - existiert & ist dir
      - enthält mind. ein passendes Paar (*.param, *.bin) mit gleichem Basenamen
      - optional: enthält Files für die gewünschte scale (2/3/4)
    Liefert (ok, reason).
    """
    try:
        if not dirp or not dirp.exists() or not dirp.is_dir():
            return False, "models-dir missing or not a directory"
        params = sorted(dirp.glob("*.param"))
        bins = sorted(dirp.glob("*.bin"))
        if not params or not bins:
            return False, "missing .param or .bin files"

        # Basen abgleichen
        p_stems = {p.stem for p in params}
        b_stems = {b.stem for b in bins}
        common = p_stems & b_stems
        if not common:
            return False, "no matching .param/.bin base names"

        # Optional: Scale-spezifische Plausibilität (locker: '2x' oder 'up2x' o.ä.)
        if scale is not None:
            pat = str(scale)
            has_scale = any(
                (pat + "x") in s.lower() or (f"up{pat}x" in s.lower()) for s in common
            )
            if not has_scale:
                # Nicht hart scheitern – evtl. liegen Multi-Scale-Modelle vor.
                return (
                    True,
                    "no obvious files for requested scale, but .param/.bin pairs exist",
                )

        return True, "ok"
    except Exception as e:
        return False, f"error: {e!r}"


def _summarize_model_dir(dirp: Path, k: int = 6) -> str:
    try:
        ps = [p.name for p in sorted(dirp.glob("*.param"))[:k]]
        bs = [b.name for b in sorted(dirp.glob("*.bin"))[:k]]
        return f"params={len(list(dirp.glob('*.param')))} sample={ps} | bins={len(list(dirp.glob('*.bin')))} sample={bs}"
    except Exception as e:
        return f"<list error: {e!r}>"


def resolve_realcugan_models_dir(
    model_ui: str,
    ncnn_bin: str | Path,
    *,
    scale: Optional[int] = None,
    allow_fallback_se_for_diag: bool = True,
) -> Optional[Path]:
    """
    Strikter Resolver für Real-CUGAN models-{se|pro|nose}:
      - prüft echte Nutzbarkeit (param+bin Paare), optional scale-Hinweise,
      - durchsucht gängige Orte (ENV, defin, neben Binary, Repo, CWD),
      - liefert *None* mit logbaren Hinweisen statt späterem Segfault,
      - optional: diagnostische Prüfung auf models-se (nur Log-Hinweis).
    """
    try:
        want = (
            "models-se"
            if "realcugan-se" in (model_ui or "").lower()
            else (
                "models-pro"
                if "realcugan-pro" in (model_ui or "").lower()
                else (
                    "models-nose" if "nose" in (model_ui or "").lower() else "models-se"
                )
            )
        )

        def _ok_dir(p: Path) -> Optional[Path]:
            ok, why = _realcugan_models_ok(p, scale=scale)
            if ok:
                print_log(f"[realcugan] using {p} ({why})")
                print_log(f"[realcugan] {p} → {_summarize_model_dir(p)}")
                return p.resolve()
            else:
                print_log(f"[realcugan] reject {p}: {why}")
                return None

        def _first_ok(cands: list[Path]) -> Optional[Path]:
            for c in cands:
                hit = _ok_dir(c)
                if hit:
                    return hit
            return None

        # 1) ENV
        env_raw = (os.environ.get("REALCUGAN_MODELS_DIR", "") or "").strip()
        if env_raw:
            env_dir = Path(env_raw).expanduser()
            hit = _first_ok([env_dir, env_dir / want, env_dir / "models" / want])
            if hit:
                return hit

        # 2) defin.REALCUGAN_MODELS_DIR
        defin_dir = None
        try:
            defin_dir = getattr(defin, "REALCUGAN_MODELS_DIR", None)
        except Exception:
            pass
        if defin_dir:
            d = Path(defin_dir).expanduser()
            hit = _first_ok([d, d / want, d / "models" / want])
            if hit:
                return hit

        # 3) share neben Binary
        binp = Path(ncnn_bin).resolve()
        share_root = binp.parent.parent / "share" / "realcugan-ncnn-vulkan"
        hit = _first_ok([share_root / want, share_root / "models" / want])
        if hit:
            return hit

        # 4) direkt neben Binary
        bin_root = binp.parent
        hit = _first_ok([bin_root / want, bin_root / "models" / want])
        if hit:
            return hit

        # 5) repo-root relativ zu diesem Modul
        repo_root = Path(
            os.environ.get("VM_BASE", Path(__file__).resolve().parents[1])
        ).resolve()
        rr = repo_root / "realcugan-ncnn-vulkan"
        hit = _first_ok([rr / want, rr / "models" / want])
        if hit:
            return hit

        # 6) CWD-Variante
        cwd_rr = Path.cwd() / "realcugan-ncnn-vulkan"
        hit = _first_ok([cwd_rr / want, cwd_rr / "models" / want])
        if hit:
            return hit

        # 7) Flache Suche (sparsam)
        probe_roots = [
            p
            for p in (
                Path(env_raw) if env_raw else None,
                share_root,
                bin_root,
                rr,
                cwd_rr,
            )
            if p
        ]
        for root in probe_roots:
            try:
                if not root.exists():
                    continue
                for p in root.rglob(want):
                    hit = _ok_dir(p)
                    if hit:
                        return hit
            except Exception:
                pass

        # Optionaler Diagnose-Hinweis: gibt es wenigstens models-se?
        if allow_fallback_se_for_diag and want == "models-pro":
            se = _first_ok(
                [p / "models-se" for p in probe_roots if (p and p.exists())]
                + [bin_root / "models-se", bin_root / "models" / "models-se"]
            )
            if se:
                print_log(
                    f"[realcugan] NOTE: found usable models-se at {se}, but requested {want}. PRO likely missing."
                )

        return None
    except Exception as e:
        print_log(f"[realcugan] resolver error: {e!r}")
        return None


def _ncnn_supports_noise_flag(ncnn_bin: str | Path) -> bool:
    """
    Prüft, ob die Hilfe *irgendeinen* Noise-Flag erwähnt.
    Real-CUGAN nutzt *Kurzflag* `-n` (noise-level), nicht zwingend `--noise-level`.
    """
    try:
        help_txt = probe_ncnn_help(str(ncnn_bin))
        ht = (help_txt or "").lower()
        return (
            ("-n " in ht and "noise" in ht)
            or ("--noise" in ht)
            or ("noise-level" in ht)
        )
    except Exception:
        return False


def append_noise_arg(
    cmd: list[str], ncnn_bin: str | Path, noise_level: Optional[int]
) -> list[str]:
    """
    Hängt `-n <level>` an, wenn Noise unterstützt wird und ein Level gesetzt ist.
    Erzwingt *kein* Warnspam, wenn der Flag nicht in der Hilfe auftaucht.
    """
    if noise_level is None:
        return cmd
    try:
        if _ncnn_supports_noise_flag(ncnn_bin):
            cmd += ["-n", str(int(noise_level))]
        else:
            # still silent fallback; Real-CUGAN ohne Noise läuft weiter
            print_log(
                f"[ncnn] noise unsupported in help → skip noise (requested level={noise_level})"
            )
    except Exception:
        pass
    return cmd


def run_esrgan_per_frame_ncnn(
    ncnn_bin: str,
    model: str,
    raw_dir: Path,
    up_dir: Path,
    *,
    scale: int,
    workers: int,
    tta: bool = False,
    gpu_id: Optional[str] = None,
    chunk_idx: Optional[int] = 0,
    chunks_total: Optional[int] = None,
    ui_phase_id: Optional[int] = None,
    noise_level: Optional[int] = None,
    is_cugan: bool = False,
) -> bool:
    """NCNN per-frame (robuster RealCUGAN-Pfad mit strikter Modeldir-Validierung)."""
    from collections import deque

    retries = int(getattr(defin, "AI_PER_FRAME_RETRIES", 3) or 3)
    mg.install_global_cancel_handlers()

    input_dir = ab.ensure_png_inputs_for_tool(raw_dir, tmp_root=raw_dir.parent)
    up_dir.mkdir(parents=True, exist_ok=True)
    (up_dir / "__logs__").mkdir(parents=True, exist_ok=True)

    sess = ab.persist_context(input_dir, up_dir, tag="ncnn-per-frame")
    fs.write_json(
        sess / "env.json",
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "(auto)"),
            "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "BIN": ncnn_bin,
            "model_ui": model,
            "is_cugan": is_cugan,
            "scale": scale,
            "workers": workers,
            "tta": tta,
            "gpu_id": gpu_id,
        },
    )

    inputs = sorted(input_dir.glob("frame_*.png"), key=ih.frame_index_from_name)
    if not inputs:
        co.print_warning("[NCNN-pf] keine PNG-Inputs gefunden")
        return False

    fmt_flag = ncnn_format_flag(ncnn_bin)
    ncnn_path = Path(ncnn_bin)
    candidate_models_dir = ncnn_path.parent / "models"
    explicit_models_dir: Optional[Path] = (
        candidate_models_dir if candidate_models_dir.is_dir() else None
    )

    model_dir: Optional[Path] = None
    if is_cugan:
        model_dir = resolve_realcugan_models_dir(model, ncnn_bin, scale=scale)
        if not model_dir:
            snap = ab.persist_context(input_dir, up_dir, tag="ncnn-per-frame-fail")
            co.print_error(
                "[NCNN] RealCUGAN models directory for requested variant not usable (missing .bin/.param pairs or wrong path).\n"
                "Bitte REALCUGAN_MODELS_DIR setzen (Ordner mit models-pro/models-se/models-nose) "
                "oder defin.REALCUGAN_MODELS_DIR korrekt konfigurieren.\n"
                f"Snapshot: {snap}"
            )
            return False

    gpu_flag = pick_ncnn_gpu_flag(ncnn_bin)

    # UI-Bars
    try:
        term_cols = shutil.get_terminal_size((80, 20)).columns
    except Exception:
        term_cols = 80
    bar_len = max(20, min(80, max(20, term_cols - len(" 100% []"))))
    two_bars = isinstance(chunks_total, int) and (chunks_total or 0) > 1
    cur_chunk = (chunk_idx or 0) + 1
    backend_name = "Real-CUGAN" if is_cugan else "Real-ESRGAN"

    def _title(done: int) -> str:
        return ab.fmt_phase_title(
            chunk_idx=cur_chunk,
            chunks_total=(chunks_total or 1),
            phase=f"Upscaling (Per-Frame{' • parallel' if workers and workers > 1 else ''})",
            backend=f"NCNN: {backend_name}{' + TTA' if tta else ''}",
            model=model,
            scale=scale,
            workers=workers,
            tile=None,
            fp32=None,
            tta=tta,
            finished=done,
            total=len(inputs),
        )

    def _spawn(p: Path) -> subprocess.Popen[str]:
        out = up_dir / f"{p.stem}_out.png"
        cmd = [
            ncnn_bin,
            "-i",
            str(p),
            "-o",
            str(out),
            "-s",
            str(scale),
            fmt_flag,
            "png",
        ]
        if is_cugan:
            md = model_dir  # type: Optional[Path]
            if md is None:
                raise RuntimeError("model_dir missing for RealCUGAN")
            cmd += ["-m", str(md.resolve())]
            cmd = append_noise_arg(cmd, ncnn_bin, noise_level)
        else:
            cmd += ["-n", model]
            if explicit_models_dir is not None:
                cmd += ["-m", str(explicit_models_dir.resolve())]

        if tta:
            cmd += ["-x"]
        if gpu_id is not None and gpu_flag:
            cmd += [gpu_flag, str(gpu_id)]

        print_log(f"[NCNN] spawn: {' '.join(cmd)}")
        persisted_log = sess / "per_frame" / f"ncnn_{p.stem}.log"
        shm_log = up_dir / "__logs__" / f"ncnn_{p.stem}.log"
        ab.stub_shm_log(shm_log, persisted_log)
        persisted_log.parent.mkdir(parents=True, exist_ok=True)
        return mg.popen(
            cmd, env=ab.build_esrgan_env(), text=True, log_to=str(persisted_log)
        )

    queue = deque(inputs)
    running: list[tuple[Path, subprocess.Popen[str]]] = []
    attempts: dict[str, int] = {}
    finished = 0

    p_chunk = max(0.0, min((cur_chunk - 1) / float(chunks_total or 1), 1.0))
    top0, _t = gh.make_bar(p_chunk, bar_len)
    bot0, _t = gh.make_bar(0.0, bar_len)
    gh.draw_chunk_block_cond(
        two_bars=two_bars,
        title=_title(0),
        top_bar=top0,
        bot_bar=bot0,
        hint=_("cancel_hint"),
        ui_phase_id=ui_phase_id,
    )

    while queue and len(running) < max(1, int(workers)):
        p = queue.popleft()
        running.append((p, _spawn(p)))

    with mg.escape_cancel_guard(nonintrusive=True):
        while running or queue:
            if mg.is_cancelled():
                mg.kill_all()
                break
            new_running: list[tuple[Path, subprocess.Popen[str]]] = []
            for pth, proc in running:
                ret = proc.poll()
                if ret is None:
                    new_running.append((pth, proc))
                    continue
                out = up_dir / f"{pth.stem}_out.png"
                if out.exists():
                    finished += 1
                    frac = min(finished / max(1, len(inputs)), 1.0)
                    top, _t = gh.make_bar(
                        (
                            max(
                                0.0,
                                min(
                                    (cur_chunk - 1 + frac) / float(chunks_total or 1),
                                    1.0,
                                ),
                            )
                            if two_bars
                            else frac
                        ),
                        bar_len,
                    )
                    bot, _t = gh.make_bar(frac, bar_len)
                    gh.draw_chunk_block_cond(
                        two_bars=two_bars,
                        title=_title(finished),
                        top_bar=top,
                        bot_bar=bot,
                        hint=_("cancel_hint"),
                        ui_phase_id=ui_phase_id,
                    )
                else:
                    key = pth.stem
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] < retries:
                        print_log(
                            f"[NCNN-pf] Retry {pth.name} ({attempts[key]}/{retries}) …"
                        )
                        new_running.append((pth, _spawn(pth)))
                    else:
                        snap = ab.persist_context(
                            input_dir, up_dir, tag="ncnn-per-frame-fail"
                        )
                        log_here = (
                            snap
                            / "samples"
                            / up_dir.name
                            / "__logs__"
                            / f"ncnn_{pth.stem}.log"
                        )
                        co.print_error(
                            f"[NCNN-pf] {pth.name} nach {attempts[key]} Versuchen fehlgeschlagen → Abbruch.\n"
                            f"Snapshot: {snap}\nEinzel-Log: {log_here}"
                        )
                        mg.kill_all()
                        return False
            running = new_running
            while queue and len(running) < max(1, int(workers)):
                nxt = queue.popleft()
                running.append((nxt, _spawn(nxt)))
            time.sleep(0.02)

    produced = len(list(up_dir.glob("*_out.png"))) + len(
        list(up_dir.glob("frame_*.png"))
    )
    return produced >= len(inputs)


def run_esrgan_ncnn_dir(
    ncnn_bin: str,
    raw_dir: Path,
    up_dir: Path,
    model: str,
    scale: int,
    tta: bool = False,
    gpu_id: Optional[str] = None,
    timeout_sec: int = 0,
    noise_level: Optional[int] = None,
    is_cugan: bool = False,
) -> int:
    """NCNN Directory-Modus — RealESRGAN/RealCUGAN mit strikter Modeldir-Validierung."""
    fmt_flag = ncnn_format_flag(ncnn_bin)
    ncnn_path = Path(ncnn_bin)
    candidate_models_dir = ncnn_path.parent / "models"
    explicit_models_dir: Optional[Path] = (
        candidate_models_dir if candidate_models_dir.is_dir() else None
    )

    model_dir: Optional[Path] = None
    if is_cugan:
        model_dir = resolve_realcugan_models_dir(model, ncnn_bin, scale=scale)
        if not model_dir:
            snap = ab.persist_context(raw_dir, up_dir, tag="ncnn-fail")
            co.print_error(
                "[NCNN] RealCUGAN models directory for requested variant not usable (missing .bin/.param pairs or wrong path).\n"
                "Bitte REALCUGAN_MODELS_DIR setzen (Ordner mit models-pro/models-se/models-nose) "
                "oder defin.REALCUGAN_MODELS_DIR korrekt konfigurieren.\n"
                f"Snapshot: {snap}"
            )
            return 2

    gpu_flag = pick_ncnn_gpu_flag(ncnn_bin)

    cmd = [
        ncnn_bin,
        "-i",
        str(raw_dir),
        "-o",
        str(up_dir),
        "-s",
        str(scale),
        fmt_flag,
        "png",
    ]
    if is_cugan:
        md = model_dir  # Optional[Path]
        if md is None:
            co.print_error("[NCNN] internal: model_dir=None in directory-mode")
            return 2
        cmd += ["-m", str(md.resolve())]
        cmd = append_noise_arg(cmd, ncnn_bin, noise_level)
    else:
        cmd += ["-n", model]
        if explicit_models_dir is not None:
            cmd += ["-m", str(explicit_models_dir.resolve())]
    if tta:
        cmd += ["-x"]
    if gpu_id is not None and gpu_flag:
        cmd += [gpu_flag, str(gpu_id)]

    frames = ih.count_raw_frames(raw_dir)
    print_log(
        f"BackendUsed (plan): ncnn-vulkan ({'Real-CUGAN' if is_cugan else 'Real-ESRGAN'}); model={model}; scale={scale}; tta={he.yesno(tta)}; frames≈{frames}"
    )
    print_log(f"[NCNN] cmd={' '.join(cmd)}")

    sess = ab.persist_context(raw_dir, up_dir, tag="ncnn-dir")
    log_file = sess / "ncnn_dir.log"
    t0 = time.time()
    try:
        cp = mg.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=(timeout_sec or None),
            env=ab.build_esrgan_env(),
        )
        out = cp.stdout or ""
        ret = int(cp.returncode or 0)
        ab.log_head_tail("NCNN-dir-out", out, head_lines=80, tail_lines=40)
        try:
            log_file.write_text(out, encoding="utf-8")
        except Exception:
            pass
        dt = time.time() - t0
        if ret == 0:
            print_log(
                f"BackendUsed (effective): ncnn-vulkan; success=yes; duration={dt:.1f}s; persisted_log={log_file}"
            )
        else:
            co.print_warning(
                _("ncnn_exit_status").format(
                    code=ret, duration=f"{dt:.1f}s", log=log_file
                )
            )
        return ret
    except subprocess.TimeoutExpired:
        co.print_warning(_("ncnn_timeout"))
        return 124
    except Exception as e:
        co.print_warning(_("ncnn_exception").format(error=e))
        return 1
