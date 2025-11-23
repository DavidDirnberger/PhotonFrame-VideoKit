#!/usr/bin/env python3
# ai_tta.py
# ─────────────────────────────────────────────────────────────────────────────
# TTA (Test-Time Augmentation) – Fähigkeiten prüfen, Emulation & Pool-Workflow
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import os
import re
import shutil
import traceback

# ========== Standardbibliothek =================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, cast

# ========== Drittanbieter (optional) ===========================================
# NumPy optional: wird nur benötigt, wenn TTA-Emulation (Mittelung) aktiv ist
try:
    import numpy as _np  # runtime

    _NP_OK = True
except Exception:
    _NP_OK = False
    _np = None  # type: ignore[assignment]

# ========== Projekt-Imports (lokal) ===========================================
import ai_backend as ab
import ai_ncnn_backend as an
import consoleOutput as co
import graphic_helpers as gh
import image_helper as ih
import mem_guard as mg
from i18n import _
from loghandler import print_log
from pil_image import PIL_OK, Image  # zentrales Pillow-Shim

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 ÖFFENTLICHE HILFSFUNKTIONEN / CAPABILITIES               ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def adjust_workers_for_emulated_tta(
    base_workers: int,
    vram_mb: int,
    *,
    honor_min: bool = False,
    user_profile: Optional[str] = None,
) -> int:
    """
    Reduziert Worker konservativ, wenn TTA×8 emuliert wird.
    Faustregeln abhängig vom VRAM; „minimal“-Profile respektieren ggf. ein Minimum.

    Args:
        base_workers: Ausgangszahl parallel laufender Worker.
        vram_mb: Verfügbarer VRAM in MiB.
        honor_min: Ob bei Minimal-Profilen ein Mindestwert erzwungen wird.
        user_profile: Name des Worker-Profils (z. B. "minimal").

    Returns:
        Angepasste Workerzahl ≥ 1.
    """
    base = max(1, int(base_workers))
    prof = ab.normalize_worker_profile_name(user_profile)

    if base <= 1:
        return 1
    if vram_mb >= 24_576:  # 24 GB+
        w = max(1, base - 1)
    elif vram_mb >= 16_384:  # 16 GB
        w = max(1, math.ceil(base * 0.8))
    else:  # <= 12 GB
        w = max(1, math.ceil(base * 0.6))

    if honor_min and prof in ("minimal",):
        w = max(2, w)
    return int(w)


def can_emulate_tta_x8() -> bool:
    """
    Prüft, ob die Emulation von TTA×8 grundsätzlich möglich ist.
    Benötigt Pillow (über pil_image-Shim) + NumPy.
    """
    return bool(PIL_OK and _NP_OK)


def tta_capabilities(
    venv_python, esr_script, ncnn_bin_or_none
) -> tuple[bool, bool, bool]:
    """
    Liefert (py_has_tta, ncnn_has_tta, emu_tta).

    - py_has_tta: Python-Inference bietet '--tta' / '--tta-mode'.
    - ncnn_has_tta: NCNN-Binary unterstützt TTA (-x / tta).
    - emu_tta: TTA per x8 Self-Ensemble emulierbar (Pillow + NumPy vorhanden).
    """
    try:
        py_has = python_supports_tta(venv_python, esr_script)
    except Exception:
        py_has = False
    try:
        ncnn_has = ncnn_supports_tta(ncnn_bin_or_none) if ncnn_bin_or_none else False
    except Exception:
        ncnn_has = False
    emu = can_emulate_tta_x8()
    return (bool(py_has), bool(ncnn_has), bool(emu))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   CAPABILITY-PROBES (Python / NCNN)                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def python_supports_tta(venv_python: Path, esr_script: Path) -> bool:
    """
    Prüft, ob die Python-Inference '--tta' (oder '--tta-mode') anbietet.
    WICHTIG: Das NCNN-Flag '-x' gehört nicht hierher.
    """
    ht = ab.probe_realesrgan_help(venv_python, esr_script)
    if not ht:
        return False
    return bool(re.search(r"(?m)(?:^|\s)(--tta(?:\b|=)|--tta-mode(?:\b|=))", ht))


def ncnn_supports_tta(ncnn_bin: str) -> bool:
    """
    Erkennt TTA-Unterstützung in NCNN-Binaries.
    Häufige Varianten:
      - '-x' (Real-ESRGAN/RealCUGAN)
      - 'tta' irgendwo im Hilfetext
    """
    ht = an.probe_ncnn_help(ncnn_bin)
    if not ht:
        return False
    if re.search(r"(?m)^\s*-x\b", ht):
        return True
    if re.search(r"\btta\b", ht, re.I):
        return True
    return False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        INTERNE TTA-TRANSFORM-HILFEN                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _tta_flag_list() -> list[tuple[int, int, int]]:
    """
    Reihenfolge identisch zu _tta_ops():
    t in {0,1}, hf in {0,1}, vf in {0,1}
    """
    flags: list[tuple[int, int, int]] = []
    for t in (0, 1):
        for hf in (0, 1):
            for vf in (0, 1):
                flags.append((t, hf, vf))
    return flags


def _tta_transform_file(src: str, dst: str, t: int, hf: int, vf: int) -> bool:
    """
    Top-Level Worker: PNG laden → Transforms (Transpose/HFlip/VFlip) → PNG speichern.
    """
    try:
        im = ih.load_png(Path(src))
    except Exception:
        im = Image.open(src).convert("RGB")

    if t:
        im = im.transpose(cast(Any, Image.Transpose.TRANSPOSE))
    if hf:
        im = im.transpose(cast(Any, Image.Transpose.FLIP_LEFT_RIGHT))
    if vf:
        im = im.transpose(cast(Any, Image.Transpose.FLIP_TOP_BOTTOM))
    ih.png_save_fast(im, Path(dst))
    return True


def _tta_ops() -> List[tuple[List[Callable[[Any], Any]], List[Callable[[Any], Any]]]]:
    """
    Liefert 8 (ops, inverse_ops)-Paare.
    Jede Einzel-Operation ist involutiv; Inverse ist ops[::-1] (gleiche Ops rückwärts).
    """

    def transpose_hw(im: Any) -> Any:
        return im.transpose(Image.Transpose.TRANSPOSE)

    def flip_lr(im: Any) -> Any:
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def flip_tb(im: Any) -> Any:
        return im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    combos: List[tuple[List[Callable[[Any], Any]], List[Callable[[Any], Any]]]] = []
    for t in (0, 1):
        for hf in (0, 1):
            for vf in (0, 1):
                ops: List[Callable[[Any], Any]] = []
                if t:
                    ops.append(transpose_hw)
                if hf:
                    ops.append(flip_lr)
                if vf:
                    ops.append(flip_tb)
                inv = list(reversed(ops))  # Inverse
                combos.append((ops, inv))
    return combos  # 8 Stück


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 DATENSTRUKTUREN FÜR POOL-VARIANTEN (x8)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class _TTAVariant:
    idx: int
    ops: List[Callable[[Any], Any]]
    inv_ops: List[Callable[[Any], Any]]
    in_dir: Path
    out_dir: Path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   VARIANTEN-SETUP (Verzeichnis-Shards)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def prepare_tta_variant_dirs(
    *, input_dir: Path, tmp_root: Path, threads: Optional[int] = None
) -> List[_TTAVariant]:
    """
    Erzeugt bis zu 8 Varianteneingänge (je eine Transform-Kombination) unterhalb
    eines temporären Root-Verzeichnisses. Identität wird per Hardlink/Copy abgebildet,
    alle anderen Varianten per paralleler Transformation.

    Returns:
        Liste tatsächlich erzeugter Varianten (leere werden übersprungen).
    """
    if not ab.pil_or_warn() or mg.is_cancelled():
        return []

    # Basis-Tmpdir: respektiert AI_TMPDIR falls gesetzt
    base_root = Path(os.environ.get("AI_TMPDIR", str(tmp_root))).resolve()
    shutil.rmtree(base_root, ignore_errors=True)
    base_root.mkdir(parents=True, exist_ok=True)

    frames = sorted(input_dir.glob("frame_*.png"), key=ih.frame_index_from_name)
    if not frames:
        return []

    pairs = _tta_ops()
    flags = _tta_flag_list()

    variants: List[_TTAVariant] = []
    cpu_w = max(1, (threads or os.cpu_count() or 4))

    for i, ((ops, inv_ops), (t, hf, vf)) in enumerate(zip(pairs, flags)):
        in_dir = base_root / f"tta_{i:02d}" / "in"
        out_dir = base_root / f"tta_{i:02d}" / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        if t == 0 and hf == 0 and vf == 0:
            # Identität → möglichst schnell (Hardlinks; Fallback Copy)
            ok = 0
            for p in frames:
                dst = in_dir / p.name
                try:
                    os.link(p, dst)
                    ok += 1
                except Exception:
                    shutil.copy2(p, dst)
                    ok += 1
            print_log(f"[TTA-PREP] var {i}/8 (I) → {ok}/{len(frames)}")
        else:
            # Parallel-Transform: bevorzugt ProcessPool; Fallback ThreadPool/seriell
            def _run_process_pool() -> int:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                done = 0
                with ProcessPoolExecutor(max_workers=cpu_w) as ex:
                    futs = [
                        ex.submit(
                            _tta_transform_file, str(p), str(in_dir / p.name), t, hf, vf
                        )
                        for p in frames
                    ]
                    for fu in as_completed(futs):
                        try:
                            if fu.result():
                                done += 1
                        except Exception:
                            pass
                return done

            def _run_thread_pool() -> int:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                done = 0
                with ThreadPoolExecutor(max_workers=cpu_w) as ex:
                    futs = [
                        ex.submit(
                            _tta_transform_file, str(p), str(in_dir / p.name), t, hf, vf
                        )
                        for p in frames
                    ]
                    for fu in as_completed(futs):
                        try:
                            if fu.result():
                                done += 1
                        except Exception:
                            pass
                return done

            created = 0
            try:
                created = _run_process_pool()
            except Exception as e:
                print_log(
                    f"[TTA-PREP] proc-pool fail (flags t={t},hf={hf},vf={vf}) → {e!r}; falling back to thread-pool"
                )
                try:
                    created = _run_thread_pool()
                except Exception as e2:
                    print_log(
                        f"[TTA-PREP] thread-pool fail → {e2!r}; falling back to serial"
                    )
                    # Seriell – langsam aber robust
                    created = 0
                    for p in frames:
                        try:
                            _tta_transform_file(str(p), str(in_dir / p.name), t, hf, vf)
                            created += 1
                        except Exception:
                            pass
            print_log(
                f"[TTA-PREP] var {i}/8 (t={t},hf={hf},vf={vf}) → {created}/{len(frames)}"
            )

        # Sanity: Variante muss Frames haben – sonst später „no inputs“
        have = len(list(in_dir.glob("frame_*.png")))
        if have == 0:
            ab.warn_once(
                f"tta.var.{i}.empty",
                f"[TTA-PREP] Variante {i} hat keine Inputs erzeugt – wird übersprungen.",
            )
            continue

        variants.append(
            _TTAVariant(idx=i, ops=ops, inv_ops=inv_ops, in_dir=in_dir, out_dir=out_dir)
        )

    if not variants:
        ab.warn_once(
            "tta.prep.none", "[TTA-POOL] Vorbereitung hat keine Varianten erzeugt."
        )
    else:
        print_log(
            f"[TTA-PREP] prepared variants={len(variants)} (of 8) root={base_root}"
        )
    return variants


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         FUSION / MITTELUNG (x8)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _fuse_tta_outputs_to_file(
    var_outputs: List[Tuple[Path, List[Callable[[Any], Any]]]], out_path: Path
) -> bool:
    """
    Mittelt die vorhandenen Variante-Outputs (nach Rücktransformation) zu einem
    Ergebnisbild und speichert es als PNG.
    """
    if not _NP_OK or _np is None:
        ab.warn_once("tta.numpy", "NumPy nicht verfügbar – TTA-Fusion übersprungen.")
        return False
    np = cast(Any, _np)

    try:
        meta_lines: List[str] = []
        imgs: List[Any] = []
        min_w: Optional[int] = None
        min_h: Optional[int] = None

        for idx, (vout, inv_ops) in enumerate(var_outputs):
            im = ih.load_png(vout)
            meta_lines.append(
                f"  [{idx}] file={vout} size={im.width}x{im.height} mode={getattr(im, 'mode', '?')}"
            )
            im = ab.apply_ops(im, inv_ops)
            if getattr(im, "mode", "RGB") != "RGB":
                im = im.convert("RGB")
            arr = ih.np_from_img(im).astype("float32", copy=False)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            min_w = w if min_w is None else min(min_w, w)
            min_h = h if min_h is None else min(min_h, h)
            imgs.append(arr)

        print_log("[TTA-EMUL] outputs before fuse:\n" + "\n".join(meta_lines))
        if not imgs or min_w is None or min_h is None:
            co.print_warning("[TTA-EMUL] FUSE: keine gültigen Teilbilder – Abbruch.")
            return False

        mw: int = int(min_w)
        mh: int = int(min_h)
        if any(a.shape[0] != mh or a.shape[1] != mw for a in imgs):
            print_log(f"[TTA-EMUL] NOTICE: shape mismatch → cropping all to {mw}x{mh}")
        imgs = [a[:mh, :mw, :3] for a in imgs]

        # float32-Mittelung (vektorisiert)
        acc = np.zeros((mh, mw, 3), dtype="float32")
        for a in imgs:
            acc += a
        mean = np.clip(acc / float(len(imgs)), 0.0, 255.0).astype("uint8")

        ih.png_save_fast(mean, out_path)
        print_log(f"[TTA-EMUL] wrote fused: {out_path} size={mw}x{mh}")
        return out_path.exists() and out_path.stat().st_size > 0

    except Exception as e:
        print_log(f"[TTA-EMUL] FUSE ERROR: {e!r}")
        print_log(traceback.format_exc())
        return False


def fuse_tta_pool_variants_to_updir(
    *,
    variants: List[_TTAVariant],
    up_dir: Path,
    input_dir: Path,
    esrgan_root: Path,
    ui_title: str | None = None,
    two_bars: bool = False,
    cur_chunk: int = 1,
    chunks_total: int = 1,
    ui_phase_id: Optional[int] = None,
) -> int:
    """
    Führt pro Frame die vorhandenen Variantenausgaben in 'up_dir' zusammen.
    Fehlende Varianten werden protokolliert und übersprungen (Mittelung der
    vorhandenen).
    """
    frames = sorted(input_dir.glob("frame_*.png"), key=ih.frame_index_from_name)
    total = len(frames)
    if total == 0:
        return 0

    # Progressbar-Größen
    try:
        term_cols = shutil.get_terminal_size((80, 20)).columns
    except Exception:
        term_cols = 80
    static_len = len(" 100% []")
    bar_len = max(20, min(80, max(20, term_cols - static_len)))

    # Threading-Grad: über AI_FUSE_PROCS steuerbar
    try:
        procs = int(os.environ.get("AI_FUSE_PROCS", "0") or "0")
    except Exception:
        procs = 0
    if procs <= 0:
        procs = max(1, min(8, os.cpu_count() or 4))

    out_dirs = [str(v.out_dir) for v in variants]
    inv_ops_list = [v.inv_ops for v in variants]

    finished = 0
    ok_count = 0

    # Worker: nimmt nur vorhandene Outputs einer Frame-Variante
    def _fuse_one(stem: str) -> bool:
        var_outputs: List[Tuple[Path, List[Callable[[Any], Any]]]] = []
        miss = 0
        for out_dir, inv_ops in zip(out_dirs, inv_ops_list):
            cand = Path(out_dir) / f"{stem}_out.png"
            if cand.exists() and cand.stat().st_size > 0:
                var_outputs.append((cand, inv_ops))
            else:
                miss += 1
        if not var_outputs:
            print_log(f"[TTA-POOL] FUSE skip (no candidates) for {stem}")
            return False
        if miss:
            ab.warn_once(
                f"tta.fuse.missing.{stem}",
                f"[TTA-POOL] FUSE: {miss} fehlende Varianten für {stem} – mitteln vorhandene.",
            )
        return _fuse_tta_outputs_to_file(var_outputs, up_dir / f"{stem}_out.png")

    if procs == 1:
        # Seriell (mit UI-Progress)
        for p in frames:
            if mg.is_cancelled():
                break
            if _fuse_one(p.stem):
                ok_count += 1
            finished += 1
            if ui_title:
                p_frames = min(finished / max(1, total), 1.0)
                p_chunk_draw = (
                    max(
                        0.0,
                        min((cur_chunk - 1 + p_frames) / float(chunks_total or 1), 1.0),
                    )
                    if two_bars
                    else p_frames
                )
                top, _t = gh.make_bar(p_chunk_draw if two_bars else p_frames, bar_len)
                bot, _t = gh.make_bar(p_frames, bar_len)
                gh.draw_chunk_block_cond(
                    two_bars=two_bars,
                    title=f"{ui_title} • Fuse {finished}/{total}",
                    top_bar=top,
                    bot_bar=bot,
                    hint=_("cancel_hint"),
                    ui_phase_id=ui_phase_id,
                )
        return ok_count

    # Parallel (Threads) mit UI-Progress
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=procs) as ex:
        futs = {ex.submit(_fuse_one, p.stem): p.stem for p in frames}
        for fu in as_completed(futs):
            if mg.is_cancelled():
                break
            try:
                if fu.result():
                    ok_count += 1
            except Exception as e:
                print_log(f"[TTA-POOL] fuse error (thread): {e!r}")
            finished += 1
            if ui_title:
                p_frames = min(finished / max(1, total), 1.0)
                p_chunk_draw = (
                    max(
                        0.0,
                        min((cur_chunk - 1 + p_frames) / float(chunks_total or 1), 1.0),
                    )
                    if two_bars
                    else p_frames
                )
                top, _t = gh.make_bar(p_chunk_draw if two_bars else p_frames, bar_len)
                bot, _t = gh.make_bar(p_frames, bar_len)
                gh.draw_chunk_block_cond(
                    two_bars=two_bars,
                    title=f"{ui_title} • Fuse {finished}/{total}",
                    top_bar=top,
                    bot_bar=bot,
                    hint=_("cancel_hint"),
                    ui_phase_id=ui_phase_id,
                )

    return ok_count


def _tta_pool_make_args_builder(
    *,
    help_text: str,
    model: str,
    face_enhance: bool,
    denoise: Optional[float],
    scale: float,
    force_fp32: bool,
    tile_size: int,
    tile_pad: int,
    cuda_mask: Optional[str],
):
    """Closure für pool_workers.run_sharded_dir_job: baut CLI-Args pro Shard/Variante."""

    def _make_args(in_dir: Path, out_dir: Path) -> List[str]:
        common = ab.build_realesrgan_cli_common(
            help_text,
            model,
            out_dir=out_dir,
            face_enhance=face_enhance,
            denoise=denoise,
            scale=scale,
            include_ext_and_suffix=True,
            force_fp32=force_fp32,
            tile_size=int(tile_size or 0),
            tile_pad=int(tile_pad or 20),
            tta=False,  # echtes --tta hier AUS; wir emulieren selbst
        )
        in_flag = ab.pick_flag(help_text, ["-i", "--input"], "-i")
        args = ([in_flag, str(in_dir)] if in_flag else []) + common
        gpu_flag = ab.pick_flag(help_text, ["-g", "--gpu-id"], None)
        if gpu_flag and (cuda_mask is None or str(cuda_mask).strip().lower() != "cpu"):
            args += [gpu_flag, "0"]
        print_log(
            f"[PT-pf][TTA-builder] in='{in_dir.name}' out='{out_dir.name}' "
            f"fp32={force_fp32} tile={tile_size} gpuflag={'yes' if gpu_flag else 'no'} → args={' '.join(map(str, args))}",
            "_pytorch",
        )
        return args

    return _make_args
