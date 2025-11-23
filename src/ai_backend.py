#!/usr/bin/env python3
# ai_backend.py

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time

# ========== 1) IMPORTS =======================================================
# --- Standardbibliothek ---
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Drittanbieter (optional) ---
try:
    import numpy as _np  # runtime

    _NP_OK = True
except Exception:
    _NP_OK = False
    _np = None  # type: ignore[assignment]

# --- Projekt-Module (lokal) ---
import consoleOutput as co
import definitions as defin
import helpers as he
import image_helper as ih
import mem_guard as mg
import VideoEncodersCodecs as vec
from loghandler import print_log
from pil_image import PIL_OK  # zentrales Pillow-Shim

# ========== 2) STATE & DATENSTRUKTUREN ======================================
# Warn/Info-once Registry (ent-spammt Meldungen)
_WARNED_ONCE: set[str] = set()


@dataclass
class SystemCaps:
    """System-/Backend-Fähigkeiten für Laufzeitentscheidungen."""

    os_name: str
    torch_device: Optional[str]  # "cuda" | "mps" | "cpu" | None
    torch_cuda_ok: bool
    torch_mps_ok: bool
    ncnn_ok: bool
    ncnn_bins: List[str]
    realcugan_ok: bool
    python_tta_ok: bool
    ncnn_tta_ok: bool


# ========== 3) ALLGEMEINE HELPERS ===========================================


def warn_once(key: str, msg: str) -> None:
    """Gibt eine Warnung pro Schlüssel maximal einmal aus."""
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    try:
        co.print_warning(msg)
    except Exception:
        print(f"WARNING: {msg}")


def pil_or_warn() -> bool:
    """Guard für Pillow+NumPy – notwendig für TTA (Py)."""
    if PIL_OK and _NP_OK:
        return True
    warn_once(
        "py.tta.pil", "Pillow/NumPy nicht verfügbar – TTA (Py) wird übersprungen."
    )
    return False


def apply_ops(im: Any, ops: List[Callable[[Any], Any]]) -> Any:
    """Wendet eine Folge von Bildoperationen in Reihenfolge an."""
    for op in ops:
        im = op(im)
    return im


def debug_enabled() -> bool:
    """Globale Debug-Schalter (AI_DEBUG)."""
    return str(os.environ.get("AI_DEBUG", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def capture_logs_enabled() -> bool:
    """Erzwingt Log-Capture in Debug, sonst via AI_CAPTURE_LOGS."""
    if debug_enabled():
        return True
    return str(os.environ.get("AI_CAPTURE_LOGS", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_int(name: str, default: int) -> int:
    """Robustes Lesen einer int-Umgebungsvariable."""
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def tile_label(t: Optional[int]) -> str:
    """String-Repräsentation für Tile-Größe (0/None → 'none')."""
    try:
        if t is None or int(t) <= 0:
            return "none"
        return str(int(t))
    except Exception:
        return "none"


def env_cuda_mask() -> str:
    """Liest CUDA_VISIBLE_DEVICES; zeigt '(auto)', wenn unset oder nicht vorhanden."""
    try:
        v = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if v is None:
            return "(auto)"
        # Leerer String bedeutet „CPU erzwingen“ – klar anzeigen:
        return v if v != "" else ""
    except Exception:
        return "(auto)"


def _force_torch_device_from_env() -> Optional[str]:
    """Liest AI_FORCE_TORCH_DEVICE {cpu|cuda|mps|auto} und normalisiert."""
    try:
        v = str(os.environ.get("AI_FORCE_TORCH_DEVICE", "")).strip().lower()
    except Exception:
        return None
    if not v or v == "auto":
        return None
    return v if v in {"cpu", "cuda", "mps"} else None


def _logs_root() -> Path:
    try:
        base = Path(getattr(defin, "LOG_DIR", "logs"))
    except Exception:
        base = Path("logs")
    root = base / "ncnn_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def report_outputs_and_missing(
    *,
    up_dir: Path,
    input_dir: Path,
    esrgan_root: Path,
    tag: str = "outputs-report",
    max_list: int = 25,
) -> None:
    """Loggt eine ausführliche Bestandsaufnahme der Outputs und listet fehlende Stems (falls noch nicht normalisiert)."""
    try:
        import json as _json

        # 0) Basiszustand (exist/dir/perm + PNG-Zähler)
        log_paths_state(
            title=f"report:{tag}",
            raw_dir=input_dir,
            input_dir=input_dir,
            up_dir=up_dir,
            esrgan_root=esrgan_root,
        )

        # 1) Counts in up_dir / results / pool_tmp
        pool_tmp = up_dir / "__pool_tmp__"
        results = esrgan_root / "results"
        n_out_up = len(list(up_dir.glob("*_out.png")))
        n_seq_up = len(list(up_dir.glob("frame_*.png")))
        n_out_res = len(list(results.rglob("*_out.png"))) if results.exists() else 0
        n_seq_res = len(list(results.rglob("frame_*.png"))) if results.exists() else 0
        n_in_pool = (
            len(list(pool_tmp.rglob("in/frame_*.png"))) if pool_tmp.exists() else 0
        )
        n_out_pool = len(list(pool_tmp.rglob("out/*.png"))) if pool_tmp.exists() else 0

        # 2) Input-Stems
        src_inputs = sorted(input_dir.glob("frame_*.png"), key=ih.frame_index_from_name)
        stems_in = [p.stem.replace("_out", "") for p in src_inputs]
        total = len(stems_in)

        # 3) Gefundene *_out-Stems überall (up_dir, results, pool_tmp/out)
        found_out: set[str] = set()
        for p in up_dir.glob("*_out.png"):
            found_out.add(p.stem.replace("_out", ""))
        if results.exists():
            for p in results.rglob("*_out.png"):
                found_out.add(p.stem.replace("_out", ""))
        if pool_tmp.exists():
            for p in pool_tmp.rglob("out/*_out.png"):
                found_out.add(Path(p.name).stem.replace("_out", ""))

        # 4) Missing nur dann sinnvoll, wenn (noch) *_out existieren sollen
        missing: list[str] = []
        if n_seq_up == 0:  # noch nicht normalisiert → *_out erwartet
            missing = [s for s in stems_in if s not in found_out]
        miss_show = missing[:max_list]

        payload = {
            "tag": tag,
            "up_dir": str(up_dir),
            "counts": {
                "inputs_png": total,
                "up_dir": {"*_out": n_out_up, "frame_*": n_seq_up},
                "results": {"*_out": n_out_res, "frame_*": n_seq_res},
                "pool_tmp": {"in/frame_*": n_in_pool, "out/*.png": n_out_pool},
            },
            "missing": {"n": len(missing), "sample": miss_show},
            "samples": {
                "up_out": [p.name for p in list(up_dir.glob("*_out.png"))[:5]],
                "up_frame": [p.name for p in list(up_dir.glob("frame_*.png"))[:5]],
            },
        }
        print_log("[REPORT] " + _json.dumps(payload, indent=2), "_pytorch")
        if missing:
            print_log(
                f"[REPORT] missing first {len(miss_show)}/{len(missing)}: {miss_show}",
                "_pytorch",
            )
        else:
            print_log(
                "[REPORT] no missing stems detected (either all present or already normalized).",
                "_pytorch",
            )
    except Exception as e:
        print_log(f"[REPORT] error: {e!r}", "_pytorch")


def scan_pool_tmp_inventory(
    up_dir: Path, *, tag: str = "pool-scan", max_list: int = 12
) -> None:
    """Listet pro Shard die In/Out-Bestände und einige Beispieldateien."""
    try:
        pool_tmp = up_dir / "__pool_tmp__"
        if not pool_tmp.exists():
            print_log(f"[POOL-SCAN] {tag}: no __pool_tmp__ at {pool_tmp}", "_pytorch")
            return
        shards = sorted([d for d in pool_tmp.glob("shard_*") if d.is_dir()])
        print_log(
            f"[POOL-SCAN] {tag}: shards={len(shards)} root={pool_tmp}", "_pytorch"
        )
        for sd in shards:
            try:
                si = sd / "in"
                so = sd / "out"
                nin = len(list(si.glob("frame_*.png"))) if si.exists() else 0
                nout = len(list(so.glob("*.png"))) if so.exists() else 0
                smpi = [p.name for p in list(si.glob("frame_*.png"))[:max_list]]
                smpo = [p.name for p in list(so.glob("*.png"))[:max_list]]
                print_log(
                    f"[POOL-SCAN] {sd.name} nin={nin} nout={nout} smp_in={smpi} smp_out={smpo}",
                    "_pytorch",
                )
            except Exception as e:
                print_log(f"[POOL-SCAN] {sd.name} err: {e!r}", "_pytorch")
    except Exception as e:
        print_log(f"[POOL-SCAN] fatal: {e!r}", "_pytorch")


def _tail_file_lines(p: Path, max_lines: int = 160) -> str:
    """Liest robust die letzten Zeilen einer Textdatei (für Shard-Logs)."""
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            dq = deque(f, maxlen=max_lines)
        return "".join(dq)
    except Exception as e:
        return f"<tail error: {e!r}>"


def _sample_list(dirp: Path, pattern: str, k: int = 5) -> list[str]:
    try:
        xs = sorted(dirp.glob(pattern))
        return [x.name for x in xs[:k]]
    except Exception:
        return []


def debug_dump_pool_failure(
    *,
    up_dir: Path,
    input_dir: Path,
    esrgan_root: Path,
    tag: str = "torch-pool-postmortem",
    also_cat_logs: bool = True,
) -> None:
    """
    Postmortem-Dump, wenn der POOL fehlgeschlagen ist:
      - Verzeichnisbäume & Zählwerte
      - pro Shard: IN/OUT-Counts, ein paar Beispieldateien
      - Tail der Shard-Logfiles
    """
    try:
        pool_tmp = up_dir / "__pool_tmp__"
        print_log(f"[PT-pf] POSTMORTEM tag={tag} pool_tmp={pool_tmp}", "_pytorch")
        log_paths_state(
            title=f"postmortem:{tag}",
            raw_dir=input_dir,
            input_dir=input_dir,
            up_dir=up_dir,
            esrgan_root=esrgan_root,
        )
        if not pool_tmp.exists():
            print_log(
                "[PT-pf] POSTMORTEM: __pool_tmp__ fehlt – keine Shard-Artefakte vorhanden",
                "_pytorch",
            )
            return

        # Kurzer Tree-Dump (gedeckelt):
        try:
            tree_txt = _list_dir_tree(pool_tmp, max_entries=300)
            print_log(
                f"[PT-pf] POSTMORTEM tree of __pool_tmp__\n{tree_txt}", "_pytorch"
            )
        except Exception as e:
            print_log(f"[PT-pf] POSTMORTEM tree error: {e!r}", "_pytorch")

        # Pro Shard Details
        shards = sorted([d for d in pool_tmp.glob("shard_*") if d.is_dir()])
        for sd in shards:
            in_dir = sd / "in"
            out_dir = sd / "out"
            n_in = len(list(in_dir.glob("*.png"))) if in_dir.is_dir() else 0
            n_out = len(list(out_dir.glob("*.png"))) if out_dir.is_dir() else 0
            smpl_in = _sample_list(in_dir, "*.png", 5)
            smpl_out = _sample_list(out_dir, "*.png", 5)
            log_file = next((p for p in out_dir.glob("*.log")), None)
            print_log(
                f"[PT-pf] SHARD {sd.name}: in={n_in} out={n_out} "
                f"samples_in={smpl_in} samples_out={smpl_out} log={log_file}",
                "_pytorch",
            )

            if also_cat_logs and log_file and log_file.exists():
                tail = _tail_file_lines(log_file, max_lines=200)
                # Kopf/Ende hübsch splitten:
                log_head_tail(f"{sd.name}.log", tail, head_lines=120, tail_lines=0)

        # Extra: Ergebnisse in realesrgan/results
        res_dir = esrgan_root / "results"
        if res_dir.exists():
            n_res = len(list(res_dir.rglob("*_out.png")))
            print_log(f"[PT-pf] POSTMORTEM results/ detected files={n_res}", "_pytorch")
    except Exception as e:
        print_log(f"[PT-pf] POSTMORTEM failed: {e!r}", "_pytorch")


def _new_session_dir(tag: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    sid = f"{ts}_{tag}_pid{os.getpid()}"
    out = _logs_root() / sid
    out.mkdir(parents=True, exist_ok=True)
    print_log(f"[persist] session_dir={out}")
    return out


def _list_dir_tree(root: Path, max_entries: int = 200) -> str:
    lines = []
    try:
        i = 0
        for base, dirs, files in os.walk(root):
            rel = os.path.relpath(base, root)
            lines.append(f"[{rel}]")
            for d in sorted(dirs):
                lines.append(f"  <dir> {d}")
                i += 1  # count dirs too
                if i >= max_entries:
                    break
            for f in sorted(files):
                fp = Path(base) / f
                try:
                    sz = fp.stat().st_size
                    mt = time.strftime("%H:%M:%S", time.localtime(fp.stat().st_mtime))
                    lines.append(f"  {f}  ({sz} B, mtime {mt})")
                except Exception:
                    lines.append(f"  {f}")
                i += 1
                if i >= max_entries:
                    break
            if i >= max_entries:
                break
    except Exception as e:
        lines.append(f"[tree] error: {e!r}")
    return "\n".join(lines)


def build_env_for_gpu(gpu_spec: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    # NEU: Leerstring wie None behandeln → GPUs sichtbar lassen
    if gpu_spec is None or str(gpu_spec).strip() == "":
        if env.get("CUDA_VISIBLE_DEVICES", None) == "":
            env.pop("CUDA_VISIBLE_DEVICES", None)  # CPU-Erzwingung entfernen
            print_log(
                "[env] removed inherited CUDA_VISIBLE_DEVICES='' → GPUs visible again"
            )
        return env
    m = str(gpu_spec).strip().lower()
    if m == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # bewusst CPU erzwingen
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_spec).strip()
    return env


def parse_gpu_id_for_ncnn(cuda_mask: Optional[str]) -> Optional[str]:
    """
    Mappt CUDA_VISIBLE_DEVICES Maske auf NCNN -g:
      - "cpu" -> -1
      - "0" | "1" -> selbe Zahl
      - "0,1" -> "0,1" (Multi-GPU)
      - None -> None (auto)
    """
    if cuda_mask is None:
        return None
    m = str(cuda_mask).strip().lower()
    if m == "" or m == "cpu":
        return "-1"  # CPU
    return m  # "0", "0,1", ...


def get_model_meta() -> Dict[str, Dict[str, Any]]:
    """Klon-Map von defin.MODEL_META (schreibsicher)."""
    try:
        return dict(getattr(defin, "MODEL_META", {}))
    except Exception:
        return {}


def desc_from_meta(meta: Dict[str, Any]) -> str:
    """Liefert lokalisierte Modellbeschreibung aus Meta-Map (Fallbacks de/en/erstes Element)."""
    d = meta.get("desc", "")
    if isinstance(d, dict):
        lang = getattr(defin, "LANG", None)
        if isinstance(lang, str) and lang in d:
            return str(d[lang])
        for k in ("de", "en"):
            if k in d:
                return str(d[k])
        try:
            return str(next(iter(d.values())))
        except Exception:
            return ""
    return str(d or "")


def build_esrgan_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Torch/FFI-freundliches Env, inkl. Split-Allocator & TF32 Defaults.
    (Belasst CUDA_VISIBLE_DEVICES so, wie es 'base' vorgibt.)
    """
    env = dict(os.environ if base is None else base)
    conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if not conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:256,garbage_collection_threshold:0.7,expandable_segments:True"
        )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("CUDA_MODULE_LOADING", os.environ.get("CUDA_MODULE_LOADING", "LAZY"))
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TORCH_ALLOW_TF32", "1")
    env.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "64")
    return env


def fmt_phase_title(
    *,
    chunk_idx: int,
    chunks_total: int,
    phase: str,
    backend: str,
    model: str,
    scale: Optional[int | float] = None,
    workers: Optional[int] = None,
    tile: Optional[int] = None,
    fp32: Optional[bool] = None,
    tta: Optional[bool] = None,
    finished: Optional[int] = None,
    total: Optional[int] = None,
) -> str:
    """Kurze UI-Überschrift für Fortschrittsblöcke."""
    bits: List[str] = [f"Chunk {chunk_idx}/{chunks_total} – {phase} [{backend}]"]
    if finished is not None and total is not None and total > 0:
        bits.append(f"{finished}/{total}")
    return " • ".join(bits)


def probe_torch_device_in_venv(
    venv_python: Optional[Path],
) -> Tuple[Optional[str], bool, bool]:
    """
    Liefert (device, cuda_ok, mps_ok) robust via Subprozess:
      device ∈ {"cuda","mps","cpu", None}; None → Torch nicht importierbar.
    Respektiert AI_FORCE_TORCH_DEVICE.
    """
    override = _force_torch_device_from_env()
    if override in {"cpu", "cuda", "mps"}:
        print_log(f"[torch] env override AI_FORCE_TORCH_DEVICE={override}")
        return override, (override == "cuda"), (override == "mps")

    try:
        py = venv_python if venv_python else Path(shutil.which("python3") or "python3")
    except Exception:
        py = Path(shutil.which("python3") or "python3")

    code = r"""
import json, importlib
info = {'tor_ok': False, 'dev': None, 'cuda': False, 'mps': False, 'torch': None, 'cuda_ver': None, 'mps_built': None, 'detail': {}}
try:
    torch = importlib.import_module('torch'); info['tor_ok']=True; info['torch']=getattr(torch,'__version__',None)
    try:
        info['cuda_ver']=getattr(getattr(torch,'version',object()),'cuda',None)
        info['cuda']=bool(getattr(getattr(torch,'cuda',object()),'is_available',lambda:False)())
    except Exception as e:
        info['detail']['cuda_err']=str(e)
    try:
        from torch.backends import mps
        mb=bool(getattr(mps,'is_built',lambda:False)()); ma=bool(getattr(mps,'is_available',lambda:False)())
        info['mps_built']=mb; info['mps']=bool(mb and ma)
    except Exception as e:
        info['detail']['mps_err']=str(e)
    info['dev']='cuda' if info['cuda'] else ('mps' if info['mps'] else 'cpu')
except Exception as e:
    info['detail']['import_err']=str(e)
print(json.dumps(info))
""".strip()

    try:
        proc = mg.run(
            [str(py), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=build_esrgan_env(),
        )
        out = proc.stdout or "{}"
    except Exception as e:
        print_log(f"[torch.probe] exec failed: {e!r}")
        return None, False, False

    try:
        import json as _json
        import re as _re

        m = _re.search(r"\{.*\}$", out, _re.S)
        data = _json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}

    tor_ok = bool(data.get("tor_ok"))
    dev = data.get("dev")
    cu_ok = bool(data.get("cuda"))
    mps_ok = bool(data.get("mps"))
    tver = data.get("torch")
    cu_ver = data.get("cuda_ver")

    if tor_ok:
        extras = []
        if cu_ver:
            extras.append(f"cuda={cu_ver}")
        print_log(
            f"[torch] ok: dev={dev}, cuda_ok={cu_ok}, mps_ok={mps_ok}, torch={tver}{(' ' + ', '.join(extras)) if extras else ''}"
        )
        if dev not in {"cuda", "mps", "cpu"}:
            dev = "cpu"
        return dev, cu_ok, mps_ok

    head = "\n".join((out or "").splitlines()[:30])
    print_log(f"[torch] import failed\n--- head ---\n{head}\n--- end ---")
    return None, False, False


def ensure_png_inputs_for_tool(raw_dir: Path, *, tmp_root: Path) -> Path:
    pngs = list(raw_dir.glob("frame_*.png"))
    if pngs:
        print_log(
            f"ensure_png_inputs_for_tool: using existing PNGs in {raw_dir} (count={len(pngs)})"
        )
        return raw_dir
    bmps = sorted(raw_dir.glob("frame_*.bmp"), key=ih.frame_index_from_name)
    if not bmps:
        print_log(
            f"ensure_png_inputs_for_tool: no PNGs/BMPs found in {raw_dir} – using as-is"
        )
        return raw_dir

    png_dir = raw_dir.parent / (raw_dir.name + "_png")
    shutil.rmtree(png_dir, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    exe = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y",
        "-f",
        "image2",
        "-start_number",
        "1",
        "-i",
        str(raw_dir / "frame_%06d.bmp"),
        "-vsync",
        "0",
        str(png_dir / "frame_%06d.png"),
    ]
    print_log(f"[BMP→PNG] cmd={' '.join(map(str, cmd))}")

    t0 = time.time()
    out = (
        mg.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        or ""
    )
    dur = time.time() - t0
    log_head_tail("BMP→PNG-out", out, head_lines=20, tail_lines=10)

    new_pngs = list(png_dir.glob("frame_*.png"))
    print_log(
        f"ensure_png_inputs_for_tool: converted BMP→PNG in {png_dir} (src={len(bmps)} -> dst={len(new_pngs)}; dt={dur:.1f}s)"
    )
    meta_src = raw_dir / "frames_meta.json"
    if meta_src.exists():
        try:
            shutil.copy2(meta_src, png_dir / "frames_meta.json")
            print_log("[BMP→PNG] copied frames_meta.json")
        except Exception as e:
            co.print_warning(f"[BMP→PNG] copy frames_meta.json failed: {e}")
    persist_context(raw_dir, png_dir, tag="bmp2png")
    return png_dir if new_pngs else raw_dir


def persist_context(raw_dir: Path, up_dir: Path, tag: str) -> Path:
    sess = _new_session_dir(tag)
    # Pfad-Zustand
    log_paths_state(
        title=f"snapshot:{tag}", raw_dir=raw_dir, input_dir=raw_dir, up_dir=up_dir
    )
    # Listings + Samples
    try:
        (sess / "tree").mkdir(exist_ok=True)
        (sess / "tree" / "raw.txt").write_text(
            _list_dir_tree(raw_dir), encoding="utf-8"
        )
        if up_dir.exists():
            (sess / "tree" / "up.txt").write_text(
                _list_dir_tree(up_dir), encoding="utf-8"
            )
    except Exception:
        pass
    ih.copy_samples(raw_dir, sess / "samples" / raw_dir.name)
    if up_dir.exists():
        ih.copy_samples(up_dir, sess / "samples" / up_dir.name)
    return sess


def normalize_upscaled_sequence(
    up_dir: Path,
    input_dir: Path,
    esrgan_root: Path,
    expected_total: Optional[int] = None,
    created_after: Optional[float] = None,
) -> int:
    import re
    import shutil

    up_dir.mkdir(parents=True, exist_ok=True)
    ts = float(created_after) if (created_after is not None) else 0.0

    def _is_recent(p: Path) -> bool:
        try:
            return (created_after is None) or (p.stat().st_mtime >= ts)
        except Exception:
            return True

    # (A) Cleanup: nur "fertige" finale Namen löschen, **nie** *_out.png!
    final_name_re = re.compile(r"^frame_\d{6}\.png$")
    removed = 0
    for p in up_dir.iterdir():
        if not p.is_file():
            continue
        n = p.name
        if n.endswith("_out.png"):
            continue  # wichtig: Outputs bleiben liegen
        if final_name_re.match(n) and _is_recent(p):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        print_log(f"[PT-norm] cleaned pre-existing finals: {removed}", "_pytorch")

    # (B) Reihenfolge: bevorzugt aus input_dir; Fallback: aus den *_out in up_dir ableiten
    src_inputs = []
    try:
        if input_dir and input_dir.exists():
            src_inputs = sorted(
                input_dir.glob("frame_*.png"), key=ih.frame_index_from_name
            )
    except Exception:
        src_inputs = []

    if src_inputs:
        stems = [p.stem.replace("_out", "") for p in src_inputs]
    else:
        outs = sorted(up_dir.glob("frame_*_out.png"), key=ih.frame_index_from_name)
        stems = [re.sub(r"_out$", "", Path(p.stem).name) for p in outs]
        print_log(
            f"[PT-norm] inputs empty → derived stems from up_dir (*_out): {len(stems)}",
            "_pytorch",
        )

    if expected_total and len(stems) != int(expected_total):
        print_log(
            f"[PT-norm] WARN stems={len(stems)} expected={expected_total}", "_pytorch"
        )

    # (C) Normalisieren
    wrote = 0
    for i, stem in enumerate(stems, 1):
        dst_out = up_dir / f"{stem}_out.png"
        final = up_dir / f"frame_{i:06d}.png"

        if not dst_out.exists():
            relocated = relocate_output_for_frame(
                stem=stem,
                up_dir=up_dir,
                input_dir=input_dir,
                esrgan_root=esrgan_root,
                raw_dir=input_dir,
            )
            if not (relocated and relocated.exists()):
                continue

        try:
            if final.exists():
                try:
                    final.unlink()
                except Exception:
                    pass
            try:
                dst_out.replace(final)
            except Exception:
                shutil.copy2(dst_out, final)
                try:
                    dst_out.unlink()
                except Exception:
                    pass
            wrote += 1
        except Exception as e:
            co.print_warning(f"[normalize] {stem}_out -> {final.name} failed: {e}")

    # (D) Aufräumen
    for p in up_dir.glob("*_out.png"):
        try:
            p.unlink()
        except Exception:
            pass
    if wrote > 0:
        shutil.rmtree(up_dir / "__pool_tmp__", ignore_errors=True)

    # (E) Notfallpfad: Falls wrote==0 aber noch *_out vorhanden → direkt nummerieren
    if wrote == 0:
        outs = sorted(up_dir.glob("frame_*_out.png"), key=ih.frame_index_from_name)
        if outs:
            print_log(
                f"[PT-norm] EMERGENCY: wrote=0 but *_out={len(outs)} → direct rename",
                "_pytorch",
            )
            for i, src in enumerate(outs, 1):
                final = up_dir / f"frame_{i:06d}.png"
                try:
                    if final.exists():
                        try:
                            final.unlink()
                        except Exception:
                            pass
                    try:
                        src.replace(final)
                    except Exception:
                        shutil.copy2(src, final)
                        try:
                            src.unlink()
                        except Exception:
                            pass
                    wrote += 1
                except Exception as e:
                    co.print_warning(
                        f"[normalize/emergency] {src.name} -> {final.name} failed: {e}"
                    )
            for p in up_dir.glob("*_out.png"):
                try:
                    p.unlink()
                except Exception:
                    pass
            if wrote > 0:
                shutil.rmtree(up_dir / "__pool_tmp__", ignore_errors=True)

    exp_str = (
        str(expected_total)
        if (isinstance(expected_total, int) and expected_total > 0)
        else "?"
    )
    print_log(f"[PT-norm] EXIT wrote={wrote} expected≈{exp_str}", "_pytorch")
    return wrote


def stub_shm_log(shm_path: Path, persisted: Path) -> None:
    """Erzeuge eine kleine 'Stub'-Logdatei im /dev/shm, die auf den persistenten Logpfad verweist."""
    try:
        shm_path.parent.mkdir(parents=True, exist_ok=True)
        msg = f"This log is persisted at:\n{persisted}\n"
        if not shm_path.exists():
            shm_path.write_text(msg, encoding="utf-8")
    except Exception:
        pass


def ensure_cli_has_weightish(
    help_text: str,
    argv: List[str],
    *,
    model_name: Optional[str] = None,
    model_path: Optional[Path] = None,
) -> List[str]:
    """
    Stellt sicher, dass die CLI ein Weight-Argument bekommt:
      - bevorzugt '--weights' wenn vorhanden (dein Runner),
      - sonst '--model_path' (Upstream).
    Wenn beides nicht unterstützt → keine Änderung.
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

    # Keine Zielinfo? dann Ende.
    if not model_path or not isinstance(model_path, Path) or not model_path.is_file():
        return argv

    if has_weights_flag:
        print_log(f"[PT-util] inject --weights {model_path}")
        return [*argv, "--weights", str(model_path)]
    if has_model_path:
        print_log(f"[PT-util] inject --model_path {model_path}")
        return [*argv, "--model_path", str(model_path)]
    # Nichts bekannt – neutral bleiben
    print_log("[PT-util] no supported weight flag in help → skip injection")
    return argv


# ========== 8) PYTHON-WEGE (PER-FRAME/DIR) + TTA-EMUL. ======================


def _is_upscaled_vs_raw(raw_dir: Path, candidate: Path, stem: str) -> bool:
    """True, wenn candidate größer als das Roh-Frame ist (und ganzzahliges Vielfaches)."""
    raw = ih.original_frame_path(raw_dir, stem)
    if not raw or not candidate.exists():
        return False
    w0, h0 = ih.get_wh_for_input(raw)
    if candidate.suffix.lower() == ".png":
        w1, h1 = ih.read_png_wh(candidate)
    else:
        w1, h1 = ih.get_wh_for_input(candidate)
    if w0 <= 0 or h0 <= 0 or w1 <= 0 or h1 <= 0:
        return False
    return w1 > w0 and h1 > h0 and (w1 % w0 == 0) and (h1 % h0 == 0)


def _collect_possible_outputs_for_stem(
    *, stem: str, up_dir: Path, input_dir: Path, esrgan_root: Path, raw_dir: Path
) -> List[Path]:
    """Sammelt potenzielle Output-Dateien (auch aus results/ und pool_tmp/**/out) und
    filtert Dubletten/Roh-Inputs weg."""
    pats = [f"{stem}_out.png", f"{stem}.png"]
    found: List[Path] = []

    # 1) Hauptausgabeordner
    for pat in pats:
        found += list(up_dir.glob(pat))

    # 2) Pool-Outputs (nur OUT-Verzeichnisse!)
    pool_tmp = up_dir / "__pool_tmp__"
    if pool_tmp.exists():
        found += list(pool_tmp.rglob(f"out/{stem}_out.png"))  # nur *_out.png zulassen

    # 3) realesrgan/results
    results_dir = esrgan_root / "results"
    if results_dir.exists():
        found += list(results_dir.rglob(f"{stem}_out.png"))
        # selten erzeugen Runner auch plain frame_*.png in results – nur wenn wirklich upscaled:
        cand_plain = list(results_dir.rglob(f"{stem}.png"))
        for p in cand_plain:
            if _is_upscaled_vs_raw(raw_dir, p, stem):
                found.append(p)

    # 4) input_dir (nur *_out.png; plain nur wenn sicher upscaled)
    found += list(input_dir.glob(f"{stem}_out.png"))
    plain_inp = input_dir / f"{stem}.png"
    if plain_inp.exists() and _is_upscaled_vs_raw(raw_dir, plain_inp, stem):
        found.append(plain_inp)

    # Entdoppeln & rohe Inputs verwerfen
    uniq: List[Path] = []
    seen: set[str] = set()
    raw_inp = input_dir / f"{stem}.png"
    for p in found:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        if p == raw_inp and not _is_upscaled_vs_raw(raw_dir, p, stem):
            print_log(f"[collect_out] DROP raw input (not upscaled): {p}")
            continue
        if p.exists():
            uniq.append(p)
            seen.add(rp)

    print_log(
        f"[collect_out] stem={stem} → candidates={len(uniq)}; up_dir={up_dir}; pool_tmp_outs={'yes' if pool_tmp.exists() else 'no'}"
    )
    return uniq


def relocate_output_for_frame(
    *, stem: str, up_dir: Path, input_dir: Path, esrgan_root: Path, raw_dir: Path
) -> Optional[Path]:
    """Kopiert/verschiebt den besten gefundenen Output an den Zielnamen im up_dir."""
    cands = _collect_possible_outputs_for_stem(
        stem=stem,
        up_dir=up_dir,
        input_dir=input_dir,
        esrgan_root=esrgan_root,
        raw_dir=raw_dir,
    )
    if not cands:
        print_log(f"[COLLECT] no candidates for {stem}")
        return None
    dst = up_dir / f"{stem}_out.png"
    for src in cands:
        try:
            if src.resolve() == dst.resolve():
                print_log(f"[COLLECT] already at destination for {stem}: {dst}")
                return dst
        except Exception:
            pass
    try:
        if not dst.exists():
            src_best = sorted(
                cands,
                key=lambda p: (
                    p.stat().st_mtime if p.exists() else 0,
                    p.stat().st_size if p.exists() else 0,
                ),
                reverse=True,
            )[0]
            print_log(
                f"[COLLECT] choose src={src_best} (mtime={src_best.stat().st_mtime}, bytes={src_best.stat().st_size}) → {dst}"
            )
            dst.write_bytes(src_best.read_bytes())
            try:
                if src_best.parent != up_dir:
                    src_best.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            print_log(
                f"[COLLECT] relocated {stem} → {dst} (bytes={dst.stat().st_size})"
            )
    except Exception as e:
        co.print_warning(f"Relocate failed for {stem}: {e}")
    return dst if dst.exists() else None


def build_realesrgan_cli_common(
    help_text: str,
    model: str,
    *,
    out_dir: Path,
    face_enhance: bool,
    denoise: Optional[float],
    scale: float,
    include_ext_and_suffix: bool = True,
    force_fp32: bool = False,
    tile_size: int = 0,
    tile_pad: int = 20,
    tta: bool = False,
) -> List[str]:
    # Flag-Discovery
    out_flag = pick_flag(
        help_text, ["--output", "--outdir", "--save-dir", "-o"], "--output"
    )
    dn_flag = pick_flag(help_text, ["-dn", "--denoise_strength"], None)
    fe_flag = pick_flag(help_text, ["--face_enhance"], None)
    scale_flag = pick_flag(help_text, ["-s", "--outscale"], "-s")
    ext_flag = pick_flag(help_text, ["--ext"], "--ext")
    suffix_fl = pick_flag(help_text, ["--suffix"], "--suffix")
    fp32_flag = pick_flag(help_text, ["--fp32"], None)
    tile_flag = pick_flag(help_text, ["--tile", "-t"], "--tile")
    tpad_flag = pick_flag(help_text, ["--tile_pad", "--tile-pad", "-tp"], "--tile_pad")
    tta_flag = pick_flag(help_text, ["--tta", "--tta-mode"], None)
    # Achtung: -n/--model_name ist nicht bei deinem Runner vorhanden → nur anhängen, wenn es im Help-Text steht
    model_flag = pick_flag(help_text, ["-n", "--model_name", "--model-name"], None)

    args: List[str] = []
    if model_flag:
        args += [model_flag, model]

    if out_flag:
        args += [out_flag, str(out_dir)]
    if ("--save-dir" in help_text) and (out_flag not in ("--save-dir",)):
        args += ["--save-dir", str(out_dir)]

    if dn_flag and denoise is not None:
        args += [dn_flag, str(max(0.0, min(1.0, float(denoise))))]

    if scale_flag and scale:
        args += [scale_flag, str(scale)]

    if fe_flag and face_enhance:
        args += [fe_flag]

    if include_ext_and_suffix:
        if ext_flag:
            args += [ext_flag, "png"]
        if suffix_fl:
            args += [suffix_fl, "out"]

    if tile_flag and tile_size and tile_size > 0:
        args += [tile_flag, str(int(tile_size))]
        if tpad_flag:
            args += [tpad_flag, str(int(tile_pad))]

    if force_fp32 and fp32_flag:
        args += [fp32_flag]

    if tta and tta_flag:
        args += [tta_flag]

    print_log(f"[FLAGS] built-args: {' '.join(map(str, args))}")
    return args


# ========== 4) ESRGAN/NCNN HELPERS ==========================================


def probe_realesrgan_help(venv_python: Path, esr_script: Path) -> str:
    """Liest Hilfeausgabe von Python-Realesrgan (-h)."""
    try:
        proc = mg.run(
            [str(venv_python), str(esr_script), "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=build_esrgan_env(),
        )
        return proc.stdout or ""
    except Exception:
        return ""


# ========== 5) MEDIA & REMUX ================================================


def fps_label_from_meta(frames_dir: Path, fallback_fps: str | float | int) -> str:
    """
    Liefert eine FPS-Angabe, bevorzugt 'num/den', aus frames_meta.json.
    Fällt andernfalls auf fallback_fps zurück.
    """
    m = ih.read_frames_meta_from_dir(frames_dir)
    try:
        fn = int(m.get("fps_num") or 0)
        fd = int(m.get("fps_den") or 0)
        if fn > 0 and fd > 0:
            return f"{fn}/{fd}"
    except Exception:
        pass
    try:
        f = float(m.get("fps") or 0.0)
        if f > 0:
            return str(f)
    except Exception:
        pass
    return str(fallback_fps)


def probe_source_media_info(src: Path) -> Dict[str, Any]:
    """
    Erfasst grundlegende Container-/Streamdaten der Quelle (für spätere Remux-Map/Codecs/FPS).
    """
    exe = shutil.which("ffprobe") or "ffprobe"
    try:
        out = (
            mg.run(
                [
                    exe,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    "-i",
                    str(src),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            ).stdout
            or "{}"
        )
        data = json.loads(out)
    except Exception:
        data = {}
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    v0 = next((s for s in streams if s.get("codec_type") == "video"), {})
    r = v0.get("avg_frame_rate") or v0.get("r_frame_rate") or "0/1"
    fps_str = r if ("/" in r) else str(r)
    return {
        "container": (fmt.get("format_name") or "").split(",")[0],
        "streams": streams,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "has_subs": any(s.get("codec_type") == "subtitle" for s in streams),
        "has_attach": any(s.get("codec_type") == "attachment" for s in streams),
        "has_data": any(s.get("codec_type") == "data" for s in streams),
        "chapters": [],
        "fps_str": fps_str,
    }


def remux_with_source_streams(
    *,
    video_only: Path,
    source_media: Path,
    out_path: Path,
    container: Optional[str] = None,
    tmp_root: Optional[Path] = None,
) -> bool:
    """
    Nimmt das fertige **Video-only** (Upscale) und remuxt **alle** weiteren Streams (Audio, Subs, Attachments, Chapters)
    aus der Quelle wieder dazu. Erst Stream-Copy, dann graduelle Fallbacks.
    """
    exe = shutil.which("ffmpeg") or "ffmpeg"
    logs_dir = (tmp_root or out_path.parent) / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_file = logs_dir / "ffmpeg_remux.log"

    def _run(cmd: List[str]) -> Tuple[bool, str]:
        try:
            p = mg.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            out = p.stdout or ""
            ok = (p.returncode or 0) == 0
            return ok and out_path.exists() and out_path.stat().st_size > 0, out
        except Exception as e:
            return False, f"Exception: {e}"

    # 1) Versuch: alles stream-copy
    cmd1 = [
        exe,
        "-hide_banner",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source_media),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map",
        "1:t?",
        "-map",
        "1:d?",
        "-map_chapters",
        "1",
        "-map_metadata",
        "1",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "copy",
        "-c:t",
        "copy",
        "-c:d",
        "copy",
    ]
    if (container or "").lower() in ("mp4", "mov", "ismv"):
        cmd1 += ["-movflags", "+faststart"]
    cmd1 += [str(out_path)]

    ok, out = _run(cmd1)
    if ok:
        return True

    # 2) Fallback: Audio transkodieren
    cmd2 = [
        exe,
        "-hide_banner",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source_media),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map",
        "1:t?",
        "-map",
        "1:d?",
        "-map_chapters",
        "1",
        "-map_metadata",
        "1",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-c:s",
        "copy",
        "-c:t",
        "copy",
        "-c:d",
        "copy",
    ]
    if (container or "").lower() in ("mp4", "mov", "ismv"):
        cmd2 += ["-movflags", "+faststart"]
    cmd2 += [str(out_path)]
    ok, out2 = _run(cmd2)
    if ok:
        return True

    # 3) Letzter Fallback: keine Subs/Attachments/Data
    cmd3 = [
        exe,
        "-hide_banner",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source_media),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_chapters",
        "1",
        "-map_metadata",
        "1",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        str(out_path),
    ]
    ok, out3 = _run(cmd3)

    if not ok:
        try:
            log_file.write_text(out or out2 or out3 or "", encoding="utf-8")
        except Exception:
            pass
    return ok


# ========== 6) BACKEND-VERFÜGBARKEIT & FLAGS =================================


def backend_runtime_available(backend: str, caps: SystemCaps, model_key: str) -> bool:
    """Prüft grob, ob ein Backend für ein Model-Key lauffähig ist."""
    b = backend.lower().strip()
    if b == "pytorch":
        return caps.torch_device is not None  # Torch importierbar
    if b == "ncnn":
        # RealCUGAN braucht realcugan-ncnn-vulkan; Real-ESRGAN ncnn realesrgan-ncnn-vulkan
        if model_key.lower().startswith("realcugan"):
            return caps.realcugan_ok
        return caps.ncnn_ok
    if b == "onnx":
        # optional später
        return False
    return False


def pick_flag(
    help_text: str, candidates: List[str], default: Optional[str] = None
) -> Optional[str]:
    """
    Sucht Kandidaten-Flags robust in einem Help-Text.
    Treffer nur, wenn das Flag als eigenständiges Token vorkommt
    (links: Zeilenanfang/Whitespace/Trennzeichen; **kein '-' links**,
     rechts: Whitespace/Trennzeichen/=/Zeilenende).
    Verhindert False-Positives wie '-x' in 'realesr-general-x4v3'.
    """
    ht = help_text or ""
    if not ht:
        return default
    for f in candidates:
        token = f.strip()
        if not token:
            continue
        pattern = rf"(?m)(?:^|[\s,\[\(]){re.escape(token)}(?:\s|,|=|\)|\]|$)"
        if re.search(pattern, ht):
            print_log(f"[flags] detected {token!r}")
            return token
    print_log(f"[flags] none of {candidates} found → default={default}")
    return default


def log_head_tail(
    tag: str, text: str, head_lines: int = 60, tail_lines: int = 30
) -> None:
    """Loggt Kopf- und Endeabschnitt eines größeren Textes (z. B. Prozess-Output)."""
    try:
        lines = (text or "").splitlines()
        head = "\n".join(lines[: max(0, head_lines)])
        tail = (
            "\n".join(lines[-max(0, tail_lines) :]) if len(lines) > head_lines else ""
        )
        print_log(f"[{tag}] ---- HEAD ----\n{head}")
        if tail:
            print_log(f"[{tag}] ----  TAIL ----\n{tail}")
    except Exception as e:
        print_log(f"[{tag}] log format error: {e!r}")


def log_cmd(tag: str, argv: list[str]) -> None:
    """Schreibt eine schön lesbare Kommandozeile ins Log."""
    try:
        printable = " ".join(map(str, argv))
        print_log(f"[{tag}] cmd: {printable}")
    except Exception:
        pass


def dump_flag_diagnostics(context: str, help_text: str) -> None:
    """Hilfstext (Head) zur Flag-Diagnostik im Log anreißen."""
    try:
        head = "\n".join((help_text or "").splitlines()[:40])
        print_log(
            f"[FLAGS] Context={context}\n--- help (head) ---\n{head}\n--- end help ---"
        )
    except Exception:
        pass


def detect_gpu_info(venv_python: Path) -> Tuple[bool, str, int]:
    """Ermittelt (cuda_ok, name, vram_mb) via Torch im Subprozess (robust gegen Noise)."""
    try:
        proc = mg.run(
            [
                str(venv_python),
                "-c",
                "import json, importlib; "
                "torch=importlib.import_module('torch'); "
                "ok=torch.cuda.is_available(); "
                "name=torch.cuda.get_device_name(0) if ok else ''; "
                "vram=int(torch.cuda.get_device_properties(0).total_memory/1024/1024) if ok else 0; "
                "print(json.dumps({'ok':ok,'name':name,'vram':vram}))",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=build_esrgan_env(),
        )
        out = proc.stdout or "{}"
        m_ok = re.search(r'"ok"\s*:\s*(true|false)', out, re.I)
        m_name = re.search(r'"name"\s*:\s*"([^"]*)"', out)
        m_vram = re.search(r'"vram"\s*:\s*(\d+)', out)
        ok = m_ok and m_ok.group(1).lower() == "true"
        name = m_name.group(1) if m_name else ""
        vram = int(m_vram.group(1)) if m_vram else 0
        return bool(ok), name, vram
    except Exception:
        return False, "", 0


def prefer_fp32_default() -> bool:
    """Standard-FP32-Zwang (via ESRGAN_FORCE_FP32=1)."""
    return str(os.environ.get("ESRGAN_FORCE_FP32", "")).strip() == "1"


def pick_tile_size(vram_mb: int, force_fp32: bool, scale: int) -> int:
    """
    Heuristik zur Tile-Größe (0 = Full Frame).
    FP32 kostet VRAM → konservativere Kappen.
    """
    if vram_mb <= 0:
        return 256 if force_fp32 else 512
    if not force_fp32 and vram_mb >= 15360:
        return 0  # volle Bilder
    if force_fp32 and vram_mb >= 20480:
        return 0
    if force_fp32:
        if vram_mb >= 16384:
            return 640
        if vram_mb >= 12288:
            return 512
        if vram_mb >= 8192:
            return 384
        return 256
    else:
        if vram_mb >= 12288:
            return 768
        if vram_mb >= 8192:
            return 640
        if vram_mb >= 6144:
            return 512
        if vram_mb >= 4096:
            return 384
        return 256


def torch_versions(venv_python: Path) -> Tuple[str, str]:
    """Liest (torch_version, cuda_version) robust aus dem Venv."""
    try:
        code = "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': getattr(torch.version,'cuda',None)}))"
        proc = mg.run(
            [str(venv_python), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_esrgan_env(),
            check=False,
        )
        out = proc.stdout or ""
        m1 = re.search(r'"torch"\s*:\s*"([^"]+)"', out)
        m2 = re.search(r'"cuda"\s*:\s*"([^"]+)"', out)
        return (m1.group(1) if m1 else "unknown", m2.group(1) if m2 else "unknown")
    except Exception:
        return ("unknown", "unknown")


# ========== 7) WORKER-PLANUNG & PROFILE =====================================


def normalize_worker_profile_name(p: Optional[str]) -> Optional[str]:
    """Normalisiert/aliast Profilnamen auf kanonische Tokens."""
    if not p:
        return None
    s = str(p).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "no": "no_parallelisation",
        "none": "no_parallelisation",
        "serial": "no_parallelisation",
        "no_parallel": "no_parallelisation",
        "no_parallelisation": "no_parallelisation",
        "minimal": "minimal",
        "min": "minimal",
        "minimum": "minimal",
        "low": "minimal",
        "medium": "medium",
        "med": "medium",
        "mid": "medium",
        "auto": "auto",
        "max": "max",
        "maximum": "max",
        "performance": "max",
    }
    return aliases.get(s, s)


def _model_worker_bias(model: str) -> int:
    """
    Worker-Bias je nach Modellfamilie:
      - realesr-general-x4v3         → +1
      - realesr-animevideov3         → +2
      - Plus-Variante (enthält 'plus') → -2
    """
    m = (model or "").strip().lower()
    if not m:
        return 0
    if "RealESRGAN_x4plus_anime_6B" in m:
        return -3
    if "plus" in m:
        return -3
    if "realesr-animevideov3" in m:
        return +2
    if "realesr-general-x4v3" in m:
        return 0
    if "realcugan" in m:
        return +5
    return 0


def suggest_parallel_workers(
    *,
    vram_mb: int,
    tile_size: int,
    prefer_fp32: bool,
    total_frames: int,
    input_megapixels: float,
    scale: int,
    model: str,
    face_enhance: Optional[bool] = False,
    user_profile: Optional[str] = None,
) -> int:
    """
    Schlägt einen sinnvollen Worker-Wert inkl. Profile vor:
      - no_parallelisation → 1
      - minimal            → 2
      - medium             → ≈65% des Basiswerts (>=2, <=Basis)
      - auto               → Basis * (1 - Headroom)
      - max                → konservativ erhöht, VRAM-Kappen
    """
    # Hard-Override via Env
    if os.environ.get("AI_WORKERS"):
        try:
            return max(1, int(os.environ["AI_WORKERS"]))
        except Exception:
            pass

    prof = normalize_worker_profile_name(user_profile)

    # feste Profile
    if prof == "no_parallelisation":
        return 1
    if prof == "minimal":
        return 3 if total_frames >= 20 else 2

    # 0) Modell-Minimum für stabilen Pool (x4plus braucht ≥2)
    m_norm = (model or "").strip().lower().replace("-", "_")
    model_min = 3 if m_norm == "realesrgan_x4plus" else 2

    # 1) VRAM-basierter Basiswert (etwas aggressiver)
    if vram_mb >= 32768:
        base = 18
    elif vram_mb >= 24576:
        base = 14
    elif vram_mb >= 16384:
        base = 10
    elif vram_mb >= 12288:
        base = 6
    elif vram_mb >= 8192:
        base = 5
    else:
        base = 3

    # 2) FP32 kostet VRAM → drosseln (bei sehr viel VRAM milder)
    if prefer_fp32:
        if vram_mb < 16384:
            base -= 1
        if int(tile_size or 0) == 0 and vram_mb < 24576:
            base -= 1

    # 3) Tile=0 (Full Frame) → drosseln bei moderatem VRAM
    if int(tile_size or 0) == 0 and vram_mb < 12288:
        base -= 1

    # 4) Content-Schwere: MP * scale^2
    try:
        heavy = float(max(0.0, input_megapixels)) * float(max(1, int(scale))) ** 2
    except Exception:
        heavy = 0.0

    if heavy >= 40.0 and base > 3:
        base -= 3
    elif heavy >= 25.0 and base > 2:
        base -= 2
    elif heavy <= 2.5 and base < 12:
        base += 1
    elif heavy <= 1.2 and base < 12:
        base += 1

    # 5) Modell-Bias + leichtes Plus für x4plus
    base += _model_worker_bias(model)
    if m_norm == "realesrgan_x4plus":
        base += 1

    if face_enhance:
        base -= 1
    base = max(1, base)

    # 6) VRAM-basierte harte Obergrenzen
    if vram_mb >= 32768:
        vram_cap = 20
    elif vram_mb >= 24576:
        vram_cap = 16
    elif vram_mb >= 16384:
        vram_cap = 12
    elif vram_mb >= 12288:
        vram_cap = 8
    elif vram_mb >= 8192:
        vram_cap = 6
    else:
        vram_cap = 4

    # 7) Profile
    if prof == "medium":
        med = int(max(2, math.floor(base * 0.65)))
        med = min(med, base, vram_cap)
        if total_frames:
            med = min(med, int(total_frames))
        med = max(model_min if (total_frames or 2) >= model_min else 1, med)
        return med

    if prof == "max":
        bump = 0
        if not prefer_fp32:
            if vram_mb >= 24576:
                bump += 7
            elif vram_mb >= 16384:
                bump += 6
            elif vram_mb >= 12288:
                bump += 5
        if int(tile_size or 0) > 0:
            bump += 1
        w = base + bump
        w = min(w, vram_cap, base * 2)  # nie >2× Basis
        w = max(1, w)
        if total_frames:
            w = min(w, int(total_frames))
        w = max(model_min if (total_frames or 2) >= model_min else 1, w)
        return w

    # 8) "auto" – etwas geringerer Headroom (aggressiver)
    headroom_env = os.environ.get("AI_AUTO_HEADROOM", "").strip()
    try:
        headroom = float(headroom_env) if headroom_env else 0.25
    except Exception:
        headroom = 0.25
    headroom = min(0.5, max(0.0, headroom))

    auto_w = int(math.floor(base * (1.0 - headroom)))
    auto_w = max(1, auto_w)
    auto_w = min(auto_w, vram_cap)
    if total_frames:
        auto_w = min(auto_w, int(total_frames))
    auto_w = max(model_min if (total_frames or 2) >= model_min else 1, auto_w)

    return auto_w


# ========== 8) BLENDING / DIAG / CHUNKING ====================================


def blend_with_original(
    upscaled: Path, original: Path, opacity: float, out_path: Path
) -> bool:
    """Mischt Upscale (Video 0) mit skaliertem Original (Video 1) – einfacher Blend."""
    try:
        exe = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(upscaled),
            "-i",
            str(original),
            "-filter_complex",
            "[1:v][0:v]scale2ref[orig][up];[up][orig]blend=all_mode=normal:all_opacity=%0.4f[outv]"
            % float(opacity),
            "-map",
            "[outv]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "veryfast",
            str(out_path),
        ]
        print_log(f"[blend] cmd={' '.join(cmd)}")
        mg.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok = out_path.exists() and out_path.stat().st_size > 0
        print_log(f"[blend] wrote={ok} → {out_path}")
        return ok
    except Exception as e:
        co.print_warning(f"Blend-Fehler: {e}")
        return False


def log_paths_state(
    *,
    title: str,
    raw_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    up_dir: Optional[Path] = None,
    esrgan_root: Optional[Path] = None,
) -> None:
    """Ausführliche Path-/Perm-/Counter-Diagnostik (auch für /dev/shm)."""

    def _one(label: str, p: Optional[Path]) -> dict:
        info = {"label": label, "path": str(p) if p else None}
        if not p:
            return info
        try:
            exists = p.exists()
            is_dir = p.is_dir() if exists else False
            r = os.access(p, os.R_OK) if exists else False
            w = os.access(p, os.W_OK) if exists else False
            n_png = len(list(p.glob("*.png"))) if exists and is_dir else 0
            n_bmp = len(list(p.glob("*.bmp"))) if exists and is_dir else 0
            info.update(
                dict(
                    exists=exists,
                    is_dir=is_dir,
                    can_read=r,
                    can_write=w,
                    png=n_png,
                    bmp=n_bmp,
                )
            )
        except Exception as e:
            info["error"] = repr(e)
        return info

    payload = {
        "title": title,
        "raw": _one("raw_dir", raw_dir),
        "inp": _one("input_dir", input_dir),
        "up": _one("up_dir", up_dir),
        "root": _one("esrgan_root", esrgan_root),
    }
    # /dev/shm Disk-Usage (wenn vorhanden)
    try:
        shm = Path("/dev/shm")
        if shm.exists():
            st = shutil.disk_usage(str(shm))
            payload["dev_shm"] = {
                "total_mb": int(st.total / 1024 / 1024),
                "used_mb": int(st.used / 1024 / 1024),
                "free_mb": int(st.free / 1024 / 1024),
            }
    except Exception:
        pass
    print_log("[PATHS] " + json.dumps(payload, indent=2))


def compute_dynamic_chunk(
    free_gb: float,
    width: int,
    height: int,
    scale: int,
    total_frames: int,
    default_chunk: int,
) -> int:
    """
    Schätzt eine Chunk-Größe aus freiem Speicher & Bildbudget.
    Ziel: ~55% Budget nutzen, harte Min/Max-Klammern, Frames berücksichtigen.
    """
    free_bytes = max(0.0, float(free_gb)) * (1024**3)
    budget = max(256 * 1024**2, int(free_bytes * 0.55))
    bpp = 0.75
    raw_bytes = width * height * bpp
    up_bytes = (width * scale) * (height * scale) * bpp
    per_frame = int((raw_bytes + up_bytes) * 1.2)
    if per_frame <= 0:
        return default_chunk
    est = max(1, budget // per_frame)
    MIN_CHUNK = 100
    MAX_CHUNK = 2000
    chunk = int(max(MIN_CHUNK, min(MAX_CHUNK, est)))
    chunk = min(chunk, total_frames)
    if total_frames > 0:
        max_segments = 100
        chunk = max(chunk, max(1, total_frames // max_segments))
    if default_chunk and default_chunk <= MAX_CHUNK:
        if default_chunk * per_frame <= budget * 1.1:
            chunk = max(chunk, default_chunk)
    print_log(
        f"[dynamic_chunk] free_gb={free_gb:.2f} in={width}x{height} s={scale} frames={total_frames} → chunk={chunk}"
    )
    return int(chunk)


# ========== 9) AUFLÖSUNG / MODEL-SCALE ======================================


def detect_resolution_via_helper(path: Path, venv_python: Path) -> tuple[int, int]:
    """Ermittelt W×H via ffmpeghelper (Subprozess) mit ffprobe-Fallback."""
    print_log(f"[detect_resolution] path={path}")
    try:
        out = (
            he.run_py_out(
                venv_python,
                f"import ffmpeghelper;print(ffmpeghelper.detect_resolution({repr(str(path))}))",
            )
            or ""
        )
        m = re.search(r"(\d+)\D+(\d+)", str(out))
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if w > 0 and h > 0:
                print_log(f"[detect_resolution] via helper → {w}x{h}")
                return w, h
    except Exception as e:
        co.print_warning(f"[detect_resolution] helper fail: {e}")
    try:
        w_opt, h_opt, tmp = vec.ffprobe_geometry(path)
        if (
            isinstance(w_opt, int)
            and isinstance(h_opt, int)
            and w_opt > 0
            and h_opt > 0
        ):
            print_log(f"[detect_resolution] via ffprobe → {w_opt}x{h_opt}")
            return w_opt, h_opt
    except Exception as e:
        co.print_warning(f"[detect_resolution] ffprobe fail: {e}")
    print_log("[detect_resolution] fallback → 1280x720")
    return 1280, 720


def infer_model_scale(model_name: str) -> int:
    """
    Leitet die Faktor-Skalierung eines Models ab (defin.AI_MODEL_SCALES,
    sonst aus 'xN' im Namen; RealCUGAN → konservativ 2).
    """
    try:
        mapp = getattr(defin, "AI_MODEL_SCALES", None)
        if isinstance(mapp, dict) and model_name in mapp:
            s = int(mapp[model_name])
            return max(1, min(8, s))
    except Exception:
        pass
    m = re.search(r"[xX](\d+)", model_name or "")
    if m:
        try:
            return max(1, min(8, int(m.group(1))))
        except Exception:
            pass
    if str(model_name).lower().startswith("realcugan"):
        return 2
    return 4
