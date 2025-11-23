# ai_weights.py
from __future__ import annotations

import os
import sysconfig
from pathlib import Path

_GFPGAN_FILENAMES = ("GFPGANv1.4.pth", "GFPGANv1.3.pth")


def _as_path(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p).expanduser().resolve()
    return q if q.is_file() else None


def _site_pkg_dirs() -> list[Path]:
    out: list[Path] = []
    try:
        site = sysconfig.get_paths().get("purelib") or ""
        if site:
            out.append(Path(site))
    except Exception:
        pass
    return out


def resolve_gfpgan_weight(vm_base: Path | None = None) -> Path | None:
    """
    Reihenfolge:
      1) $GFPGAN_MODEL_PATH
      2) $VM_BASE/_thirdparty/GFPGAN/weights/GFPGANv1.4.pth
      3) $VM_BASE/real-esrgan/experiments/pretrained_models/GFPGANv1.4.pth
      4) site-packages/*/GFPGAN*/weights/GFPGANv*.pth
      5) ~/.cache/**/GFPGANv*.pth
    """
    # 1) ENV
    p = _as_path(os.environ.get("GFPGAN_MODEL_PATH"))
    if p:
        return p

    # 2) VM_BASE → _thirdparty
    bases: list[Path] = []
    if vm_base:
        bases += [
            vm_base / "_thirdparty" / "GFPGAN" / "weights",
            vm_base / "real-esrgan" / "experiments" / "pretrained_models",
            vm_base / "GFPGAN" / "weights",  # falls anderes Layout
        ]

    # 3) site-packages (grob suchen)
    for s in _site_pkg_dirs():
        bases += list(Path(s).glob("**/GFPGAN*/weights"))

    # 4) Cache-Orte
    home = Path.home()
    bases += [
        home / ".cache" / "gfpgan",
        home / ".cache",
    ]

    # Suche in Reihenfolge nach bekannten Dateinamen
    for b in bases:
        if not b or not b.is_dir():
            continue
        for name in _GFPGAN_FILENAMES:
            cand = b / name
            if cand.is_file():
                return cand

        # großzügige Suche nach GFPGANv*.pth
        hits = sorted(b.glob("**/GFPGANv*.pth"))
        if hits:
            return hits[0].resolve()

    return None
