#!/usr/bin/env python3
# loghandler

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ANSI-Escape-Sequenzen (Farben etc.) für das Log entfernen
_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
_DEFAULT_BACKUPS = 3


def _env_int(name: str, default: int) -> int:
    """Liest eine int-Umgebungsvariable robust ein."""
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


_MAX_LOG_BYTES = _env_int("VIDEOMANAGER_LOG_MAX_BYTES", _DEFAULT_MAX_BYTES)
_LOG_BACKUPS = max(0, _env_int("VIDEOMANAGER_LOG_BACKUPS", _DEFAULT_BACKUPS))


def _resolve_log_path(präfix: Optional[str] = None) -> str:
    """Ermittle den Log-Pfad mit sinnvollen Defaults."""
    log_path = None
    try:
        import definitions as defin

        p = getattr(defin, "LOG_FILE", None)
        if p:
            log_path = os.fspath(p)
    except Exception:
        pass

    env_path = os.environ.get("VIDEOMANAGER_LOG")
    if env_path and log_path is None:
        log_path = env_path

    if log_path is None:
        xdg_state = os.environ.get("XDG_STATE_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "state"
        )
        log_path = os.path.join(xdg_state, "videoManager", "videoManager.log")

    if präfix:
        p = Path(log_path)
        all_suffixes = "".join(p.suffixes)
        base = p.name[: -len(all_suffixes)] if all_suffixes else p.name
        new_name = f"{base}{präfix}{all_suffixes}"
        log_path = str(p.with_name(new_name))

    return log_path


def _rotate_log_file(log_path: str) -> None:
    """
    Einfache Größen-Rotation: file -> file.1 -> file.2 ...
    Wird ausgelöst, sobald die Datei größer als _MAX_LOG_BYTES ist.
    """
    try:
        max_bytes = _MAX_LOG_BYTES
        backups = _LOG_BACKUPS
        if max_bytes <= 0 or backups < 0:
            return

        p = Path(log_path)
        if not p.exists() or p.stat().st_size < max_bytes:
            return

        if backups == 0:
            # Kein Backup gewünscht → einfach truncaten
            with open(p, "w", encoding="utf-8"):
                return

        oldest = p.with_name(f"{p.name}.{backups}")
        if oldest.exists():
            try:
                oldest.unlink()
            except Exception:
                pass  # Rotation darf Logging nicht komplett blockieren

        for idx in range(backups, 0, -1):
            src = p.with_name(f"{p.name}.{idx - 1}" if idx > 1 else p.name)
            dst = p.with_name(f"{p.name}.{idx}")
            if src.exists():
                try:
                    src.rename(dst)
                except Exception:
                    pass
    except Exception:
        pass


def print_log(message: str, prefix: Optional[str] = None) -> None:
    """
    Schreibt eine Log-Zeile mit lokalem Zeitstempel ins Logfile.
    Beispielzeile: "[2025-10-17 19:42:11+02:00] Deine Nachricht"
    """
    log_path = _resolve_log_path(prefix)

    # Ordner sicherstellen (auch wenn definitions.LOG_FILE auf einen custom Pfad zeigt)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    _rotate_log_file(log_path)

    # Lokale Zeit inkl. UTC-Offset als "+HH:MM"
    now = datetime.now().astimezone()
    ts = now.strftime("%Y-%m-%d %H:%M:%S%z")  # "+HHMM"
    if len(ts) >= 5:  # zu "+HH:MM" umformen
        ts = ts[:-5] + ts[-5:-2] + ":" + ts[-2:]

    line = f"[{ts}] {str(message)}"
    line = _ANSI_RE.sub("", line)  # Farben/ESC aus Logs entfernen

    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
            f.flush()
    except Exception as e:
        # Fallback, wenn Schreiben fehlschlägt
        print(f"[print_log] {line} (write failed: {e})", file=sys.stderr)


def start_log() -> None:
    print_log(">>>>>>>>>>>>>>>>>>>>> Started VideoManager <<<<<<<<<<<<<<<<<<<<<<<<<")
