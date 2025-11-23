#!/usr/bin/env python3
from __future__ import annotations

import os

import mem_guard as mg


def safe_input(prompt: str = "") -> str:
    """
    Liest Benutzereingabe robust:
      • pausiert ESC-Listener temporär
      • liest bevorzugt direkt von /dev/tty (unabhängig von sys.stdin)
      • fällt auf normales input() zurück, wenn /dev/tty nicht verfügbar ist
    """
    with mg.suspend_escape_cancel():
        # POSIX: direkt am Terminal lesen
        if os.name == "posix":
            try:
                with (
                    open("/dev/tty", "r", encoding="utf-8", errors="replace") as tin,
                    open(
                        "/dev/tty", "w", encoding="utf-8", buffering=1, errors="replace"
                    ) as tout,
                ):
                    if prompt:
                        tout.write(prompt)
                        tout.flush()
                    line = tin.readline()
                    if line == "":
                        raise EOFError
                    return line.rstrip("\n")
            except EOFError:
                # echtes EOF am Terminal – weiterreichen
                raise
            except Exception:
                # Fallback: normales input()
                return input(prompt)

        # Windows/sonst: best effort
        return input(prompt)
