# mem_guard.py
from __future__ import annotations

import atexit
import gc
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Global Cancel & Signal-Handler
# ──────────────────────────────────────────────────────────────────────────────
CANCEL = threading.Event()
_GLOBAL_TRACKER: Optional["PopenTracker"] = None  # für LSP/Type-Checker sichtbar
_GUARDED_LOCK = threading.Lock()
# Merkt Pfade + Option "recursive" (True=Ordner rekursiv löschen)
_GUARDED_PATHS: Dict[Path, bool] = {}  # {path: recursive}


def is_cancelled() -> bool:
    return CANCEL.is_set()


# mem_guard.py — ADD near top
_HANDLERS_INSTALLED: bool = False
_ESC_THREAD: Optional[threading.Thread] = None
_ESC_STOP: Optional[threading.Event] = None
_ESC_TTY_FH: Optional[Any] = None
_ESC_TTY_OLD: Optional[Any] = None


def _esc_is_active() -> bool:
    t = _ESC_THREAD
    return bool(t and t.is_alive())


@contextmanager
def escape_cancel_guard(nonintrusive: bool = True):
    """Aktiviert ESC-Abbruch nur innerhalb des Blocks (empfohlen für Upscaling-Phasen)."""
    enable_escape_cancel(nonintrusive=nonintrusive)
    try:
        yield
    finally:
        disable_escape_cancel()


@contextmanager
def suspend_escape_cancel():
    """ESC-Listener während einer Benutzereingabe pausieren (und anschließend wiederherstellen)."""
    was_active = _esc_is_active()
    if was_active:
        try:
            disable_escape_cancel()
        except Exception:
            pass
    try:
        yield
    finally:
        if was_active:
            try:
                enable_escape_cancel()
            except Exception:
                pass


def install_global_cancel_handlers() -> None:
    """Setzt SIGINT/SIGTERM-Handler genau einmal: setzt CANCEL und wirft KeyboardInterrupt."""
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return

    def _h(_signum, _frame):
        try:
            CANCEL.set()
        except Exception:
            pass
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _h)
        except Exception:
            pass
    _HANDLERS_INSTALLED = True


def enable_escape_cancel(*, nonintrusive: bool = True) -> None:
    """
    ESC-Abbruch, ohne die normale Terminal-Line-Disziplin kaputt zu machen:
      - TTY bleibt i.d.R. im Originalzustand (kanonisch, mit Echo).
      - In der Schleife wird für ein kurzes Fenster auf "raw" (ICANON/ECHO aus) umgeschaltet,
        0..N Bytes gelesen und sofort der Originalzustand wiederhergestellt.
      - Nur ein "plain ESC" (allein, keine Folgebytes binnen kurzer Frist) triggert Cancel.
      - Alle anderen Bytes (inkl. Backspace 0x7F, Pfeile, Alt-Sequenzen) werden mit TIOCSTI
        verlustfrei zurück in den TTY-Puffer injiziert (mit Echo kurz aus, um Doppel-Echo zu vermeiden).
    """
    global _ESC_THREAD, _ESC_STOP, _ESC_TTY_FH, _ESC_TTY_OLD

    if _ESC_THREAD and _ESC_THREAD.is_alive():
        return
    if os.name != "posix":
        return
    if not sys.stdin.isatty():
        return

    try:
        import fcntl
        import select
        import termios  # type: ignore
    except Exception:
        return

    try:
        _ESC_TTY_FH = open("/dev/tty", "rb", buffering=0)
    except Exception:
        _ESC_TTY_FH = None
        return

    fd = _ESC_TTY_FH.fileno()
    try:
        _ESC_TTY_OLD = termios.tcgetattr(fd)
    except Exception:
        try:
            _ESC_TTY_FH.close()
        except Exception:
            pass
        _ESC_TTY_FH = None
        _ESC_TTY_OLD = None
        return

    if not hasattr(termios, "TIOCSTI"):
        try:
            _ESC_TTY_FH.close()
        except Exception:
            pass
        _ESC_TTY_FH = None
        _ESC_TTY_OLD = None
        return

    _ESC_STOP = threading.Event()

    def _inject_back(bs: bytes) -> None:
        """Bytes verlustfrei zurück in den TTY-Puffer injizieren (Echo kurz aus)."""
        if not bs:
            return
        try:
            orig = termios.tcgetattr(fd)
            noecho = list(orig)
            noecho[3] = noecho[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, noecho)
            try:
                for b in bs:
                    fcntl.ioctl(fd, termios.TIOCSTI, bytes([b]))  # type: ignore[arg-type]
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, orig)
        except Exception:
            pass

    def _read_raw_once(timeout: float) -> bytes:
        """Kurzzeitig RAW (ICANON+ECHO aus), nonblocking lesen, sofort zurückstellen."""
        try:
            orig = termios.tcgetattr(fd)
        except Exception:
            return b""
        raw = list(orig)
        raw[3] = raw[3] & ~(termios.ICANON | termios.ECHO)  # raw, kein Echo
        raw[6][termios.VMIN] = 0
        raw[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(fd, termios.TCSANOW, raw)
            r, _, _ = select.select([_ESC_TTY_FH], [], [], max(0.0, float(timeout)))
            if not r:
                return b""
            try:
                return os.read(fd, 64)  # mehrere Bytes auf einmal einsammeln
            except Exception:
                return b""
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, orig)
            except Exception:
                pass

    def _loop():
        try:
            ESC_GRACE = 0.12  # Zeitfenster für Alt/Pfeiltasten
            ESC = 0x1B
            CTRL_C = 0x03
            CTRL_K = 0x0B

            while not (_ESC_STOP.is_set() if _ESC_STOP else False):
                data = _read_raw_once(0.05)
                if not data:
                    time.sleep(0.03)
                    continue

                # Strg+C / Strg+K -> immer als Abbruch werten
                if any(b in (CTRL_C, CTRL_K) for b in data):
                    try:
                        CANCEL.set()
                    except Exception:
                        pass
                    try:
                        os.kill(os.getpid(), signal.SIGINT)
                    except Exception:
                        pass
                    break

                # Kein ESC drin → einfach zurück in den Input-Puffer
                if ESC not in data:
                    _inject_back(data)
                    continue

                # ESC steckt drin
                # Mehrere Bytes und erstes ist ESC → sehr wahrscheinlich eine Escape-Sequenz (Pfeile/Alt)
                if data[0] == ESC and len(data) > 1:
                    _inject_back(data)
                    continue

                # Genau ein ESC-Byte → könnte "plain ESC" sein
                if data == bytes([ESC]):
                    t_end = time.time() + ESC_GRACE
                    tail = bytearray()
                    while time.time() < t_end:
                        more = _read_raw_once(0.0)
                        if not more:
                            time.sleep(0.005)
                            continue
                        tail.extend(more)

                    if not tail:
                        # Wirklich nur ESC → Cancel
                        try:
                            CANCEL.set()
                        except Exception:
                            pass
                        try:
                            os.kill(os.getpid(), signal.SIGINT)
                        except Exception:
                            pass
                        break

                    # ESC + weitere Bytes → Escape-Sequenz → alles zurück injizieren
                    _inject_back(bytes([ESC]) + bytes(tail))
                    continue

                # Fallback: ESC irgendwo in der Mitte → zurück injizieren
                _inject_back(data)

        finally:
            try:
                if _ESC_TTY_OLD is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, _ESC_TTY_OLD)
            except Exception:
                pass
            try:
                if _ESC_TTY_FH:
                    _ESC_TTY_FH.close()
            except Exception:
                pass

    _ESC_THREAD = threading.Thread(target=_loop, daemon=True)
    _ESC_THREAD.start()


def disable_escape_cancel() -> None:
    """Stoppt den globalen ESC-Listener (falls aktiv) und stellt TTY-Modus wieder her."""
    global _ESC_THREAD, _ESC_STOP
    try:
        if _ESC_STOP is not None:
            _ESC_STOP.set()
        if _ESC_THREAD and _ESC_THREAD.is_alive():
            _ESC_THREAD.join(timeout=1.0)
    except Exception:
        pass
    finally:
        _ESC_THREAD = None
        _ESC_STOP = None


# Am Ende der Datei existiert schon atexit.register(_on_exit). Ergänze:
atexit.register(disable_escape_cancel)


# ──────────────────────────────────────────────────────────────────────────────
# Guarded Paths: bei Exit/Abbruch löschen
# ──────────────────────────────────────────────────────────────────────────────
_GUARDED_LOCK = threading.Lock()
# Merkt Pfade + Option "recursive" (True=Ordner rekursiv löschen)
_GUARDED_PATHS: Dict[Path, bool] = {}  # {path: recursive}


def guard_path(path: str | Path, *, recursive: bool = True) -> Path:
    """
    Registriert eine Datei oder einen Ordner für die automatische Löschung
    beim Programmende/Abbruch. Ordner werden mit recursive=True via rmtree gelöscht.
    Mehrfaches Registrieren ist unkritisch.
    """
    p = Path(path)
    try:
        # resolve() wenn möglich, sonst absolute Pfadangabe
        p = p.resolve(strict=False)
    except Exception:
        p = p.absolute()
    with _GUARDED_LOCK:
        _GUARDED_PATHS[p] = bool(recursive)
    return p


def unguard_path(path: str | Path) -> None:
    """Entfernt einen Pfad wieder aus der Guard-Liste (falls vorhanden)."""
    p = Path(path)
    try:
        p = p.resolve(strict=False)
    except Exception:
        p = p.absolute()
    with _GUARDED_LOCK:
        _GUARDED_PATHS.pop(p, None)


def guard_paths(paths: Iterable[str | Path], *, recursive: bool = True) -> None:
    """Mehrere Pfade auf einmal registrieren."""
    for x in paths:
        guard_path(x, recursive=recursive)


def _safe_delete(p: Path, recursive: bool) -> None:
    try:
        if p.is_symlink() or p.is_file():
            # Symlink/Datei: unlink
            try:
                p.unlink(missing_ok=True)  # Py3.8+: fallback per try/except
            except TypeError:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            return
        if p.is_dir():
            if recursive:
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.rmdir()
                except Exception:
                    # Fällt still zurück (z. B. wenn nicht leer)
                    pass
    except Exception:
        # Robust: niemals Exceptions propagieren
        pass


def cleanup_guarded_paths_now() -> None:
    """
    Sofortige Bereinigung aller registrierten Pfade (z. B. wenn du KeyboardInterrupt
    abfängst und direkt aufräumen willst). Danach ist die Liste leer.
    """
    with _GUARDED_LOCK:
        items = list(_GUARDED_PATHS.items())
        _GUARDED_PATHS.clear()
    # Dateien zuerst, dann Verzeichnisse (flacher→tiefer), um Kollisionen zu vermeiden
    files = [p for p, _rec in items if p.is_file() or p.is_symlink()]
    # Dateien löschen
    for p in files:
        _safe_delete(p, recursive=False)
    # Verzeichnisse (rekursiv nach Bedarf), tiefste zuerst
    for p, rec in sorted(
        items, key=lambda kv: (str(kv[0]).count(os.sep)), reverse=True
    ):
        if p in files:
            continue
        _safe_delete(p, recursive=rec)


# ──────────────────────────────────────────────────────────────────────────────
# CUDA/RAM-Freigabe – **initialisiert CUDA NICHT unnötig**
# ──────────────────────────────────────────────────────────────────────────────
def _torch_cuda_is_initialized() -> bool:
    """Ohne Import von torch prüfen, ob in DIESEM Prozess CUDA bereits initialisiert ist."""
    try:
        if "torch" not in sys.modules:
            return False
        torch = sys.modules["torch"]
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return False
        # bevorzugt öffentliche API, fallback auf interne Flagge
        if hasattr(cuda, "is_initialized"):
            return bool(cuda.is_initialized())  # type: ignore[attr-defined]
        return bool(getattr(cuda, "_initialized", False))
    except Exception:
        return False


def free_ram_vram(
    *, aggressive: bool = False, only_if_initialized: bool = True
) -> None:
    """Gibt belegten RAM und – falls CUDA bereits initialisiert – VRAM frei.
    Wichtig: Initialisiert CUDA **nicht** in Prozessen, die bisher ohne GPU auskamen."""
    # VRAM (best effort), aber nur, wenn CUDA in diesem Prozess bereits lebt
    try:
        if (not only_if_initialized) or _torch_cuda_is_initialized():
            torch = sys.modules.get("torch", None)
            if torch is None:
                # Nur importieren, wenn wir wirklich freigeben wollen
                import torch as _t  # type: ignore

                torch = _t
            try:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()  # gibt unbenutzte Blöcke an den Treiber frei
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()  # räumt evtl. IPC-Segmente
                except Exception:
                    pass
                if aggressive and hasattr(torch.cuda, "reset_peak_memory_stats"):
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    # RAM
    try:
        gc.collect()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# NVML-Snapshots (optional) – wer hält VRAM?
# ──────────────────────────────────────────────────────────────────────────────
def gpu_process_snapshot(device_index: int = 0) -> List[Dict[str, Any]]:
    """Liefert eine Liste laufender GPU-Prozesse (PID, Name, used_mem, typ). Benötigt `pynvml`."""
    out: List[Dict[str, Any]] = []
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            dev = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            # compute + graphics separat abfragen (v2 liefert usedGpuMemory)
            for getter, typ in (
                (getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2"), "compute"),
                (
                    getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v2"),
                    "graphics",
                ),
            ):
                try:
                    for p in getter(dev):
                        name = ""
                        try:
                            # /proc/<pid>/cmdline ist oft ausreichend, psutil wäre optional
                            cmd = Path(f"/proc/{p.pid}/cmdline")
                            if cmd.exists():
                                name = (
                                    cmd.read_text(errors="ignore")
                                    .replace("\x00", " ")
                                    .strip()
                                )
                        except Exception:
                            pass
                        out.append(
                            {
                                "pid": int(p.pid),
                                "used_mem": int(getattr(p, "usedGpuMemory", 0) or 0),
                                "type": typ,
                                "cmd": name or "",
                            }
                        )
                except Exception:
                    pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        # pynvml nicht installiert oder keine NV-Karte
        pass
    return out


def log_gpu_processes(tag: str = "GPU-PROCS", device_index: int = 0) -> None:
    """Einfaches Logging der aktuellen VRAM-Halter."""
    try:
        procs = gpu_process_snapshot(device_index=device_index)
        if not procs:
            print(f"[{tag}] no GPU processes detected (device={device_index})")
            return
        total = sum(p["used_mem"] for p in procs)
        print(
            f"[{tag}] device={device_index} procs={len(procs)} total_mem={total / 1024 / 1024:.1f}MB"
        )
        for p in sorted(procs, key=lambda x: -x["used_mem"]):
            mb = p["used_mem"] / (1024 * 1024)
            print(
                f"  - pid={p['pid']:>6}  {mb:6.1f}MB  {p['type']:<8}  {p.get('cmd', '')[:120]}"
            )
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Subprozess-Helfer
# ──────────────────────────────────────────────────────────────────────────────
def _close_quietly(fh) -> None:
    try:
        if fh and hasattr(fh, "close"):
            fh.close()
    except Exception:
        pass


def close_popen_streams(proc: subprocess.Popen) -> None:
    """Schließt evtl. angeheftete Logfile-Handles (siehe PopenTracker.spawn)."""
    try:
        _close_quietly(getattr(proc, "_stdout_to_close", None))
    except Exception:
        pass


def terminate_process(
    proc: subprocess.Popen, *, name: str = "subprocess", timeout: float = 2.0
) -> None:
    """Robustes Beenden: SIGINT → SIGTERM → SIGKILL / kill(), inkl. Prozessgruppe."""
    if not proc:
        return
    try:
        if proc.poll() is not None:
            return
        # 1) freundlich
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except Exception:
            pass

        if proc.poll() is None:
            # 2) bestimmter
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=timeout)
            except Exception:
                pass

        if proc.poll() is None:
            # 3) hart
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # POSIX
                else:
                    proc.kill()  # Windows
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=timeout)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        close_popen_streams(proc)


class PopenTracker:
    """Verwaltet gestartete Subprozesse & sorgt für sauberes Aufräumen."""

    def __init__(self, label: str = "job"):
        self._label = label
        self._lock = threading.Lock()
        self._procs: List[subprocess.Popen] = []

    def spawn(
        self,
        args: Iterable[str] | List[str],
        *,
        cwd: Optional[Path | str] = None,
        env: Optional[Dict[str, str]] = None,
        log_to: Optional[Path | str] = None,
        text: bool = True,
        **popen_kwargs: Any,
    ) -> subprocess.Popen[str]:
        stdout_dest: Any = subprocess.DEVNULL
        if log_to:
            try:
                p = Path(log_to)
                p.parent.mkdir(parents=True, exist_ok=True)
                stdout_dest = open(p, "w", encoding="utf-8", buffering=1)
            except Exception:
                stdout_dest = subprocess.DEVNULL

        # Eigene Prozessgruppe (POSIX) / neues Proc-Group-Flag (Windows)
        popen_kwargs.setdefault("start_new_session", True)  # POSIX
        if os.name == "nt":
            popen_kwargs.setdefault(
                "creationflags", getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )

        proc = subprocess.Popen(  # type: ignore[call-arg]
            list(map(str, args)),
            cwd=(str(cwd) if cwd else None),
            env=env,
            stdout=stdout_dest,
            stderr=subprocess.STDOUT,
            text=text,
            **popen_kwargs,
        )
        if stdout_dest not in (None, subprocess.DEVNULL):
            try:
                setattr(proc, "_stdout_to_close", stdout_dest)
            except Exception:
                pass
        with self._lock:
            self._procs.append(proc)
        return proc

    def track(self, proc: subprocess.Popen) -> subprocess.Popen:
        with self._lock:
            self._procs.append(proc)
        return proc

    def kill_all(self, *, timeout: float = 1.5) -> None:
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for p in procs:
            try:
                terminate_process(p, name=self._label, timeout=timeout)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.kill_all()
        # ACHTUNG: hier NICHT erzwungen CUDA initialisieren
        free_ram_vram(only_if_initialized=True)


# ──────────────────────────────────────────────────────────────────────────────
# vRAM-Guard
# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def vram_guard():
    try:
        yield
    finally:
        free_ram_vram(only_if_initialized=True)


# ──────────────────────────────────────────────────────────────────────────────
# Drop-in Wrapper für run/popen mit sanfterm Speicher-Freigabe
# ──────────────────────────────────────────────────────────────────────────────
def run(*args, **kwargs):
    """
    Drop-in Wrapper für subprocess.run mit kleinen Default-Sicherungen:
    - eigene Prozessgruppe (besseres Killen auf POSIX/Windows)
    - nach dem Aufruf RAM/VRAM freigeben (ohne CUDA neu zu initialisieren)
    """
    kwargs.setdefault("text", True)
    kwargs.setdefault("start_new_session", True)
    if os.name == "nt":
        kwargs.setdefault(
            "creationflags", getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    try:
        return subprocess.run(*args, **kwargs)
    finally:
        free_ram_vram(only_if_initialized=True)


def popen(*args, **kwargs):
    """
    Drop-in Wrapper für subprocess.Popen, der über einen globalen PopenTracker
    spawnt (für sauberes Aufräumen/Killen). Optional `log_to="file.log"` nutzen.
    """
    global _GLOBAL_TRACKER
    log_to = kwargs.pop("log_to", None)
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = PopenTracker("global")
    return _GLOBAL_TRACKER.spawn(*args, log_to=log_to, **kwargs)


def kill_all():
    """Beendet alle über mem_guard.popen() gestarteten Prozesse (und räumt Speicher)."""
    global _GLOBAL_TRACKER
    try:
        if _GLOBAL_TRACKER is not None:
            _GLOBAL_TRACKER.kill_all()
    finally:
        free_ram_vram(only_if_initialized=True)


# ──────────────────────────────────────────────────────────────────────────────
# atexit: erst Prozesse killen, dann Speicher räumen (ohne CUDA neu zu initialisieren)
# ──────────────────────────────────────────────────────────────────────────────
def _on_exit():
    try:
        kill_all()
    except Exception:
        pass
    try:
        cleanup_guarded_paths_now()
    except Exception:
        pass
    finally:
        free_ram_vram(only_if_initialized=True)


# einmalig registrieren (ersetzt früheres direktes free_ram_vram-atexit)
atexit.register(_on_exit)
