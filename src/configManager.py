#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_manager.py – Zentrale Verwaltung für videoManager-Config (INI)

Features
- Sucht/merged Config in dieser Reihenfolge (höchste Priorität zuerst):
  1) ENV Overrides (VIDEO_MANAGER__SECTION__KEY=...)
  2) Projekt-Config:   <project_root>/config.ini
  3) Benutzer-Config:  ~/.config/videoManager/config.ini
  4) Defaults (Template unten)
- Typisierte Getter (str/int/float/bool/path/enum) mit Fallback & Validierung
- Setzen von Werten und persistentes Speichern (user|project|auto)
- Kommentar-schonende, zeilenweise Key-Updates (Inline-Kommentare bleiben)
- Atomare Saves mit .bak-Backup
- Default-Datei/fehlende Keys automatisch anlegen (Migration)
- Pfad-Utilities (Expand ~, $VARS, absolut machen)
- „auto“-Resolver: cpu_threads, ffmpeg_path, aria2_path, torch_device, gpu_backend
- Einfache Heuristik für GPU-Backend (nvidia-smi → cuda, rocminfo → rocm, macOS → mps)

Abhängigkeiten: nur Standardbibliothek (configparser, shutil, pathlib, re, etc.)
Optional: integriert consoleOutput.py (co.debug/info/warn/error), sonst print.

Kurzbeispiel
------------
    cm = ConfigManager(project_root=Path(__file__).parent)
    cm.load()                         # lädt + merged + migriert fehlende Keys
    out_dir = cm.get_path("paths", "default_output_dir")
    threads = cm.get_int("hardware", "cpu_threads", auto_resolve=True)
    cm.set_value("ffmpeg", "preset", "slow", target="user")  # sicher & atomar
    cm.save("user")                    # explizit speichern (bei batch Updates)
"""

from __future__ import annotations

import configparser
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

# ───────────────────────── Optionales Projekt-Logging ─────────────────────────
try:
    import consoleOutput as co  # dein Projektmodul

    def _log_debug(*a: Any, **k: Any) -> None:
        getattr(co, "debug", lambda *args, **kwargs: None)(*a, **k)

    def _log_info(*a: Any, **k: Any) -> None:
        getattr(co, "info", print)(*a, **k)

    def _log_warn(*a: Any, **k: Any) -> None:
        getattr(co, "warn", print)(*a, **k)

    def _log_error(*a: Any, **k: Any) -> None:
        getattr(co, "error", lambda *args, **kwargs: print(*args, file=sys.stderr))(
            *a, **k
        )

except Exception:

    def _log_debug(*a: Any, **k: Any) -> None:
        return None

    def _log_info(*a: Any, **k: Any) -> None:
        print(*a, **k)

    def _log_warn(*a: Any, **k: Any) -> None:
        print(*a, **k)

    def _log_error(*a: Any, **k: Any) -> None:
        print(*a, file=sys.stderr, **k)


# ───────────────────────────── Default-Template ───────────────────────────────
DEFAULT_TEMPLATE = """# videoManager configuration
[general]
language = de              # de|en|es|...
os = auto                  # auto|linux|windows|mac
batch_mode_default = false
check_updates = true

[paths]
default_output_dir = ~/Videos/videoManager
temp_dir = ~/.cache/videoManager/tmp
ffmpeg_path = auto         # auto oder absoluter Pfad
aria2_path = auto          # auto oder absoluter Pfad

[hardware]
gpu_enabled = auto         # auto|true|false
gpu_backend = auto         # auto|cuda|rocm|mps|opencl|vulkan|none
cpu_threads = auto         # auto oder Zahl
prefer_vulkan = false

[ai]
enabled = true
backend = torch            # torch|ncnn|onnx
torch_device = auto        # auto|cuda|cpu|mps
realesrgan_model = realesr-general-x4v3
realesrgan_strength = 0.5
gfpgan_enabled = false
codeformer_enabled = false

[ffmpeg]
loglevel = error           # quiet|error|warning|info|debug
overwrite = false
hwaccel = auto             # auto|nvenc|vaapi|qsv|none
encoder_video = libx264    # z.B. libx264|h264_nvenc|libx265|hevc_nvenc|...
encoder_audio = aac
crf = 18
preset = casual

[ui]
unicode = true
colors = true
term_width = auto          # auto oder Zahl

[logging]
level = INFO               # DEBUG|INFO|WARNING|ERROR
file = ~/.local/share/videoManager/videoManager.log
rotate = true
keep_files = 5

[network]
download_retries = 10
timeout_sec = 90

[experimental]
ncnn_bin_dir =             # Pfad zu ncnn .bin/.param (falls nötig)
onnx_runtime_enabled = false

[safety]
dry_run = false
"""

# Für schnelle Validierung/Auto-Resolution (nur Keys, die besondere Behandlung haben)
_SPECIAL_KEYS = {
    ("hardware", "cpu_threads"),
    ("paths", "ffmpeg_path"),
    ("paths", "aria2_path"),
    ("hardware", "gpu_backend"),
    ("ai", "torch_device"),
    ("ui", "term_width"),
}

ENV_PREFIX = "VIDEO_MANAGER__"  # VIDEO_MANAGER__SECTION__KEY=...


def _optionxform_lower(optionstr: str) -> str:
    return optionstr.lower()


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _expand_path(p: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))


def _to_bool(val: Union[str, bool, int]) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _strip_inline_comment(val: str) -> str:
    """
    Entfernt einen Inline-Kommentar am Zeilenende.
    Respektiert Quotes und Escape-Sequenzen: # oder ; bleiben literal.
    """
    if not isinstance(val, str):
        return val
    out = []
    escaped = False
    in_s = False  # in single quotes
    in_d = False  # in double quotes

    for ch in val:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_d:
            in_s = not in_s
            out.append(ch)
            continue
        if ch == '"' and not in_s:
            in_d = not in_d
            out.append(ch)
            continue
        # Kommentarstart nur außerhalb von Quotes
        if not in_s and not in_d and ch in ("#", ";"):
            break
        out.append(ch)

    # Nur Tabs/Spaces am Rand kappen (kein Quote-Stripping o.ä.)
    return "".join(out).strip(" \t")


def _is_auto(val: Any) -> bool:
    return isinstance(val, str) and val.strip().lower() == "auto"


def _cpu_threads_auto() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def _detect_gpu_backend(prefer_vulkan: bool = False) -> str:
    # Sehr einfache Heuristik; kann bei Bedarf verfeinert werden.
    system = platform.system().lower()
    if system == "darwin":
        return "mps"
    if _which("nvidia-smi"):
        return "cuda" if not prefer_vulkan else "vulkan"
    if _which("rocminfo"):
        return "rocm"
    # Optional: Vulkan/OpenCL prüfen
    if prefer_vulkan and (
        _which("vulkaninfo") or os.path.exists("/usr/share/vulkan/icd.d")
    ):
        return "vulkan"
    if _which("clinfo"):
        return "opencl"
    return "none"


def _auto_ffmpeg_path() -> Optional[str]:
    p = _which("ffmpeg")
    return _expand_path(p) if p else None


def _auto_aria2_path() -> Optional[str]:
    p = _which("aria2c")
    return _expand_path(p) if p else None


def _auto_torch_device() -> str:
    # Minimal: CUDA vorhanden?
    if _which("nvidia-smi"):
        return "cuda"
    if platform.system().lower() == "darwin":
        return "mps"
    return "cpu"


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
        f.write(data)
    # Backup
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        try:
            shutil.copy2(path, bak)
        except Exception:
            pass
    os.replace(tmp_name, path)


def _find_section_span(text: str, section: str) -> Tuple[int, int]:
    """
    Liefert (start, end)-Indices des Abschnitts (inkl. Header) in 'text'.
    Wenn nicht vorhanden, return (-1, -1).
    """
    sec_re = re.compile(rf"^\[{re.escape(section)}\]\s*$", re.MULTILINE)
    m = sec_re.search(text)
    if not m:
        return (-1, -1)
    start = m.start()
    # Ende: nächster Abschnitt oder EOF
    next_m = sec_re.search(text, m.end() + 1)
    if next_m:
        # Das wäre der *gleiche* Abschnitt; wir brauchen "nächsten beliebigen Abschnitt"
        pass
    any_section_re = re.compile(r"^\[.+?\]\s*$", re.MULTILINE)
    next_any = any_section_re.search(text, m.end())
    end = next_any.start() if next_any else len(text)
    return (start, end)


def _update_key_in_section_block(
    block: str, key: str, new_value: str
) -> Tuple[str, bool]:
    """
    Ersetzt in 'block' die Zeile 'key = ...' (nur Wert) und erhält Inline-Kommentar.
    Falls key nicht existiert, fügt 'key = new_value' am Ende des Blocks ein.
    Returns: (neuer_block, found_and_replaced: bool)
    """
    # Suche Zeile mit Schlüssel; erhalte Einrückung, Delimiter, Inline-Kommentar
    # Beispiel: "key = old  # comment"
    key_re = re.compile(
        rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*(#.*))?$",
        re.MULTILINE,
    )
    m = key_re.search(block)
    if m:
        prefix = m.group(1)
        # m.group(2) ist alter Wert
        inline = m.group(3) or ""
        replacement_line = f"{prefix}{new_value}{inline}"
        new_block = block[: m.start()] + replacement_line + block[m.end() :]
        return new_block, True
    # Nicht gefunden → einfügen vor dem Blockende (vor erstem Trailing-Whitespace)
    insert_line = f"{key} = {new_value}\n"
    if block.endswith("\n"):
        return block + insert_line, False
    else:
        return block + "\n" + insert_line, False


def _set_value_linewise(text: str, section: str, key: str, new_value: str) -> str:
    """
    Kommentar-schonendes Setzen eines Werts in Rohtext.
    Legt Abschnitt/Key an, wenn sie fehlen.
    """
    start, end = _find_section_span(text, section)
    if start == -1:
        # Abschnitt fehlt → anhängen
        add = f"\n[{section}]\n{key} = {new_value}\n"
        # Falls Datei leer, kein Leading-Newline
        if not text.strip():
            add = f"[{section}]\n{key} = {new_value}\n"
        return text + add
    block = text[start:end]
    new_block, _ = _update_key_in_section_block(block, key, new_value)
    return text[:start] + new_block + text[end:]


def _parse_ini(text: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(
        interpolation=None, delimiters=("=",)
    )  # echtes 1-Tuple
    cp.optionxform = _optionxform_lower
    # configparser akzeptiert Inline-Kommentare mit '#' standardmäßig
    from io import StringIO

    cp.read_file(StringIO(text or ""))
    return cp


def _merge(
    a: configparser.ConfigParser, b: configparser.ConfigParser
) -> configparser.ConfigParser:
    """Merge b → a (b hat höhere Priorität)."""
    for sec in b.sections():
        if not a.has_section(sec):
            a.add_section(sec)
        for k, v in b.items(sec):
            a.set(sec, k, v)
    return a


class ConfigManager:
    def __init__(
        self, project_root: Optional[Path] = None, app_name: str = "videoManager"
    ) -> None:
        self.app_name = app_name
        self.project_root = Path(project_root) if project_root else None
        self.user_config_path = Path.home() / ".config" / app_name / "config.ini"
        self.project_config_path = (
            (self.project_root / "config.ini") if self.project_root else None
        )

        self._cp = configparser.ConfigParser(interpolation=None)
        self._cp.optionxform = _optionxform_lower
        self._source: Dict[Tuple[str, str], str] = (
            {}
        )  # (sec,key) -> 'env'|'project'|'user'|'default'
        self._raw_user_text = ""
        self._raw_project_text = ""
        self._raw_default_text = DEFAULT_TEMPLATE

    # ───────────────────────────── Laden / Mergen ─────────────────────────────
    def load(self) -> None:
        """Liest User/Projekt/Defaults ein, merged alles, migriert fehlende Keys."""
        self._raw_user_text = _read_file(self.user_config_path)
        self._raw_project_text = (
            _read_file(self.project_config_path) if self.project_config_path else ""
        )

        cp_default = _parse_ini(self._raw_default_text)
        cp_user = _parse_ini(self._raw_user_text)
        cp_proj = _parse_ini(self._raw_project_text)

        # Merge-Reihenfolge: default <- user <- project
        cp = _parse_ini("")  # leer
        _merge(cp, cp_default)
        _merge(cp, cp_user)
        _merge(cp, cp_proj)

        # ENV overrides (höchste Priorität)
        self._apply_env_overrides(cp)

        self._cp = cp

        # Source-Map neu aufbauen
        self._rebuild_source_map(cp_default, cp_user, cp_proj)

        # Migration: fehlende Keys in Dateien ergänzen (nicht invasiv)
        self._migrate_missing_keys()

    def reload(self) -> None:
        self.load()

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return {s: {k: v for k, v in self._cp.items(s)} for s in self._cp.sections()}

    # ─────────────────────────────── Getter (typed) ───────────────────────────
    def has(self, section: str, key: str) -> bool:
        return self._cp.has_section(section) and self._cp.has_option(section, key)

    def get_str(
        self, section: str, key: str, fallback: Optional[str] = None
    ) -> Optional[str]:
        if self.has(section, key):
            raw = self._cp.get(section, key)
            return _strip_inline_comment(raw)
        return fallback.strip(" \t") if isinstance(fallback, str) else fallback

    def get_bool(
        self,
        section: str,
        key: str,
        fallback: Optional[bool] = None,
        auto_resolve: bool = False,
    ) -> Optional[bool]:
        """
        Bool-Getter mit Tri-State-Unterstützung:

            off/false/no/0  -> False
            on/true/yes/1   -> True
            auto/""/sonst   -> fallback (typisch None)

        Nur wenn auto_resolve=True und (section,key)==("hardware","gpu_enabled")
        wird 'auto' speziell behandelt.
        """
        v = self.get_str(section, key)
        if v is None:
            return fallback

        s = str(v).strip().lower()

        # Spezieller Auto-Resolver für gpu_enabled, wenn explizit gewünscht
        if auto_resolve and _is_auto(s):
            if (section, key) == ("hardware", "gpu_enabled"):
                # „auto“ → heuristisch: GPU vorhanden?
                return self._detect_gpu_present()
            # Für andere Keys: 'auto' nicht erzwingen → fallback
            return fallback

        # Explizite True-/False-Tokens
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}

        if s in truthy:
            return True
        if s in falsy:
            return False

        # 'auto', leer oder irgendwas anderes → kein hartes Bool
        return fallback

    def get_int(
        self,
        section: str,
        key: str,
        fallback: Optional[int] = None,
        auto_resolve: bool = False,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> Optional[int]:
        v = self.get_str(section, key)
        if v is None:
            return fallback
        if auto_resolve and _is_auto(v):
            if (section, key) == ("hardware", "cpu_threads"):
                return _cpu_threads_auto()
            if (section, key) == ("ui", "term_width"):
                try:
                    import shutil as _sh

                    cols = _sh.get_terminal_size((80, 20)).columns
                    return max(40, cols)
                except Exception:
                    return 80
            return fallback
        try:
            iv = int(v)
            if min_value is not None:
                iv = max(min_value, iv)
            if max_value is not None:
                iv = min(max_value, iv)
            return iv
        except Exception:
            return fallback

    def get_float(
        self,
        section: str,
        key: str,
        fallback: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Optional[float]:
        v = self.get_str(section, key)
        if v is None:
            return fallback
        try:
            fv = float(v)
            if min_value is not None:
                fv = max(min_value, fv)
            if max_value is not None:
                fv = min(max_value, fv)
            return fv
        except Exception:
            return fallback

    def get_enum(
        self,
        section: str,
        key: str,
        choices: Iterable[str],
        fallback: Optional[str] = None,
        auto_resolve: bool = False,
    ) -> Optional[str]:
        v = self.get_str(section, key)
        if v is None:
            return fallback
        if auto_resolve and _is_auto(v):
            if (section, key) == ("hardware", "gpu_backend"):
                pv = self.get_bool("hardware", "prefer_vulkan", fallback=False)
                return _detect_gpu_backend(bool(pv))
            if (section, key) == ("ai", "torch_device"):
                return _auto_torch_device()
            return fallback
        v2 = str(v).lower().strip()
        cset = {c.lower() for c in choices}
        return v if v2 in cset else fallback

    def get_path(
        self,
        section: str,
        key: str,
        fallback: Optional[Path] = None,
        auto_resolve: bool = False,
        must_exist: bool = False,
    ) -> Optional[Path]:
        v = self.get_str(section, key)
        if v is None:
            return fallback
        if auto_resolve and _is_auto(v):
            if (section, key) == ("paths", "ffmpeg_path"):
                p = _auto_ffmpeg_path()
                return Path(p) if p else fallback
            if (section, key) == ("paths", "aria2_path"):
                p = _auto_aria2_path()
                return Path(p) if p else fallback
        p = Path(_expand_path(v))
        if must_exist and not p.exists():
            return fallback
        return p

    # ─────────────────────────────── Setzen & Save ────────────────────────────
    def set_value(
        self,
        section: str,
        key: str,
        value: Union[str, int, float, bool, Path],
        target: str = "auto",
        save_immediately: bool = True,
    ) -> None:
        """
        Setzt den Wert im In-Memory-Config und schreibt ihn (kommentar-schonend)
        in die gewählte Datei:
            target='project' | 'user' | 'auto' (projekt falls vorhanden, sonst user)
        """
        # In-Memory
        if not self._cp.has_section(section):
            self._cp.add_section(section)
        self._cp.set(section, key, self._to_str(value))

        # Ziel wählen
        if target not in {"auto", "project", "user"}:
            raise ValueError("target must be 'auto'|'project'|'user'")
        path = None
        raw_text = None

        if target == "project":
            if not self.project_config_path:
                raise RuntimeError(
                    "project_root nicht gesetzt – 'project' nicht möglich."
                )
            path = self.project_config_path
            raw_text = self._raw_project_text
        elif target == "user":
            path = self.user_config_path
            raw_text = self._raw_user_text
        else:  # auto
            if self.project_config_path and self.project_config_path.exists():
                path = self.project_config_path
                raw_text = self._raw_project_text
            else:
                path = self.user_config_path
                raw_text = self._raw_user_text

        new_text = _set_value_linewise(
            raw_text or DEFAULT_TEMPLATE, section, key, self._to_str(value)
        )
        _atomic_write(path, new_text)
        _log_info(f"[config] Gespeichert: {path} → [{section}] {key} = {value}")

        # Local cache aktualisieren
        if path == self.user_config_path:
            self._raw_user_text = new_text
        else:
            self._raw_project_text = new_text

        if save_immediately:
            return
        # (Falls batch: Nutzer kann später save('user'/'project') aufrufen)

    def save(self, target: str) -> None:
        """Schreibt den aktuellen In-Memory-Stand als Ganzes in 'user' oder 'project' (ohne Kommentar-Erhalt!)."""
        if target not in {"user", "project"}:
            raise ValueError("target must be 'user' or 'project'")
        path = self.user_config_path if target == "user" else self.project_config_path
        if path is None:
            raise RuntimeError("project_root nicht gesetzt – 'project' nicht möglich.")

        # WARNUNG: Komplettschreiben entfernt ggf. Inline-Kommentare. Nur nutzen, wenn ok.
        from io import StringIO

        sio = StringIO()
        self._cp.write(sio)
        text = sio.getvalue()
        _atomic_write(path, text)
        _log_warn(f"[config] Vollschreibvorgang ohne Kommentar-Erhalt: {path}")

    # ─────────────────────────────── Nützliches ───────────────────────────────
    def where(self, section: str, key: str) -> Optional[str]:
        """Return 'env'|'project'|'user'|'default' (Quelle des Werts)"""
        return self._source.get((section.lower(), key.lower()))

    def resolved_ffmpeg(self) -> Optional[Path]:
        p = self.get_path("paths", "ffmpeg_path", auto_resolve=True)
        if p and p.exists():
            return p
        return None

    def resolved_aria2(self) -> Optional[Path]:
        p = self.get_path("paths", "aria2_path", auto_resolve=True)
        if p and p.exists():
            return p
        return None

    def resolved_threads(self) -> int:
        return (
            self.get_int(
                "hardware",
                "cpu_threads",
                auto_resolve=True,
                fallback=_cpu_threads_auto(),
            )
            or _cpu_threads_auto()
        )

    def resolved_gpu_backend(self) -> str:
        pv = self.get_bool("hardware", "prefer_vulkan", fallback=False)
        val = self.get_enum(
            "hardware",
            "gpu_backend",
            choices=["auto", "cuda", "rocm", "mps", "opencl", "vulkan", "none"],
            fallback="none",
            auto_resolve=True,
        )
        if val == "auto":
            return _detect_gpu_backend(bool(pv))
        return val or "none"

    def resolved_torch_device(self) -> str:
        return (
            self.get_enum(
                "ai",
                "torch_device",
                choices=["auto", "cuda", "cpu", "mps"],
                fallback="cpu",
                auto_resolve=True,
            )
            or "cpu"
        )

    # ────────────────────────────── Internals ─────────────────────────────────
    @staticmethod
    def _to_str(v: Union[str, int, float, bool, Path]) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, Path):
            return str(v)
        return str(v)

    def _detect_gpu_present(self) -> bool:
        return bool(
            _which("nvidia-smi")
            or _which("rocminfo")
            or platform.system().lower() == "darwin"
        )

    def _apply_env_overrides(self, cp: configparser.ConfigParser) -> None:
        # Schema: VIDEO_MANAGER__SECTION__KEY=VALUE  (Case-insensitive bei SECTION/KEY)
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(ENV_PREFIX):
                continue
            try:
                _, sec, key = env_key.split("__", 2)
            except ValueError:
                continue
            section, option = sec.lower(), key.lower()
            if not cp.has_section(section):
                cp.add_section(section)
            cp.set(section, option, env_val)
            self._source[(section, option)] = "env"
            _log_debug(f"[config] ENV override: [{section}] {option} = {env_val}")

    def _rebuild_source_map(
        self,
        dflt: configparser.ConfigParser,
        user: configparser.ConfigParser,
        proj: configparser.ConfigParser,
    ) -> None:
        self._source.clear()
        # Defaults vorbesetzen
        for s in dflt.sections():
            for k, _ in dflt.items(s):
                self._source[(s.lower(), k.lower())] = "default"
        # User überschreibt
        for s in user.sections():
            for k, _ in user.items(s):
                self._source[(s.lower(), k.lower())] = "user"
        # Projekt überschreibt
        for s in proj.sections():
            for k, _ in proj.items(s):
                self._source[(s.lower(), k.lower())] = "project"
        # ENV wurde bereits in _apply_env_overrides auf "env" gesetzt

    def _migrate_missing_keys(self) -> None:
        """
        Ergänzt fehlende Keys in vorhandenen Dateien (user/project) anhand DEFAULT_TEMPLATE,
        ohne bestehende Kommentare/Struktur zu zerstören.
        """
        default_cp = _parse_ini(self._raw_default_text)

        # User-Datei ergänzen
        if self.user_config_path:
            txt = self._raw_user_text or ""
            changed = False
            for sec in default_cp.sections():
                for k, v in default_cp.items(sec):
                    if not self._has_in_text(txt, sec, k):
                        txt2 = _set_value_linewise(txt or DEFAULT_TEMPLATE, sec, k, v)
                        if txt2 != txt:
                            txt = txt2
                            changed = True
            if changed:
                _atomic_write(self.user_config_path, txt)
                self._raw_user_text = txt
                _log_info(
                    f"[config] Migration: fehlende Keys in {self.user_config_path} ergänzt."
                )

        # Projekt-Datei ergänzen
        if self.project_config_path:
            txt = self._raw_project_text or ""
            if txt:  # nur migrieren, wenn Datei existiert
                changed = False
                for sec in default_cp.sections():
                    for k, v in default_cp.items(sec):
                        if not self._has_in_text(txt, sec, k):
                            txt2 = _set_value_linewise(txt, sec, k, v)
                            if txt2 != txt:
                                txt = txt2
                                changed = True
                if changed:
                    _atomic_write(self.project_config_path, txt)
                    self._raw_project_text = txt
                    _log_info(
                        f"[config] Migration: fehlende Keys in {self.project_config_path} ergänzt."
                    )

        # Falls gar keine Dateien existieren: User-Config mit Defaults anlegen
        if not (
            self.user_config_path.exists()
            or (self.project_config_path and self.project_config_path.exists())
        ):
            _atomic_write(self.user_config_path, DEFAULT_TEMPLATE)
            self._raw_user_text = DEFAULT_TEMPLATE
            _log_info(f"[config] Default erstellt: {self.user_config_path}")

    @staticmethod
    def _has_in_text(text: str, section: str, key: str) -> bool:
        if not text:
            return False
        s, e = _find_section_span(text, section)
        if s == -1:
            return False
        block = text[s:e]
        pat = re.compile(rf"^\s*{re.escape(key)}\s*=", re.MULTILINE)
        return pat.search(block) is not None
