#!/usr/bin/env python3
# vm_config.py – Konfigurations-Utilities für videoManager
from __future__ import annotations

import configparser
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional: sanftes Logging via consoleOutput (falls vorhanden)
try:
    import consoleOutput as co  # dein Projektmodul

    def _log(msg: str) -> None:
        getattr(co, "log_info", print)(msg)

    def _warn(msg: str) -> None:
        getattr(co, "log_warn", print)(msg)

    def _err(msg: str) -> None:
        getattr(co, "log_error", print)(msg)

except Exception:

    def _log(msg: str) -> None:
        print(msg)

    def _warn(msg: str) -> None:
        print(msg)

    def _err(msg: str) -> None:
        print(msg)


# ----------------------------- Standardpfade -----------------------------
def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))


DEFAULT_CONFIG_PATH = _xdg_config_home() / "videoManager" / "config.ini"
DEFAULT_LOG_PATH = (
    Path.home() / ".local" / "share" / "videoManager" / "videoManager.log"
)

# ----------------------------- Defaults/Schema ---------------------------
# Hinweis: Alle Werte sind Strings im INI, werden aber nach Typ geparst (bool,int,float).
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "general": {
        "language": "de",
        "os": "auto",
        "batch_mode_default": "false",
        "check_updates": "true",
    },
    "paths": {
        "default_output_dir": str(Path.home() / "Videos" / "videoManager"),
        "temp_dir": str(Path.home() / ".cache" / "videoManager" / "tmp"),
        "ffmpeg_path": "auto",
        "aria2_path": "auto",
    },
    "hardware": {
        "gpu_enabled": "auto",
        "gpu_backend": "auto",
        "cpu_threads": "auto",
        "prefer_vulkan": "false",
    },
    "ai": {
        "enabled": "true",
        "backend": "torch",
        "torch_device": "auto",
        "realesrgan_model": "realesr-general-x4v3",
        "realesrgan_strength": "0.5",
        "gfpgan_enabled": "false",
        "codeformer_enabled": "false",
    },
    "ffmpeg": {
        "loglevel": "error",
        "overwrite": "false",
        "hwaccel": "auto",
        "encoder_video": "libx264",
        "encoder_audio": "aac",
        "crf": "18",
        "preset": "medium",
    },
    "ui": {
        "unicode": "true",
        "colors": "true",
        "term_width": "auto",
    },
    "logging": {
        "level": "INFO",
        "file": str(DEFAULT_LOG_PATH),
        "rotate": "true",
        "keep_files": "5",
    },
    "network": {
        "download_retries": "5",
        "timeout_sec": "30",
    },
    "experimental": {
        "ncnn_bin_dir": "",
        "onnx_runtime_enabled": "false",
    },
    "safety": {
        "dry_run": "false",
    },
}

# Typ-Schema für robustes Parsen/Validieren
TYPES: Dict[Tuple[str, str], Any] = {
    # general
    ("general", "language"): str,
    ("general", "os"): str,
    ("general", "batch_mode_default"): bool,
    ("general", "check_updates"): bool,
    # paths
    ("paths", "default_output_dir"): str,
    ("paths", "temp_dir"): str,
    ("paths", "ffmpeg_path"): str,
    ("paths", "aria2_path"): str,
    # hardware
    ("hardware", "gpu_enabled"): str,  # auto|true|false
    ("hardware", "gpu_backend"): str,  # auto|cuda|rocm|mps|opencl|vulkan|none
    ("hardware", "cpu_threads"): str,  # auto|int
    ("hardware", "prefer_vulkan"): bool,
    # ai
    ("ai", "enabled"): bool,
    ("ai", "backend"): str,  # torch|ncnn|onnx
    ("ai", "torch_device"): str,  # auto|cuda|cpu|mps
    ("ai", "realesrgan_model"): str,
    ("ai", "realesrgan_strength"): float,
    ("ai", "gfpgan_enabled"): bool,
    ("ai", "codeformer_enabled"): bool,
    # ffmpeg
    ("ffmpeg", "loglevel"): str,
    ("ffmpeg", "overwrite"): bool,
    ("ffmpeg", "hwaccel"): str,
    ("ffmpeg", "encoder_video"): str,
    ("ffmpeg", "encoder_audio"): str,
    ("ffmpeg", "crf"): int,
    ("ffmpeg", "preset"): str,
    # ui
    ("ui", "unicode"): bool,
    ("ui", "colors"): bool,
    ("ui", "term_width"): str,  # auto|int
    # logging
    ("logging", "level"): str,
    ("logging", "file"): str,
    ("logging", "rotate"): bool,
    ("logging", "keep_files"): int,
    # network
    ("network", "download_retries"): int,
    ("network", "timeout_sec"): int,
    # experimental
    ("experimental", "ncnn_bin_dir"): str,
    ("experimental", "onnx_runtime_enabled"): bool,
    # safety
    ("safety", "dry_run"): bool,
}


# ----------------------------- Utilities ---------------------------------
def _str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_value(section: str, key: str, raw: str) -> Any:
    t = TYPES.get((section, key), str)
    if t is bool:
        return _str_to_bool(raw)
    if t is int:
        try:
            return int(str(raw).strip())
        except Exception:
            return int(str(DEFAULTS[section][key]))
    if t is float:
        try:
            return float(str(raw).strip())
        except Exception:
            return float(str(DEFAULTS[section][key]))
    # strings: gebe 1:1 zurück
    return str(raw)


def _ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _which(cmd: str) -> Optional[str]:
    try:
        p = shutil.which(cmd)
        return p if p else None
    except Exception:
        return None


# -------------------------- Laden/Speichern ------------------------------
def write_default_config(path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Erzeugt eine Default-Konfigurationsdatei, falls nicht vorhanden."""
    if path.exists():
        return
    _ensure_parents(path)
    cp = configparser.ConfigParser(interpolation=None)
    for sec, kv in DEFAULTS.items():
        cp[sec] = {}
        for k, v in kv.items():
            cp[sec][k] = str(v)
    # atomisch schreiben
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        cp.write(f)
    tmp.replace(path)
    _log(f"[config] Default geschrieben → {path}")


def load_config(
    path: Optional[Path] = None, create_if_missing: bool = True
) -> configparser.ConfigParser:
    """Lädt die INI, merged fehlende Defaults und gibt ConfigParser zurück."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if create_if_missing and not path.exists():
        write_default_config(path)

    cp = configparser.ConfigParser(interpolation=None)
    # Defaults vorinitialisieren
    for sec, kv in DEFAULTS.items():
        if sec not in cp:
            cp.add_section(sec)
        for k, v in kv.items():
            cp[sec][k] = str(v)

    # Dateiwerte darüber legen
    if path.exists():
        cp.read(path, encoding="utf-8")

    # sicherstellen, dass alle Sektionen/Keys existieren
    changed = False
    for sec, kv in DEFAULTS.items():
        if not cp.has_section(sec):
            cp.add_section(sec)
            changed = True
        for k, default_v in kv.items():
            if not cp.has_option(sec, k):
                cp.set(sec, k, str(default_v))
                changed = True

    if changed:
        save_config(cp, path)  # migriert fehlende Keys in Datei
    return cp


def save_config(cp: configparser.ConfigParser, path: Optional[Path] = None) -> None:
    """Speichert atomisch."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    _ensure_parents(path)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        cp.write(f)
    tmp.replace(path)
    _log(f"[config] Gespeichert → {path}")


# ----------------------- High-Level Get/Set ------------------------------
def get_value(
    cp: configparser.ConfigParser, dotted_key: str, default: Any = None
) -> Any:
    """
    Liest einen Wert per dotted key, z. B. 'ai.backend'.
    Gibt nach Typ-Schema geparsten Wert zurück (bool/int/float/str).
    """
    if "." not in dotted_key:
        raise ValueError("dotted_key erwartet das Format 'section.key'")
    section, key = dotted_key.split(".", 1)
    if not cp.has_section(section) or not cp.has_option(section, key):
        return default
    raw = cp.get(section, key, fallback=None)
    if raw is None:
        return default
    return _parse_value(section, key, raw)


def set_value(cp: configparser.ConfigParser, dotted_key: str, value: Any) -> None:
    """
    Schreibt einen Wert (automatisch in String gewandelt). Legt Sektion an, falls fehlend.
    """
    if "." not in dotted_key:
        raise ValueError("dotted_key erwartet das Format 'section.key'")
    section, key = dotted_key.split(".", 1)
    if not cp.has_section(section):
        cp.add_section(section)
    cp.set(section, key, str(value))


def to_dict(
    cp: configparser.ConfigParser, parsed_types: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Gibt die gesamte Config als Dict zurück (optional typ-geparst)."""
    out: Dict[str, Dict[str, Any]] = {}
    for sec in cp.sections():
        out[sec] = {}
        for key, raw in cp.items(sec):
            out[sec][key] = _parse_value(sec, key, raw) if parsed_types else raw
    return out


# --------------------- „Effective“ Konfiguration -------------------------
def _detect_os_name() -> str:
    sysname = platform.system().lower()
    if "linux" in sysname:
        return "linux"
    if "darwin" in sysname:
        return "mac"
    if "windows" in sysname or "msys" in sysname or "cygwin" in sysname:
        return "windows"
    return sysname or "linux"


def _detect_ffmpeg() -> Optional[str]:
    return _which("ffmpeg")


def _detect_aria2() -> Optional[str]:
    return _which("aria2c")


def _detect_gpu_backend() -> Tuple[bool, str]:
    """
    Versucht GPU-Verfügbarkeit leichtgewichtig zu schätzen.
    Kein hartes torch-Import, um Startup zu beschleunigen.
    """
    # NVIDIA?
    if _which("nvidia-smi"):
        return True, "cuda"
    # Apple M-Serie?
    if _detect_os_name() == "mac":
        # MPS ≈ Metal Performance Shaders (falls torch installiert)
        return True, "mps"
    # Intel/AMD via VA/Vulkan/OpenCL – hier konservativ:
    return False, "none"


def _resolve_auto_values(eff: Dict[str, Dict[str, Any]]) -> None:
    # OS
    if eff["general"]["os"] == "auto":
        eff["general"]["os"] = _detect_os_name()
    # ffmpeg / aria2
    if eff["paths"]["ffmpeg_path"] == "auto":
        eff["paths"]["ffmpeg_path"] = _detect_ffmpeg() or "ffmpeg"
    if eff["paths"]["aria2_path"] == "auto":
        eff["paths"]["aria2_path"] = _detect_aria2() or "aria2c"
    # Hardware
    gpu_enabled_cfg = str(eff["hardware"]["gpu_enabled"]).lower()
    has_gpu, backend_guess = _detect_gpu_backend()
    if gpu_enabled_cfg == "auto":
        eff["hardware"]["gpu_enabled"] = has_gpu
    else:
        eff["hardware"]["gpu_enabled"] = gpu_enabled_cfg == "true"

    if eff["hardware"]["gpu_backend"] == "auto":
        eff["hardware"]["gpu_backend"] = backend_guess

    # CPU Threads
    if eff["hardware"]["cpu_threads"] == "auto":
        try:
            eff["hardware"]["cpu_threads"] = os.cpu_count() or 4
        except Exception:
            eff["hardware"]["cpu_threads"] = 4

    # AI torch_device
    if eff["ai"]["torch_device"] == "auto":
        if eff["hardware"]["gpu_enabled"] and eff["hardware"]["gpu_backend"] == "cuda":
            eff["ai"]["torch_device"] = "cuda"
        elif eff["general"]["os"] == "mac":
            eff["ai"]["torch_device"] = "mps"
        else:
            eff["ai"]["torch_device"] = "cpu"

    # UI term_width
    if eff["ui"]["term_width"] == "auto":
        try:
            import shutil as _sh

            eff["ui"]["term_width"] = _sh.get_terminal_size().columns
        except Exception:
            eff["ui"]["term_width"] = 80


def effective_config(cp: configparser.ConfigParser) -> Dict[str, Dict[str, Any]]:
    """
    Gibt eine aufgelöste Runtime-Config (mit 'auto' → konkreter Wert) zurück,
    ohne die Datei zu verändern.
    """
    eff = to_dict(cp, parsed_types=True)
    # Strings normalisieren (ini liefert Strings; to_dict parst bereits Typen)
    _resolve_auto_values(eff)
    return eff


# -------------------- Bequeme Convenience-Wrapper ------------------------
def get_lang(cp: configparser.ConfigParser) -> str:
    return str(get_value(cp, "general.language", "de"))


def set_lang(cp: configparser.ConfigParser, lang: str) -> None:
    set_value(cp, "general.language", lang)


def is_gpu_enabled(cp: configparser.ConfigParser) -> bool:
    eff = effective_config(cp)
    return bool(eff["hardware"]["gpu_enabled"])


def set_gpu_enabled(cp: configparser.ConfigParser, enabled: bool) -> None:
    set_value(cp, "hardware.gpu_enabled", str(enabled).lower())


def get_ffmpeg_path(cp: configparser.ConfigParser) -> str:
    eff = effective_config(cp)
    return str(eff["paths"]["ffmpeg_path"])


def set_realesrgan_model(cp: configparser.ConfigParser, model: str) -> None:
    set_value(cp, "ai.realesrgan_model", model)


# -------------------------- Mini-Demo/CLI (optional) ---------------------
if __name__ == "__main__":
    """
    Beispiele:
      python vm_config.py show
      python vm_config.py get ai.backend
      python vm_config.py set ffmpeg.overwrite true
      python vm_config.py effective   # zeigt aufgelöste Werte (auto→konkret)
    """
    import argparse

    ap = argparse.ArgumentParser(description="videoManager Config Helper")
    ap.add_argument("cmd", choices=["show", "get", "set", "effective"], help="Aktion")
    ap.add_argument("key", nargs="?", help="dotted key (z.B. ai.backend)")
    ap.add_argument("value", nargs="?", help="neuer Wert für 'set'")
    ap.add_argument(
        "--path", help="Pfad zur config.ini", default=str(DEFAULT_CONFIG_PATH)
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.path), create_if_missing=True)

    if args.cmd == "show":
        print(json.dumps(to_dict(cfg, parsed_types=True), indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.cmd == "effective":
        print(json.dumps(effective_config(cfg), indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.cmd == "get":
        if not args.key:
            ap.error("Bitte 'key' angeben.")
        print(get_value(cfg, args.key))
        sys.exit(0)

    if args.cmd == "set":
        if not args.key or args.value is None:
            ap.error("Bitte 'key' und 'value' angeben.")
        set_value(cfg, args.key, args.value)
        save_config(cfg, Path(args.path))
        print(f"{args.key} = {args.value}")
        sys.exit(0)
