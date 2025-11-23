#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import psutil  # präziser Freiplatz-Check

import consoleOutput as co
import definitions as defin
import userInteraction as ui

# local
from i18n import _

GB: int = 1024**3


# ─── TMP_ROOT ermitteln mit Fallbacks ───────────────────────────────
def select_tmp_root(script_dir: Path) -> Path:
    """
    Wählt einen geeigneten temporären Wurzelordner nach Priorität:
    1) Linux tmpfs (/dev/shm/video_sr)
    2) System-Temp (…/video_sr)
    3) <skriptverzeichnis>/tmp

    Es wird der erste Ordner mit mindestens 4 GB freiem Speicher gewählt.
    """
    candidates: List[Path] = []
    if sys.platform.startswith("linux"):
        candidates.append(Path("/dev/shm") / "video_sr")
    candidates.append(Path(tempfile.gettempdir()) / "video_sr")
    candidates.append(script_dir / "tmp")

    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            du = shutil.disk_usage(str(d))
            # mind. 4 GB frei
            if du.free / GB >= 4.0:
                return d
        except Exception:
            # nächster Kandidat
            pass
    raise RuntimeError(_("tmp_no_suitable_dir"))


# Freiplatz-Check
def has_free_space(
    directory: Union[str, Path], min_gb: float = 5.0
) -> Tuple[bool, float]:
    """Gibt (ok, freie_GB) zurück."""
    du = psutil.disk_usage(str(directory))
    free_gb = du.free / GB
    return free_gb >= min_gb, free_gb


def build_output_path(
    input_path: Path,
    output_arg: Optional[str],
    default_suffix: str,
    idx: Optional[int] = None,  # 1-basiert
    total: Optional[int] = None,
    target_ext: Optional[str] = None,  # z. B. ".gif"; None => Originalendung behalten
) -> Path:
    """
    Erzeugt einen robusten Ausgabepfad für Einzel- und Batch-Verarbeitung.

    Regeln:
      • output_arg = None
          – Wenn Format gleich bleibt: <stem><default_suffix><ext>
          – Wenn Format sich ändert:  <stem><target_ext> (ohne suffix)
      • output_arg ist Verzeichnis  -> <dir>/<stem><default_suffix><ext>
      • output_arg enthält {...}    -> Template-Render
      • mehrere Inputs + output_arg ist Datei -> suffix vor ext um _<i> erweitern
      • target_ext überschreibt die Ausgabesuffix (inkl. führendem '.')
    Platzhalter:
      {stem}, {ext}, {name}, {parent}, {i}, {n}
    """
    in_path = Path(input_path)
    in_ext = in_path.suffix  # mit Punkt
    ext_no_dot = in_ext[1:] if in_ext.startswith(".") else in_ext

    # Ziel-Extension normalisieren
    if target_ext:
        use_ext = target_ext if target_ext.startswith(".") else f".{target_ext}"
    else:
        use_ext = in_ext

    # 1) Kein --output -> neue Logik
    if not output_arg:
        same_format = use_ext.lower() == in_ext.lower()
        if same_format:
            return in_path.with_stem(in_path.stem + default_suffix).with_suffix(use_ext)
        return in_path.with_suffix(use_ext)

    out_path = Path(output_arg)

    # 2) Als Verzeichnis behandeln?
    looks_like_dir = (
        out_path.is_dir()
        or output_arg.endswith(("/", "\\"))
        or (total and total > 1 and out_path.suffix == "")
    )
    if looks_like_dir:
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / (in_path.stem + default_suffix)).with_suffix(use_ext)

    # 3) Template?
    if "{" in output_arg and "}" in output_arg:
        tokens: Dict[str, Union[str, int]] = {
            "stem": in_path.stem,
            "ext": ext_no_dot,
            "name": in_path.name,
            "parent": in_path.parent.name,
            "i": idx if idx is not None else 1,
            "n": total if total is not None else 1,
        }
        rendered = output_arg.format(**tokens)
        p = Path(rendered)
        if p.suffix == "":
            p = p.with_suffix(use_ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 4) Fester Dateiname bei Batch
    if total and total > 1:
        stem_with_idx = f"{out_path.stem}_{idx if idx is not None else 1}"
        p = out_path.with_stem(stem_with_idx).with_suffix(
            use_ext if use_ext else (out_path.suffix or in_ext)
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 5) Single-Input + fester Dateiname
    p2 = out_path
    if p2.suffix == "":
        p2 = p2.with_suffix(use_ext)
    p2.parent.mkdir(parents=True, exist_ok=True)
    return p2


def save_file(file: Any, path: Path, format: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format:
        file.save(path, format=format)
    else:
        file.save(path)


def write_json(p: Path, obj: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        co.print_warning(_("persist_write_json_failed").format(error=e))


def parse_file_selection(input_str: str, max_index: int) -> List[int]:
    """
    Interpretiert eine Nutzer-Eingabe wie "1 2 5-8 +" als Indexliste (0-basiert).
    ENTER => alle, "0" => Programmende.

    Hinweis: Ranges sind wie bisher [start..end) (rechts exklusiv), um bestehendes
    Verhalten nicht zu ändern.
    """
    if not input_str.strip():  # ENTER = alle
        return list(range(max_index))
    if input_str.strip() == "0":
        sys.exit(0)

    result: List[int] = []
    parts = re.split(r"[\s,]+", input_str.strip())
    last = -1
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                result.extend(range(int(start) - 1, int(end)))
                last = int(end) - 1
        elif part == "+":
            if last >= 0:
                result.extend(range(last, max_index))
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < max_index:
                result.append(idx)
                last = idx
    # de-dupe, Reihenfolge stabil halten
    seen: set[int] = set()
    uniq: List[int] = []
    for i in result:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


# ---------- utils ----------


def _normalize_cli_files(value: Any) -> List[str]:
    """CLI-Argument 'files' robust in eine String-Liste überführen."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    try:
        return [str(x) for x in value]
    except TypeError:
        return [str(value)]


def expand_glob_patterns(patterns: Iterable[str]) -> List[str]:
    """
    Expandiert Dateimuster in 'patterns'.
    Unterstützt:
      • Shell-Globs (*, ?, [], **)
      • Prozent-Platzhalter (%) → wird zu '*'
    Literale Pfade (ohne Globs) werden unverändert übernommen.
    """
    out: List[str] = []
    for pat_in in patterns:
        pat = str(pat_in).replace("%", "*")
        pat = str(Path(pat).expanduser())
        if any(ch in pat for ch in "*?["):
            out.extend(glob.glob(pat, recursive=True))
        else:
            out.append(pat)  # wörtlicher Pfad
    return out


def prepare_files(args: Any, *extension_groups: Sequence[str]) -> List[str]:
    """
    Finale Eingabedateien als String-Pfade.
    - CLI: akzeptiert Dateien, Globs und ORDNER.
      → Ordner werden zunächst flach (nicht rekursiv) expandiert.
      → Falls dabei nichts gefunden wird, erfolgt ein *einmaliger* rekursiver Fallback.
      → `args.recursive=True` erzwingt sofort rekursive Suche.
    - Interaktiv unverändert.
    """

    def _flatten_exts(groups: Sequence[Sequence[str]]) -> set[str]:
        # Normalisiert: lower + mit führendem Punkt
        def _norm(e: str) -> str:
            e = e.strip().lower()
            return e if e.startswith(".") else f".{e}"

        if not groups:
            try:
                return {_norm(e) for e in getattr(defin, "VIDEO_EXTENSIONS", ())}
            except Exception:
                return set()
        return {_norm(e) for g in groups for e in g}

    def _natural_sort_dedup(paths: List[str]) -> List[str]:
        try:
            import re as _re

            def _natkey(s: str):
                return [
                    int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)
                ]

            return sorted(dict.fromkeys(paths), key=_natkey)
        except Exception:
            return list(dict.fromkeys(sorted(paths)))

    def _expand_dirs(
        paths: List[str], valid_exts: set[str], recursive: bool
    ) -> List[str]:
        out: List[str] = []
        for p in paths:
            P = Path(p).expanduser()
            if P.is_dir():
                itr = P.rglob("*") if recursive else P.iterdir()
                for child in itr:
                    try:
                        if child.is_file():
                            if not valid_exts or child.suffix.lower() in valid_exts:
                                out.append(str(child))
                    except Exception:
                        pass
            elif P.is_file():
                if (not valid_exts) or (Path(P).suffix.lower() in valid_exts):
                    out.append(str(P))
        return _natural_sort_dedup(out)

    valid_exts = _flatten_exts(extension_groups)
    provided = _normalize_cli_files(getattr(args, "files", None))

    # CLI-Pfad
    if provided:
        # Globs & Literale expandieren
        expanded = expand_glob_patterns(provided)
        if not expanded:
            # i18n-Key mit Platzhalter {pattern}
            for pat in provided:
                co.print_error(_("no_files_match_pattern").format(pattern=pat))
            args._files_source = "cli"
            return []

        existing = [
            str(Path(f).expanduser()) for f in expanded if Path(f).expanduser().exists()
        ]
        if not existing:
            # vorhandenen Key "file_not_found" verwenden
            for f in expanded:
                co.print_error(_("file_not_found").format(file=f))
            args._files_source = "cli"
            return []

        # 1) flach expandieren (Ordner → direkte Kinder)
        want_recursive = bool(getattr(args, "recursive", False))
        files_flat = (
            _expand_dirs(existing, valid_exts, recursive=False)
            if not want_recursive
            else []
        )
        if want_recursive:
            files_rec = _expand_dirs(existing, valid_exts, recursive=True)
            result = files_rec
        else:
            # 2) Fallback: nichts gefunden? → einmal rekursiv
            if not files_flat:
                files_rec = _expand_dirs(existing, valid_exts, recursive=True)
                result = files_rec
            else:
                result = files_flat

        if not result:
            co.print_error(_("no_matching_file_found"))
            args._files_source = "cli"
            return []

        args._files_source = "cli"
        return result

    # ─────────────────────────────────────────────────────────────
    # Flags benutzt, aber keine Dateien angegeben → Fehler
    # ─────────────────────────────────────────────────────────────
    cli_flags_used = any(isinstance(a, str) and a.startswith("-") for a in sys.argv[1:])
    if cli_flags_used:
        args._files_source = "cli"
        co.print_error(_("passed_no_file"))
        return []

    # Interaktive Auswahl (bestehend)
    files_sel = ui.select_files_interactively(*extension_groups)
    args._files_source = "interactive"

    final: List[str] = []
    seen: set[str] = set()
    for f in files_sel:
        p = Path(f).expanduser()
        if p.exists():
            sp = str(p)
            if sp not in seen:
                # Optional: interaktive Auswahl kann Ordner liefern → flach/rekursiv behandeln?
                if p.is_dir():
                    chosen = _expand_dirs([sp], valid_exts, recursive=False)
                    if not chosen:
                        chosen = _expand_dirs([sp], valid_exts, recursive=True)
                    final.extend(chosen)
                else:
                    final.append(sp)
                seen.add(sp)
    return _natural_sort_dedup(final)


def open_file_crossplatform(p: Path) -> None:
    """Öffnet eine Datei plattformabhängig mit dem Standardprogramm."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            opener = shutil.which("xdg-open")
            if opener:
                subprocess.run([opener, str(p)], check=False)
    except Exception:
        # still bleiben – Öffnen ist optional
        pass
