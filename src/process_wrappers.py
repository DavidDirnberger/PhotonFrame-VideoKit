#!/usr/bin/env python3
# process_wrappers.py
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import (
    List,
    Union,
)

import consoleOutput as co
import definitions as defin
import graphic_helpers as gh
import helpers as he

# local modules
import userInteraction as ui
from ANSIColor import ANSIColor
from i18n import _
from mem_guard import (
    CANCEL,
    cleanup_guarded_paths_now,
    free_ram_vram,
    is_cancelled,
    kill_all,
)


class FFmpegCancelled(Exception):
    """Vom Benutzer abgebrochen (ESC/Strg+C)."""


class FFmpegFailed(Exception):
    """ffmpeg ist fehlgeschlagen (Exit-Code != 0 oder Output fehlt)."""


c = ANSIColor()

# Nur unter POSIX TTY-Funktionen nutzen
if os.name == "posix":
    pass

# ---------------------------------------------------------------------------
# Helpers (zweizeilige/mehrzeilige Progress-UI)
# ---------------------------------------------------------------------------


def _collect_known_output_suffixes() -> set[str]:
    """Sammelt bekannte Suffixe aus defin.* und normalisiert auf '.ext'."""

    def dot(x: str) -> str:
        x = (x or "").lower().strip()
        return x if x.startswith(".") else f".{x}"

    suffixes: set[str] = set()

    # Video/Audio direkt übernehmen
    for x in getattr(defin, "VIDEO_EXTENSIONS", []):
        suffixes.add(dot(str(x)))
    for x in getattr(defin, "AUDIO_EXTENSIONS", []):
        suffixes.add(dot(str(x)))

    # Bild/Untertitel
    for x in getattr(defin, "IMAGE_EXTENSIONS", []):
        suffixes.add(dot(str(x)))
    for x in getattr(defin, "SUB_EXTENSIONS", []):
        suffixes.add(dot(str(x)))

    # Alles, was EXT_TO_CONTAINER kennt
    for k in getattr(defin, "EXT_TO_CONTAINER", {}).keys():
        suffixes.add(dot(str(k)))

    # Optional: reine FORMATS auch als Suffixe akzeptieren (mp4, mkv, …)
    for f in getattr(defin, "FORMATS", []):
        suffixes.add(dot(str(f)))

    return suffixes


# Lazy Cache
_KNOWN_SUFFIXES: set[str] | None = None


def _known_suffixes() -> set[str]:
    global _KNOWN_SUFFIXES
    if _KNOWN_SUFFIXES is None:
        _KNOWN_SUFFIXES = _collect_known_output_suffixes()
    return _KNOWN_SUFFIXES


def _find_output_arg_index(argv: list[str]) -> int:
    """
    Suche von hinten das plausibelste Output-Argument:
    - kein Flag
    - hat bekanntes Suffix ODER sieht nach Pfad aus
    Fallback: letztes Token.
    """
    suffixes = _known_suffixes()
    for idx in range(len(argv) - 1, -1, -1):
        tok = str(argv[idx])
        if tok.startswith("-"):
            continue
        p = Path(tok)
        if p.suffix.lower() in suffixes:
            return idx
        if (os.sep in tok) or (os.altsep and os.altsep in tok):
            return idx
    return len(argv) - 1


# ---------------------------------------------------------------------------
# ffmpeg mit Fortschritt (Zeit/Frame-Parsing aus stderr) + ESC-Abbruch
# ---------------------------------------------------------------------------

# tcgetattr/tcsetattr arbeiten mit [int, int, int, int, int, int, list]
TermiosAttr = List[Union[int, List[int]]]


def run_ffmpeg_with_progress(
    input_file: str | Path,
    ffmpeg_cmd: list[str],
    progress_line: str,
    finished_line: str,
    output_file: str | Path | None = None,
    BATCH_MODE: bool = False,
    force_overwrite: bool = False,
    total_duration: float | None = None,
    bar_len_default: int = 80,
    min_bar: int = 20,
    *,
    mode: int = 0,
    total_frames: int | None = None,
    fps_hint: float | None = None,
    analysis_mode: bool = False,
) -> Path | None | int:
    """
    Fortschritt bevorzugt über Frames (wenn total_frames bekannt), sonst über Zeit (out_time/out_time_ms),
    sonst indeterminate Bar. Kein ffmpeg-Stdout im Terminal.
    ESC, 'q' oder Strg+C brechen ffmpeg sauber ab; TTY wird garantiert restauriert.
    """

    # mode == 0 Standartmode mit Balkenausgabe
    # mode == 1 Nur Rückgabe von integer => 0 = Erfolg | 0 < Fail
    # mode == 2 Silent

    input_path = Path(input_file)

    # Bei Analyse: niemals überschreiben & nicht nachfragen
    if analysis_mode:
        if "-nostdin" not in ffmpeg_cmd:
            ffmpeg_cmd.insert(1, "-nostdin")
        if "-n" not in ffmpeg_cmd:
            ffmpeg_cmd.insert(1, "-n")

    # --- kleine Helfer -----------------------------------------------------
    def _infer_fps_from_cmd(cmd: list[str]) -> float | None:
        for i, tok in enumerate(cmd):
            if str(tok) == "-r" and i + 1 < len(cmd):
                try:
                    return float(str(cmd[i + 1]))
                except Exception:
                    pass
        vf_val = None
        for i, tok in enumerate(cmd):
            if str(tok) in ("-vf", "-filter:v") and i + 1 < len(cmd):
                vf_val = str(cmd[i + 1])
        if vf_val:
            m = re.search(r"fps=([0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+)", vf_val)
            if m:
                v = m.group(1)
                if "/" in v:
                    a, b = v.split("/", 1)
                    try:
                        return float(a) / float(b)
                    except Exception:
                        return None
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    def _try_concat_duration(cmd: list[str]) -> float | None:
        try:
            is_concat = False
            concat_path: Path | None = None
            for i, tok in enumerate(cmd):
                if (
                    tok == "-f"
                    and i + 1 < len(cmd)
                    and str(cmd[i + 1]).lower() == "concat"
                ):
                    is_concat = True
                if tok == "-i" and i + 1 < len(cmd) and is_concat:
                    concat_path = Path(cmd[i + 1])
                    break
            if not concat_path or not concat_path.exists():
                return None
            total = 0.0
            with concat_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln.lower().startswith("duration"):
                        parts = ln.split()
                        if len(parts) >= 2:
                            try:
                                total += float(parts[1])
                            except Exception:
                                pass
            return total or None
        except Exception:
            return None

    def truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    def split_progress_template(
        tmpl: str, infile: Path, outfile: Path
    ) -> tuple[list[str], int | None]:
        if "\n" in tmpl:
            lines = ["   " + s for s in tmpl.splitlines()]
            lines = [
                s.replace("#ifilename", infile.name).replace("#ofilename", outfile.name)
                for s in lines
            ]
            return lines, None
        if "#ifilename" in tmpl:
            pre_if, post_if = tmpl.split("#ifilename", 1)
            return ["   " + pre_if, " '" + infile.name + "' ", "  " + post_if], 1
        if "#ofilename" in tmpl:
            pre_if, post_if = tmpl.split("#ofilename", 1)
            return ["   " + pre_if, " '" + outfile.name + "' ", "  " + post_if], 1
        return ["   " + tmpl.replace("#ofilename", outfile.name)], None

    def split_finished_template(
        tmpl: str, infile: Path, outfile: Path
    ) -> tuple[list[str], int | None]:
        if "\n" in tmpl:
            lines = ["   " + s for s in tmpl.splitlines()]
            lines = [
                s.replace("#ofilename", outfile.name).replace("#ifilename", infile.name)
                for s in lines
            ]
            return lines, None
        if "#ofilename" in tmpl:
            pre, post = tmpl.split("#ofilename", 1)
            return ["   " + pre, " '" + outfile.name + "' ", "  " + post], 1
        if "#ifilename" in tmpl:
            pre, post = tmpl.split("#ifilename", 1)
            return ["   " + pre, " '" + infile.name + "' ", "  " + post], 1
        s = "   " + tmpl.replace("#ofilename", outfile.name).replace(
            "#ifilename", infile.name
        )
        return [s], None

    def colorize_lines(
        raw_lines: list[str], color_code: str, italic_idx: int | None
    ) -> list[str]:
        term_cols = shutil.get_terminal_size((80, 20)).columns
        out: list[str] = []
        for i, s in enumerate(raw_lines):
            colored = (
                f"{color_code}{c.get('italic')}{s}{c.get('reset')}"
                if italic_idx is not None and i == italic_idx
                else f"{color_code}{s}{c.get('reset')}"
            )
            out.append(truncate(colored, term_cols))
        return out

    def _find_output_arg_index_local(argv: list[str]) -> int:
        return _find_output_arg_index(argv)

    # --- Output-Path & Overwrite (robust, Pipe/Null beachten) --------------
    output_arg_index = _find_output_arg_index_local(ffmpeg_cmd)
    last_arg = str(ffmpeg_cmd[output_arg_index]) if ffmpeg_cmd else ""
    pipe_like = (last_arg == "-") or last_arg.startswith("pipe:")
    has_real_output = (
        False if analysis_mode else ((output_file is not None) or (not pipe_like))
    )

    if analysis_mode:
        output_path = input_path
    elif has_real_output:
        output_path = Path(output_file) if output_file is not None else Path(last_arg)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        while output_path.exists() and not force_overwrite:
            if BATCH_MODE:
                if "-y" not in ffmpeg_cmd:
                    ffmpeg_cmd.insert(output_arg_index, "-y")
                break
            outstr = f"'{output_path.name}'\n    " + _("overwrite_file")
            options = [_("yes"), _("no"), _("change_filename")]
            ans = ui.ask_user(outstr, options=options, back_button=False)  # type: ignore[call-arg]
            if ans == options[0]:
                if "-y" not in ffmpeg_cmd:
                    ffmpeg_cmd.insert(output_arg_index, "-y")
                break
            elif ans == options[1]:
                co.print_line("   ➜ " + _("skip"))
                return 0 if mode == 1 else None
            else:
                new_name = ui.ask_for_filename(str(output_path))
                new_path = Path(new_name)
                if not new_path.is_absolute():
                    new_path = output_path.with_name(new_path.name)
                if new_path.suffix == "" and output_path.suffix:
                    new_path = new_path.with_suffix(output_path.suffix)
                output_path = new_path
                ffmpeg_cmd[output_arg_index] = str(new_path)
                continue
    else:
        output_path = input_path

    # Return only code mode
    if mode > 0:
        try:
            proc = subprocess.run(
                ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return int(proc.returncode) if mode == 1 else None
        except Exception:
            return 1 if mode == 1 else None

    # --- Sicheren Progress-Modus aktivieren (key=value über pipe:2) --------
    use_progress_kv = True
    if "-progress" not in ffmpeg_cmd:
        # alte -stats/-stats_period entfernen, um doppelten Output zu vermeiden
        try:
            while "-stats_period" in ffmpeg_cmd:
                i = ffmpeg_cmd.index("-stats_period")
                del ffmpeg_cmd[i : i + 2]
        except Exception:
            pass
        try:
            while "-stats" in ffmpeg_cmd:
                ffmpeg_cmd.remove("-stats")
        except Exception:
            pass
        if "-nostats" not in ffmpeg_cmd:
            ffmpeg_cmd.insert(1, "-nostats")
        ffmpeg_cmd.insert(1, "pipe:2")
        ffmpeg_cmd.insert(1, "-progress")
    else:
        try:
            pi = ffmpeg_cmd.index("-progress")
            tgt = (ffmpeg_cmd[pi + 1] if pi + 1 < len(ffmpeg_cmd) else "") or ""
            use_progress_kv = tgt.startswith("pipe:") or tgt == "-"
        except Exception:
            use_progress_kv = True

    print("")
    print("")

    # --- UI vorbereiten ----------------------------------------------------
    term_cols = shutil.get_terminal_size((80, 20)).columns
    prog_lines_raw, prog_italic_idx = split_progress_template(
        progress_line, input_path, output_path
    )
    hint_txt = (
        _("cancel_hint")
        if hasattr(_, "__call__")
        and "cancel_hint" in getattr(defin, "MESSAGES", {}).get("de", {})
        else "Abbrechen mit ESC • Strg+C/Strg+K"
    )
    hint_line = f"{c.get('dim')}{hint_txt}{c.get('reset')}"
    reserve_lines = len(prog_lines_raw) + 2
    print("\n" * (reserve_lines - 1), end="", flush=True)

    static_len = len(" 100% []")
    bar_len = max(min_bar, min(bar_len_default, term_cols - static_len))

    def _fail_cleanup(remove_partial: bool = True) -> None:
        # (Teil-)Output aufräumen
        try:
            if (
                remove_partial
                and has_real_output
                and output_path
                and output_path.exists()
            ):
                try:
                    output_path.unlink(missing_ok=True)  # Py3.8: in try/except umsetzen
                except TypeError:
                    try:
                        output_path.unlink()
                    except FileNotFoundError:
                        pass
        except Exception:
            pass
        # registrierte Temp-Pfade löschen
        try:
            cleanup_guarded_paths_now()
        except Exception:
            pass
        # laufende Kindprozesse (aus mem_guard.popen) killen
        try:
            kill_all()
        except Exception:
            pass
        # RAM/VRAM freigeben (ohne CUDA neu zu initialisieren)
        try:
            free_ram_vram(only_if_initialized=True)
        except Exception:
            pass

    # --- Dauer/FPS bestimmen ----------------------------------------------
    fps = fps_hint or _infer_fps_from_cmd(ffmpeg_cmd)
    duration = (
        total_duration
        or he.get_duration(input_file)
        or _try_concat_duration(ffmpeg_cmd)
        or ((float(total_frames) / float(fps)) if (total_frames and fps) else None)
    )

    # Indeterminate-Bar
    def _make_indeterminate(tick: int) -> tuple[str, str]:
        width = bar_len
        block = max(1, width // 8)
        pos = tick % (width + block)
        start = max(0, min(pos, width - block))
        left = " " * start
        right = " " * (width - start - block)
        colour = c.get("cyan") if hasattr(c, "get") else ""
        inner = f"{colour}{'█' * block}{c.get('reset')}"
        return f"[{left}{inner}{right}]   …", colour or ""

    # initial
    if total_frames or duration:
        bar0, colour = gh.make_bar(0.0, bar_len)
    else:
        bar0, colour = _make_indeterminate(0)
    gh.print_progress_block(
        colorize_lines(prog_lines_raw, colour, prog_italic_idx), bar0, hint_line
    )

    # --- ffmpeg starten (kein roher Output!) -------------------------------
    FRAME_RE = re.compile(r"frame=\s*(\d+)")
    KV_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*=\s*(.*)\s*$")
    was_cancelled = False

    with subprocess.Popen(
        ffmpeg_cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        bufsize=0,
        start_new_session=True,
    ) as proc:
        old_sigint = signal.getsignal(signal.SIGINT)
        sigint_evt = threading.Event()

        def _on_sigint(signum, frame):
            # Strg+C im aktuellen Prozess → globalen CANCEL setzen
            try:
                CANCEL.set()
            except Exception:
                pass
            sigint_evt.set()
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass

        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except Exception:
            old_sigint = None

        try:
            assert proc.stderr is not None
            buf = b""
            tick = 0
            last_draw = 0.0
            seen_frame = 0
            out_time_sec = 0.0

            def _draw_by_progress(p: float) -> None:
                bar_str, colour = gh.make_bar(max(0.0, min(1.0, p)), bar_len)
                gh.print_progress_block(
                    colorize_lines(prog_lines_raw, colour, prog_italic_idx),
                    bar_str,
                    hint_line,
                )

            while True:
                if sigint_evt.is_set() or is_cancelled():
                    was_cancelled = True
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    break

                chunk = proc.stderr.read(1024)
                if not chunk and proc.poll() is not None:
                    break
                if not chunk:
                    now = time.time()
                    if not (total_frames or duration) and now - last_draw > 0.1:
                        tick += 1
                        bar_str, colour = _make_indeterminate(tick)
                        gh.print_progress_block(
                            colorize_lines(prog_lines_raw, colour, prog_italic_idx),
                            bar_str,
                            hint_line,
                        )
                        last_draw = now
                    continue

                buf += chunk
                parts = buf.split(b"\n" if use_progress_kv else b"\r")
                buf = parts.pop()

                for part in parts:
                    line = part.decode(errors="ignore").strip()

                    if use_progress_kv:
                        m = KV_RE.match(line)
                        if not m:
                            continue
                        k, v = m.group(1).lower(), m.group(2)

                        if k == "frame" and total_frames:
                            try:
                                seen_frame = max(seen_frame, int(v))
                                p = min(seen_frame / max(1, int(total_frames)), 1.0)
                                _draw_by_progress(p)
                                last_draw = time.time()
                                continue
                            except Exception:
                                pass

                        if k in ("out_time_ms", "out_time") and duration:
                            try:
                                if k == "out_time_ms":
                                    out_time_sec = float(v) / 1_000_000.0
                                else:
                                    hh, mm, ss = v.split(":")
                                    out_time_sec = (
                                        int(hh) * 3600 + int(mm) * 60 + float(ss)
                                    )
                                p = min(out_time_sec / max(0.001, float(duration)), 1.0)
                                _draw_by_progress(p)
                                last_draw = time.time()
                            except Exception:
                                pass
                        # progress=end wird einfach durch proc.wait() abgeschlossen
                        continue

                    # Fallback: klassische Stats-Zeilen
                    if total_frames:
                        mF = FRAME_RE.search(line)
                        if mF:
                            try:
                                seen_frame = max(seen_frame, int(mF.group(1)))
                                p = min(seen_frame / max(1, int(total_frames)), 1.0)
                                _draw_by_progress(p)
                                last_draw = time.time()
                                continue
                            except Exception:
                                pass

                    mT = defin.TIME_RE.search(line) if not use_progress_kv else None
                    if mT and duration:
                        h, mi, s = mT.groups()
                        current = int(h) * 3600 + int(mi) * 60 + float(s)
                        p = min(current / max(0.001, duration), 1.0)
                        _draw_by_progress(p)
                        last_draw = time.time()
                        continue

                    if not (total_frames or duration):
                        tick += 1
                        bar_str, colour = _make_indeterminate(tick)
                        gh.print_progress_block(
                            colorize_lines(prog_lines_raw, colour, prog_italic_idx),
                            bar_str,
                            hint_line,
                        )
                        last_draw = time.time()

            proc.wait()
        finally:
            try:
                if old_sigint is not None:
                    signal.signal(signal.SIGINT, old_sigint)
            except Exception:
                pass
            try:
                sys.stdout.write("\033[0m\033[?25h")
                sys.stdout.flush()
            except Exception:
                pass
            try:
                subprocess.run(
                    ["stty", "sane"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

    # --- Abschlussblock ----------------------------------------------------
    ret = proc.returncode or 0
    success = (ret == 0) and (
        not has_real_output or (has_real_output and output_path.exists())
    )
    outfile_for_template = output_path if (success and has_real_output) else input_path

    # 1) An den Anfang des reservierten Blocks springen
    sys.stdout.write(f"\033[{reserve_lines}F")

    # 2) Den gesamten reservierten Block *vollständig leeren*
    #    (so verschwinden Progress-Zeilen, Bar und Hint – egal wie viele Zeilen es waren)
    for _t in range(reserve_lines):
        sys.stdout.write(defin.CLEAR_LINE + "\n")

    # 3) Wieder an den Block-Anfang zurück
    sys.stdout.write(f"\033[{reserve_lines}F")

    # 4) Abschlussausgabe:
    if was_cancelled:
        # Abbruchzeile (orange)
        cancel_msg = (
            _("cancelled")
            if "cancelled" in getattr(defin, "MESSAGES", {}).get("de", {})
            else (
                _("aborted")
                if "aborted" in getattr(defin, "MESSAGES", {}).get("de", {})
                else "Abgebrochen."
            )
        )
        sys.stdout.write(
            defin.CLEAR_LINE
            + c.get("orange")
            + "   "
            + cancel_msg
            + c.get("reset")
            + "\n"
        )

    elif success:
        # Nur bei Erfolg finished_line ausgeben (kann 1–3 Zeilen sein)
        fin_lines_raw, fin_italic_idx = split_finished_template(
            finished_line, input_path, outfile_for_template
        )
        final_lines = colorize_lines(fin_lines_raw, c.get("green"), fin_italic_idx)
        for s in final_lines:
            sys.stdout.write(defin.CLEAR_LINE + s + "\n")

    else:
        # Failzeile (rot)
        err_msg = (
            _("ffmpeg_failed")
            if "ffmpeg_failed" in getattr(defin, "MESSAGES", {}).get("de", {})
            else "ffmpeg-Fehler – Ausgabe wurde nicht erstellt."
        )
        sys.stdout.write(
            defin.CLEAR_LINE + c.get("red") + "   " + err_msg + c.get("reset") + "\n"
        )

    sys.stdout.flush()

    # ── HARTES FEHLER-/CANCEL-VERHALTEN (nur im Standardmodus) ───────────────
    if mode == 0:
        if was_cancelled:
            try:
                CANCEL.set()
            except Exception:
                pass
            _fail_cleanup(remove_partial=True)
            # Harte Unterbrechung des Aufruferflusses:
            raise KeyboardInterrupt("Abbruch durch Benutzer (ESC/Strg+C).")

        if not success:
            # ffmpeg-Fehler oder Output fehlt → als Fehler behandeln
            _fail_cleanup(remove_partial=True)
            raise FFmpegFailed(
                f"ffmpeg fehlgeschlagen (exit={ret}); Ausgabe nicht erstellt."
            )

        # Erfolg
        return output_path  # type: ignore[return-value]

    # ── Code-/Silent-Modi behalten altes Verhalten ────────────────────────────
    if mode > 0:
        # mode==1: Exit-Code zurück
        return ret if mode == 1 else None

    # Fallback (sollte eigentlich nie hier landen)
    return output_path if (success and has_real_output) else None
