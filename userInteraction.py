#!/usr/bin/env python3
import definitions as defin
import sys
import re
import os
import subprocess
import shutil
import base64
from pathlib import Path
from typing import Sequence
from typing import Sequence, Iterable, List, Union, Any, Tuple


# ANSI-Farbcodes: Bold + Gelb, danach Reset
_BOLD_YELLOW = "\033[1;33m"
_BOLD_RED = "\033[1;31m"
_RESET        = "\033[0m"


def print_error(messgage: str):
    print(f"\n{_BOLD_RED}⚠️ ERROR: {messgage}{_RESET}\n")


def ask_user(prompt, options, descriptions=None):
    print(" ")
    print(f"{prompt}:")
    print("----------------------------------")
    for i, opt in enumerate(options):
        if descriptions and opt in descriptions:
            desc = f"{descriptions[opt]}"
        else:
            desc = opt  # Fallback: zeige Option selbst als Text
        print(f"    [{i+1}] {desc}")
    print("    [0] 🚪 Exit")
    print(" ")
    choice = input("\033[1;33m      Choose an option: \033[0m")
    if choice == "0":
        print("Exiting...")
        sys.exit(0)
    try:
        return options[int(choice)-1]
    except:
        return options[0]


def is_kitty_terminal():
    return os.environ.get("TERM", "").startswith("xterm-kitty")

def display_image_in_kitty(image_path: str | Path) -> None:
    """Zeigt *image_path* inline im Kitty-Terminal an (sofern möglich)."""
    # ------------------------------------------------------------------
    # 1) Nur im Kitty-Terminal überhaupt versuchen
    # ------------------------------------------------------------------
    if not is_kitty_terminal():
        return

    img = str(image_path)

    # ------------------------------------------------------------------
    # 2) Bevorzugt: icat-Kitten benutzen (macht Chunking & Clean-Up)
    # ------------------------------------------------------------------
    if shutil.which("kitty"):
        try:
            subprocess.run(
                ["kitty", "+kitten", "icat", "--transfer-mode=stream", "--clear", img],
                check=True,
            )
            return
        except subprocess.CalledProcessError:
            pass  # Fallback auf Raw-Protocol

    # ------------------------------------------------------------------
    # 3) Fallback: Kitty Graphics Protocol „raw“
    #    (Base64 in <= 4096-Byte-Chunks, danach Graphik löschen)
    # ------------------------------------------------------------------
    CHUNK = 4096
    with open(img, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()

    write = sys.stdout.write
    flush = sys.stdout.flush

    for i in range(0, len(data), CHUNK):
        chunk = data[i : i + CHUNK]
        more = 1 if i + CHUNK < len(data) else 0  # m=1 → noch Daten folgen
        # f=100: JPEG/PNG, t=f: file-transfer, s=Größe (nur beim ersten Chunk)
        header = (
            f"\x1b_Gf=100,t=f,m={more}"
            + (f",s={len(data)}" if i == 0 else "")
            + ";"
        )
        write(header + chunk + "\x1b\\")
        flush()

    # Bild nach Anzeige gleich wieder entfernen (optional)
    write("\x1b_Ga=d\x1b\\")  # a=d → delete graphic
    flush()




def expand_glob_patterns(file_list):
    """Expands file patterns like 'conan%.mp4' to real file paths."""
    expanded = []
    for f in file_list or []:
        if '%' in f:
            pattern = f.replace('%', '*')
            matches = sorted(str(p) for p in Path().glob(pattern) if p.is_file())
            expanded.extend(matches)
        else:
            expanded.append(f)
    return expanded



def find_files_with_extensions(*extension_groups: Sequence[str]) -> list[str]:
    """
    Suche Dateien im aktuellen Verzeichnis, deren Suffix zu einer der angegebenen Extension-Gruppen passt.
    Du kannst mehrere Listen (z. B. VIDEO_EXTENSIONS, AUDIO_EXTENSIONS, …) übergeben.
    """
    cwd = Path.cwd()
    valid_exts = {ext.lower() for group in extension_groups for ext in group}
    return [p.name for p in cwd.iterdir() if p.suffix.lower() in valid_exts]


def parse_file_selection(input_str: str, max_index: int) -> list[int]:
    if not input_str.strip():  # ENTER = alle
        return list(range(max_index))
    if input_str.strip() == '0':
        sys.exit(0)

    result = []
    parts = re.split(r"[\s,]+", input_str.strip())
    last = -1
    for part in parts:
        if '-' in part:
            start, end = part.split('-', 1)
            if start.isdigit() and end.isdigit():
                result.extend(range(int(start)-1, int(end)))
                last = int(end)-1
        elif part == '+':
            if last >= 0:
                result.extend(range(last, max_index))
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < max_index:
                result.append(idx)
                last = idx
    return result


def select_files_interactively(*extension_groups: Sequence[str]) -> list[str]:
    found_files = find_files_with_extensions(*extension_groups)
    if not found_files:
        print_error("No maching files found!")
        sys.exit(0)

    print("\n Found Files:")
    for i, f in enumerate(found_files):
        print(f"  [{i+1}]  {Path(f).name}")

    print("\nEnter numbers to select files (e.g. 1,3,5-7,+), press ENTER for all, or 0 to Exit:")
    user_input = input("> ")
    indices = parse_file_selection(user_input, len(found_files))
    print("\n\033[1;33m Selected Files:\033[0m")
    for i in indices:
        if 0 <= i < len(found_files):
            print(f"  - {found_files[i]}")
    print(" ")
    return [found_files[i] for i in indices if 0 <= i < len(found_files)]



def any_file_exists(files: Union[Iterable[str], None]) -> bool:
    """
    Liefert 𝙏𝙧𝙪𝙚, sobald mindestens eine der übergebenen Dateien existiert.
    Ist `files` leer/None oder existiert keine der Dateien, gibt die
    Funktion 𝙁𝙖𝙡𝙨𝙚 zurück und meldet fehlende Dateien auf stderr/console.
    """
    if not files:           # None, [] oder andere leere Iterables
        return False

    any_found = False
    for file in files:
        p = Path(file).expanduser().resolve()
        if p.exists():
            any_found = True
    return any_found


def get_input(prompt, default=None):
    """Benutzerabfrage mit optionalem Default-Wert."""
    if default:
        prompt = f"{prompt} [ENTER = {default}]: "
    else:
        prompt = f"{prompt}: "
    response = input(prompt).strip()
    return response if response else default


def select_from_list(prompt, options, default=None):
    """
    Interaktive Auswahl aus einer Liste von Optionen.
    Optionen ist eine Liste von Tupeln: (Wert, Label)
    """
    print(f"\n{prompt}")
    for i, (_, label) in enumerate(options):
        print(f"  {i+1}) {label}")

    while True:
        choice = input(f"\n\033[1;33mSelect [1-{len(options)}] or press ENTER for default ({default}):\033[0m ").strip()
        if choice == "" and default is not None:
            return default
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        print("Invalid selection. Please try again.")





def prepare_files(
    action_name: str,
    args: Any,
    *extension_groups: Sequence[str],
) -> List[str]:
    """
    Erledigt den immer gleichen Boilerplate für einen Menüpunkt:

    1. Überschrift ausgeben, z. B. “[convert] Start video conversion…”.
    2. Dateien bestimmen:
       • Wenn `args.files` gesetzt ist ⇒ Glob-Patterns expandieren.
       • Sonst interaktiv auswählen (über `ui.select_files_interactively`),
         basierend auf den übergebenen Extension-Gruppen.
    3. Prüfen, dass mindestens eine der Dateien tatsächlich existiert
       (`ui.any_file_exists`). Fehlermeldung und Rückkehr [] bei Nichterfolg.

    Parameter
    ---------
    action_name :
        Name des Menüpunktes, z. B. "convert", "extract", …
    args :
        Das Namespace-Objekt (meist von argparse), das optional `files` enthält.
    *extension_groups :
        Beliebig viele Sequenzen von Dateiendungen, die bei interaktiver
        Auswahl angeboten werden sollen (z. B. defin.VIDEO_EXTENSIONS).

    Rückgabe
    --------
    Liste der geprüften Dateien (Strings). Leere Liste, falls keine
    existierenden Dateien gefunden wurden.
    """
    ##print(f"\n[{action_name}] Start {action_name}...\n")
    print(f"\n\033[1;32m    ================  [{action_name}] starting  ================\033[0m\n")

    # 1. Dateien aus den Argumenten oder interaktiv bestimmen
    if getattr(args, "files", None):
        files = expand_glob_patterns(args.files)
    else:
        files = select_files_interactively(*extension_groups)

    # 2. Existenz checken
    if not any_file_exists(files):
        print_error("No file found!")
        return []

    return files



def get_parameter(
        param: str,
        args: Any,
        promt: str,
        param_list: List[str],
        param_description: List[str],
        default: str,
        force: bool,
) -> str:
    
    choice = getattr(args, param, None)
    if (not choice and not args.files) or force:
        choice = ask_user(promt, param_list, param_description)
    if choice not in param_list:
        choice = default
    return choice



def print_summary(
    files: Sequence[str],
    *info_pairs: Tuple[str, str],
    action: str = "processing",
) -> None:
    """
    Zeigt eine kompakte Übersicht im Terminal an.

    Parameter
    ----------
    files :
        Beliebig viele Pfade/Dateinamen. Darf nicht leer sein.
    *info_pairs :
        Beliebig viele (label, value)-Paare, z. B.  
        ("Format", "mp4"), ("Preset", "slow"), ...
        Die Reihenfolge bleibt erhalten.
    action :
        Verb in der Kopfzeile (Default: "processing").
    """
    if not files:
        raise ValueError("`files` can not be None.")

    # Kopfzeile --------------------------------------------------------------
    print()  # Leerzeile
    print(f"{_BOLD_YELLOW}   Start {action} {len(files)} files:{_RESET}")
    for f in files:
        print(f"   => {f}")
    print("\n" + "-" * 38 + "\n")

    # Parameterliste ---------------------------------------------------------
    for label, value in info_pairs:
        print(f"{_BOLD_YELLOW}{label}:{_RESET} {value}")
    print()  # abschließende Leerzeile

# ---------------------------------------------------------------------------


def print_finished(action_name: str):
   # print("\n  ===================================================")
    print(f"\n\033[1;32m    ================  [{action_name}] finished  ================\033[0m\n")
   # print("  ===================================================\n")


# ---------------------------------------------------------------------------
# Helper: get duration of a media file --------------------------------------
# ---------------------------------------------------------------------------


def get_duration(file_path: str | Path) -> float | None:
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(file_path)],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except FileNotFoundError:
        print_error("ffmpeg not found – install it first.")
    return None



def is_time_in_video(time_str: str, src: str | Path) -> bool:
    """
    Prüft, ob die gegebene Zeit (HH:MM:SS) mindestens 1 Sekunde kürzer
    ist als die Dauer des Videos.
    """
    try:
        h, m, s = map(int, time_str.strip().split(":"))
        input_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        print_error(f"Invalit time format: {time_str}")
        return False

    duration = get_duration(src)
    if duration is None:
        return False

    return input_seconds < (duration - 1)


# ---------------------------------------------------------------------------
# Colour gradient helper -----------------------------------------------------
# ---------------------------------------------------------------------------

def _gradient_colour(progress: float):
    """Return RGB tuple for a purple→blue→green gradient at *progress* (0–1)."""
    progress = max(0.0, min(progress, 1.0))
    if progress < 0.5:
        # Purple (255,0,255) to Blue (0,0,255)
        t = progress * 2  # scale 0–0.5 → 0–1
        r = int(255 * (1 - t))          # 255 → 0
        g = 0
        b = 255
    else:
        # Blue (0,0,255) to Green (0,255,0)
        t = (progress - 0.5) * 2        # scale 0.5–1 → 0–1
        r = 0
        g = int(255 * t)                # 0 → 255
        b = int(255 * (1 - t))          # 255 → 0
    return r, g, b


def _visible_len(text: str) -> int:
    """Printable length without ANSI escape sequences."""
    return len(defin.ANSI_REGEX.sub("", text))

# ---------------------------------------------------------------------------
# Two‑line refresh helper ----------------------------------------------------
# ---------------------------------------------------------------------------

def _print_two_lines(text_line: str, bar_line: str) -> None:
    """Helper to overwrite the previous two‑line block in place."""
    sys.stdout.write("\033[2F")           # move cursor up 2 lines
    sys.stdout.write(defin.CLEAR_LINE + text_line + "\n")
    sys.stdout.write(defin.CLEAR_LINE + bar_line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Silent ffmpeg with progress ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_ffmpeg_with_progress(
    input_file: str | Path,
    ffmpeg_cmd: list[str],
    progress_line: str,
    finished_line: str,
    total_duration: float | None = None, 
    bar_len_default: int = 80,
    min_bar: int = 20,
) -> None:
    term_cols = shutil.get_terminal_size((80, 20)).columns
    input_file = Path(input_file)

    # Truncate progress_line if too long
    if _visible_len(progress_line) > term_cols:
        progress_line = progress_line[: term_cols - 1] + "…"

    # Determine output path (last arg unless trailing -y)
    output_arg = ffmpeg_cmd[-1] if ffmpeg_cmd[-1] != "-y" else ffmpeg_cmd[-2]
    output_path = Path(output_arg)
    if output_path.exists():
        if input(f"File '{output_path}' exists. Overwrite? [y/N]: ").strip().lower() != "y":
            print("  ➜ Skipping.")
            return
        else:
            ffmpeg_cmd.insert(-1, "-y")

    duration = total_duration or (get_duration(input_file) if input_file else None)
    if not duration:
        print(progress_line)
        subprocess.run(ffmpeg_cmd)
        print(finished_line)
        return

    # Reserve two lines
    print("\n\n", end="", flush=True)

    # Bar width fixed by terminal width only
    static_len = len(" 100% []")
    bar_len = max(min_bar, min(bar_len_default, term_cols - static_len))

    def make_bar(progress: float) -> str:
        filled = int(bar_len * progress)
        r, g, b = _gradient_colour(progress)
        colour = f"\033[38;2;{r};{g};{b}m"
        inner = f"{colour}{'█'*filled}{defin.ANSI_RESET}{' '*(bar_len-filled)}"
        return f"[{inner}] {progress*100:3.0f}%", colour

    # Initial 0‑% display
    bar0, colour0 = make_bar(0)
    _print_two_lines(f"{colour0}{progress_line}{defin.ANSI_RESET}", bar0)

    # Run FFmpeg and parse carriage‑return updates
    with subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=0) as proc:
        buf = b""
        while True:
            chunk = proc.stderr.read(1024)
            if not chunk and proc.poll() is not None:
                break
            if not chunk:
                continue
            buf += chunk
            parts = buf.split(b"\r")
            buf = parts.pop()
            for part in parts:
                try:
                    line = part.decode(errors="ignore")
                except UnicodeDecodeError:
                    continue
                m = defin.TIME_RE.search(line)
                if not m:
                    continue
                h, mi, s = m.groups()
                current = int(h) * 3600 + int(mi) * 60 + float(s)
                progress = min(current / duration, 1.0)
                bar_str, colour = make_bar(progress)
                truncated_line = progress_line
                if _visible_len(truncated_line) > term_cols:
                    truncated_line = truncated_line[: term_cols - 1] + "…"
                _print_two_lines(f"{colour}{truncated_line}{defin.ANSI_RESET}", bar_str)

        proc.wait()

    # Prepare safe finished line (truncate if needed)
    MIN_TEXT_MARGIN = 8
    vis_len = _visible_len(finished_line)
    max_vis = term_cols - MIN_TEXT_MARGIN
    if vis_len > term_cols:
        plain = finished_line
        if len(plain) > term_cols:
            finished_line = plain[: term_cols - 1] + "…"
        finished_line = plain[: max_vis - 1] + "…"

    # Final line, clear bar
    green = "\033[38;2;0;255;0m"
    sys.stdout.write("\033[2F")
    sys.stdout.write(defin.CLEAR_LINE + f"{green}{finished_line}{defin.ANSI_RESET}\n")
    sys.stdout.write(defin.CLEAR_LINE)
    sys.stdout.flush()
