#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Pfade zu bestehenden Skripten oder Funktionen
SCREENCAST_SCRIPT = "/home/dave/syscripts/videoProcessing/screenrecord/screenrecord.sh"

# Optionen für Konvertierung
FORMATS = ["mp4", "mkv", "webm"]
PRESETS = {
    "smartphone": "-preset veryfast -crf 28",
    "signal": "-preset fast -crf 30 -vf scale=640:-2",
    "youtube": "-preset slow -crf 22"
}
RESOLUTIONS = ["original", "720p", "1080p"]
FRAMERATES = ["original", "24", "30", "60"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"]

def ask_user(prompt, options):
    print(f"{prompt}:")
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    choice = input("Wähle eine Option: ")
    try:
        return options[int(choice)-1]
    except:
        return options[0]

def find_video_files():
    cwd = Path.cwd()
    return [str(p) for p in cwd.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS]

def convert(args):
    print("\n[convert] Starte Videokonvertierung...\n")

    format_choice = getattr(args, 'format', None) or ask_user("Wähle Ausgabeformat", FORMATS)
    preset_choice = getattr(args, 'preset', None) or ask_user("Wähle Voreinstellung für Zielgerät", list(PRESETS.keys()))
    resolution = getattr(args, 'resolution', None) or ask_user("Wähle Zielauflösung", RESOLUTIONS)
    framerate = getattr(args, 'framerate', None) or ask_user("Wähle Ziel-Framerate", FRAMERATES)

    if hasattr(args, 'files') and args.files:
        files = args.files
    else:
        print("\nKeine Eingabedateien übergeben. Suche nach Videodateien im aktuellen Verzeichnis...\n")
        found_files = find_video_files()
        if not found_files:
            print("Keine Videodateien gefunden.")
            return
        print("Gefundene Dateien:")
        for i, f in enumerate(found_files):
            print(f"  {i+1}. {f}")
        print("\nGib Nummern der zu konvertierenden Dateien an (z. B. 1 3 4):")
        selection = input("> ").strip().split()
        files = [found_files[int(s)-1] for s in selection if s.isdigit() and 0 < int(s) <= len(found_files)]

    for file in files:
        input_path = Path(file).resolve()
        if not input_path.exists():
            print(f"Datei nicht gefunden: {input_path}")
            continue

        output_path = input_path.with_suffix(f".{format_choice}")
        ffmpeg_cmd = ["ffmpeg", "-i", str(input_path)]

        if resolution != "original":
            scale = "1280:720" if resolution == "720p" else "1920:1080"
            ffmpeg_cmd += ["-vf", f"scale={scale}"]

        if framerate != "original":
            ffmpeg_cmd += ["-r", framerate]

        ffmpeg_cmd += PRESETS[preset_choice].split()
        ffmpeg_cmd += [str(output_path)]

        print(f"\nKonvertiere: {input_path} -> {output_path}\n")
        subprocess.run(ffmpeg_cmd)

    print("\n[convert] Fertig.")


def screencast(args):
    print("[screencast] Starte Bildschirmaufnahme...")
    subprocess.run(["python3", SCREENCAST_SCRIPT])

def interpolate(args):
    print("[interpolate] Starte Frame-Interpolation...")

    factor = args.factor
    if not factor:
        factor = input("Framerate-Faktor oder Ziel-FPS (z. B. 2, 3 oder 60): ")

    files = args.files or find_video_files()
    if not files:
        print("Keine Videodateien gefunden.")
        return

    for file in files:
        path = Path(file)
        if not path.exists():
            print(f"Datei nicht gefunden: {path}")
            continue

        suffix = f"_interpolated{factor}x" if factor.isdigit() else f"_interpolated{factor}fps"
        output = path.with_stem(path.stem + suffix)

        cmd = [
            "ffmpeg", "-i", str(path),
            "-filter:v", f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={factor}'",
            str(output)
        ]
        print(f"Interpoliere: {path} → {output}")
        subprocess.run(cmd)

def trim(args):
    print("[trim] Funktion wird noch implementiert.")

def compress(args):
    print("[compress] Funktion wird noch implementiert.")

def gif(args):
    print("[gif] Funktion wird noch implementiert.")

def extract_audio(args):
    print("[extract-audio] Funktion wird noch implementiert.")

def metadata(args):
    print("[metadata] Funktion wird noch implementiert.")

def merge(args):
    print("[merge] Funktion wird noch implementiert.")


def show_info(subcommand=None):
    if subcommand == "screencast":
        infofile = SCREENCAST_INFO
    else:
        base = os.path.splitext(os.path.abspath(__file__))[0]
        infofile = f"{base}.{subcommand}.info" if subcommand else f"{base}.info"

    subprocess.run([
        "bash", "-c",
        f"source ~/syscripts/functions/scripting.sh && show_infofile \"{infofile}\""
    ])
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="Video Manager CLI - Tool zur Videobearbeitung (convert, trim, merge, etc.)",
        usage="video <command> [<args>]",
        add_help=False
    )

    parser.add_argument("--help", action="store_true", help="Zeigt die Hilfe an")
    subparsers = parser.add_subparsers(dest="command")

    commands = {
        "convert": convert,
        "screencast": screencast,
        "interpolate": interpolate,
        "trim": trim,
        "compress": compress,
        "gif": gif,
        "extract-audio": extract_audio,
        "metadata": metadata,
        "merge": merge
    }

    for cmd in commands:
        if cmd == "convert":
            parser_convert = subparsers.add_parser(cmd, help=f"{cmd} videos")
            parser_convert.add_argument("files", nargs="*", help="Eingabedateien")
            parser_convert.add_argument("--format", "-f", choices=FORMATS, help="Zielformat")
            parser_convert.add_argument("--preset", "-p", choices=list(PRESETS.keys()), help="Zielgerät-Preset")
            parser_convert.add_argument("--resolution", "-r", choices=RESOLUTIONS, help="Zielauflösung")
            parser_convert.add_argument("--framerate", "-fr", choices=FRAMERATES, help="Ziel-Framerate")
        else:
            subparsers.add_parser(cmd, help=f"{cmd} videos")

    try:
        args = parser.parse_args()
        if args.help:
            show_info(args.command)
    except SystemExit:
        show_info(args.command)

    if args.command in commands:
        commands[args.command](args)
    else:
        show_info(args.command)

if __name__ == "__main__":
    main()

