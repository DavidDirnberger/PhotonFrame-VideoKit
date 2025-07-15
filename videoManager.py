#!/usr/bin/env python3

from __future__ import annotations
import shutil
from shutil import copyfile
import argparse
import subprocess
import tempfile
import signal
import threading
import sys
import os
import re
import json, shlex
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from datetime import datetime, timedelta
from datetime import datetime
import definitions as defin
import userInteraction as ui
import helpers as he
import video_record as vr
import video_filters as vf
from video_thumbnail import set_thumbnail, check_thumbnail, _extract_mp4_cover, _embed_mp4, _has_mp4_cover


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
   
default_args = argparse.Namespace(
            files=[], format=None, preset=None, resolution=None, framerate=None,
            factor=None, start=None, duration=None, quality=None,
            title=None, artist=None, comment=None, album=None, genre=None,
            date=None, track=None, composer=None, publisher=None, language=None,
            show=None, season_number=None, episode_id=None, network=None,
            director=None, production_year=None, keywords=None, text_top=None,
            text_bottom=None, font_size=None, offset=None, pause=None, output=None,
            burn_subtitle=None, subtitle_name=None, target_res=None,
            offset_x=None, offset_y=None, audio=None, subtitle=None, frame=None,
            video=None, codec=None, crf=None, threads=None, warmth=None, tint=None
)


def convert(args):

    files = ui.prepare_files("convert", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return 

    # Prüfen ob irgendein Parameter übergeben wurde
    interactive_mode = not any([
        getattr(args, 'format', None),
        getattr(args, 'preset', None),
        getattr(args, 'resolution', None),
        getattr(args, 'framerate', None),
        getattr(args, 'codec', None),
    ])

    #print(f"interactive mode == {interactive_mode}")

    format_choice = ui.get_parameter("format",args,"Chose output format [ENTER = MP4]: ",defin.FORMATS,defin.FORMAT_DESCRIPTIONS,'mp4',interactive_mode)
    preset_choice = ui.get_parameter("preset",args,"Select preset target [ENTER = Casual]",list(defin.PRESETS.keys()),defin.PRESET_DESCRIPTIONS,"casual",interactive_mode)
    resolution = ui.get_parameter("resolution",args,"Choose target resolution [ENTER = Original]",defin.RESOLUTIONS,defin.RESOLUTION_DESCRIPTIONS,"original",interactive_mode)
    framerate = ui.get_parameter("framerate",args,"Select target frame rate [ENTER = Original]",defin.FRAMERATES,defin.FRAMERATE_DESCRIPTIONS,"original",interactive_mode)
    codec_choice = ui.get_parameter("codec",args,"Select video codec [ENTER = copy]",list(defin.CODEC_DESCRIPTIONS.keys()),defin.CODEC_DESCRIPTIONS,"copy",interactive_mode)

    print(f"codec_choice = {codec_choice}")

    ui.print_summary(
        files,
        ("Format",format_choice),
        ("Preset",preset_choice),
        ("Resolution",resolution),
        ("Framerate",framerate),
        ("Codec",codec_choice),
        action="converting"
    )

    #conversions = []
    for file in files:
        input_path = Path(file).resolve()
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        #ofile = output_path.name
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(input_path)]

        if resolution != "original":
            scale = defin.RESOLUTION_SCALES.get(resolution)
            if scale:
                ffmpeg_cmd += ["-vf", f"scale={scale}"]

        if framerate != "original":
            ffmpeg_cmd += ["-r", framerate]

        if codec_choice != "copy":
            ffmpeg_cmd += ["-c:v", codec_choice]

        in_ext  = input_path.suffix.lstrip(".").lower()
        out_ext = format_choice.lstrip(".").lower()

        if in_ext == out_ext:
            # z. B. "_convertedh264_1920x1080"
            suffix = f"_converted{codec_choice}_{resolution}"
            new_name = f"{input_path.stem}{suffix}.{in_ext}"
            opath = input_path.with_name(new_name)
        else:
            opath = input_path.with_suffix(f".{out_ext}")

        ofile = opath.name

        ffmpeg_cmd += defin.PRESETS[preset_choice].split()
        ffmpeg_cmd += [str(opath)]

        ui.run_ffmpeg_with_progress(file, ffmpeg_cmd, f"Converting '{file}' ...", f"{file} -> {ofile}")

    ui.print_finished("convert")



def screencast(args):
    print("\n [screencast] Start screencast...")
    subprocess.run(["bash", defin.SCREENCAST_SCRIPT])


# --------------------------------------------------
# Main API
# --------------------------------------------------

def record(args):
    print("\n[record] Capture wizard v3\n")

    # -------- VIDEO --------
    vids = vr._v4l2_devices(); assert vids, "No V4L2 devices"
    vdev = ui.select_from_list("Select video device:", vids, default=vids[0][0])
    while not vr._probe_ffmpeg("v4l2", vdev):
        alt = next((d for d, _ in vids if d != vdev and d.split("video")[-1] != vdev.split("video")[-1]), None)
        if alt and vr._probe_ffmpeg("v4l2", alt):
            print(f"⚠️  {vdev} busy – using {alt} instead.")
            vdev = alt; break
        print("⚠️  Device busy/unavailable."); vdev = ui.select_from_list("Video device:", vids)

    # -------- AUDIO --------
    audio_opts = [("none", "Mute (video‑only)")] + vr._alsa_devices() + vr._pulse_sources()
    adev = ui.select_from_list("Select audio source:", audio_opts, default=audio_opts[1][0] if len(audio_opts)>1 else "none")
    fmt = "pulse" if adev.startswith("alsa_input") or adev.startswith("pulse") else ("alsa" if adev!="none" else "none")
    while fmt!="none" and not vr._probe_ffmpeg(fmt, adev):
        # try pulse counterpart for busy ALSA
        if fmt=="alsa":
            cand = next((s for s,_ in vr._pulse_sources() if any(tok in s for tok in adev.split(':'))), None)
            if cand and vr._probe_ffmpeg("pulse", cand):
                print(f"⚠️  ALSA busy – switching to Pulse source {cand}.")
                adev, fmt = cand, "pulse"; break
        print("⚠️  Audio busy – choose another.")
        adev = ui.select_from_list("Audio source:", audio_opts);
        fmt = "pulse" if adev.startswith("alsa_input") else ("alsa" if adev!="none" else "none")

    # -------- Parameters --------
    dur_raw = ui.get_input("Duration (HH:MM:SS, MM:SS, SS) (empty = ENTER stop)", default="")

    if dur_raw:
        try:
            dur = he.parse_time(dur_raw)
        except ValueError as e:
            ui.print_error(f"❌ Invalid Character: {e}")
            return
    else:
        dur=""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = Path(ui.get_input("Output filename", default=f"recording_{ts}.mp4")).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.suffix == '':
        outfile = outfile.with_suffix('.mp4')


    # -------- FFmpeg command --------
#    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-i", vdev]
#    if fmt!="none": cmd += ["-f", fmt, "-i", adev]
#    cmd += ["-thread_queue_size", "512", "-vsync", "1", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p"]
#    if fmt!="none": cmd += ["-c:a", "aac", "-b:a", "128k"]
#    if dur: cmd += ["-t", dur]
#    cmd.append(str(outfile))

    

    # -------- FFmpeg command (OBS-Style) --------
    cmd = [
        "ffmpeg",
        "-y", "-hide_banner", "-loglevel", "error",

        # ---- Video-Capture ----
        "-f", "v4l2",
        "-thread_queue_size", "512",
        "-pixel_format", "nv12",          # NV12 wie OBS
        "-vsync", "1",
        "-i", vdev,
    ]

    # ---- optionale Audio-Quelle ----
    if fmt != "none":
        cmd += ["-f", fmt, "-thread_queue_size", "512", "-i", adev]

    # ---- Farb-/Format-Filter ----
    # Erzwingt Rec.709, Partial-Range und 4:2:0-Layout
    cmd += [
        "-vf", "colorspace=all=bt709:iall=bt709:fast=1,format=yuv420p"
    ]

    # ---- Video-Encoder ----
    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-profile:v", "high",
        "-level", "4.2",
        "-crf", "18",                    # Qualitätsziel ≈ OBS default
        "-pix_fmt", "yuv420p",           # 4:2:0
        "-color_range", "tv",            # Partial / Limited
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        # doppelter Boden: Tags auch in x264-Seite festschreiben
        "-x264-params", "colormatrix=bt709:range=tv",
    ]

    # ---- Audio-Encoder ----
    if fmt != "none":
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    # ---- optionale Dauerbegrenzung ----
    if dur:
        cmd += ["-t", str(dur)]

    cmd.append(str(outfile))



    # -------- Run --------
    if dur:
        ui.run_ffmpeg_with_progress(outfile.name, cmd, f"Recording to '{outfile.name}'", f"{outfile} created")
    else:
        print("\nRecording… ENTER = stop\n"); buf=[]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        threading.Thread(target=lambda: [buf.append(l.decode()) for l in proc.stderr], daemon=True).start()
        input(); proc.send_signal(signal.SIGINT); time.sleep(5);
        if proc.poll() is None: proc.terminate(); proc.wait()
        if proc.returncode: print("FFmpeg exit", proc.returncode, "– last lines:"); [print(l.strip()) for l in buf[-20:]]
    if outfile.exists() and outfile.stat().st_size:
        subprocess.run(["xdg-open", str(outfile)], check=False); ui.print_finished("video recorded")
    else: print("⚠️  No output file – see errors above.")



def interpolate(args):
    files = ui.prepare_files("interpolate", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return 

    factor_input = args.factor or input("Framerate factor or target FPS (e.g. 2x, 0.5X or 60): ").strip()

    # Entscheide, ob Faktor oder Ziel-FPS
    match = re.fullmatch(r"(?i)(\d*\.?\d+)x", factor_input)
    if match:
        # Faktor-Modus
        factor_value = float(match.group(1))
        is_factor = True
    else:
        try:
            # Ziel-FPS-Modus
            fps_value = float(factor_input)
            is_factor = False
        except ValueError:
            ui.print_error(f"Invalid input: '{factor_input}'")
            return

    for file in files:
        path = Path(file)
        if not path.exists():
            print(f"File not found: {path}")
            continue

        # Ermittele Original-FPS
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            rate_str = result.stdout.strip()
            num, denom = map(int, rate_str.split("/"))
            original_fps = num / denom
        except Exception:
            print(f"Could not determine FPS for: {path.name}")
            continue

        target_fps = factor_value * original_fps if is_factor else fps_value
        print(f"  {path.name}: original FPS = {original_fps:.2f}, target FPS = {target_fps:.2f}")

        suffix = f"_interpolated{factor_input}"
        output = path.with_stem(path.stem + suffix)

        cmd = [
            "ffmpeg", "-i", str(path),
            "-filter:v", f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={target_fps}'",
            str(output)
        ]

        ui.run_ffmpeg_with_progress(path.name,cmd,f"interpolating '{path.name}' from {original_fps:.2f} FPS to {target_fps:.2f} FPS",f"{path.name}: original FPS = {original_fps:.2f}, target FPS = {target_fps:.2f}")
        ui.print_finished("interpolating")



def trim(args):
    """Trim videos using start time + duration (both flexible formats).
    Accurate duration is guaranteed by giving `-ss` AFTER the input.
    """

    files = ui.prepare_files("trimming", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return 

    # 2) Start & Dauer abfragen --------------------------------------------
    start_raw = getattr(args, 'start', None) or input("Trim start (HH:MM:SS, MM:SS, SS): ").strip()
    dur_raw = getattr(args, 'duration', None) or input("Trim duration (HH:MM:SS, MM:SS, SS): ").strip()

    # 3) Normalisierte Strings & Sekunden ----------------------------------
    try:
        start_td = he.parse_time(start_raw)
        dur_td = he.parse_time(dur_raw)
    except ValueError as e:
        ui.print_error(f"❌ Invalid Character: {e}")
        return
    
    start_str = he.seconds_to_time(start_td)
    duration_str = he.seconds_to_time(dur_td)
    end_str = he.add_timecodes(start_str, duration_str)

    # 4) Bearbeiten ---------------------------------------------------------
    for file in files:
        path = Path(file)
        output = path.with_stem(path.stem + "_trimmed")

        cmd = [
            "ffmpeg", "-y",              # overwrite
            "-i", str(path),             # zuerst Eingabe, dann -ss für präzise Seek
            "-ss", start_str,
            "-t", duration_str,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",  # verhindert negative Timestamps
            str(output)
        ]

        ui.run_ffmpeg_with_progress(
            path.name,
            cmd,
            f"trimming '{path.name}' from {start_str} to {end_str}",
            f"{path.name}: {start_str} - {end_str} ({duration_str})"
        )

        ui.print_finished("trimming")



def compress(args):

    files = ui.prepare_files("compressing", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return 
    
    # Benutzereingabe für Qualität zwischen 0 und 100
    input_quality = args.quality or input("Compression quality (0 = max compression, 100 = best quality): ").strip()
    try:
        quality_percent = int(input_quality)
        if not (0 <= quality_percent <= 100):
            raise ValueError
    except ValueError:
        ui.print_error("Invalid input. Please enter a number between 0 and 100.")
        return

    # Mapping von 0–100 auf CRF (empfohlener Bereich 18–30)
    crf_value = 30 - (quality_percent / 100 * 12)  # 100 → 18, 0 → 30
    crf_value = round(crf_value, 1)

    print(f"  → Mapped quality {quality_percent} to CRF = {crf_value}")

    is_kitty = os.environ.get("TERM", "") == "xterm-kitty"

    for file in files:
        path = Path(file)
        output = path.with_stem(path.stem + "_compressed")
        cmd = [
            "ffmpeg", "-i", str(path),
            "-vcodec", "libx264",
            "-crf", str(crf_value),
            "-preset", "slow",
            "-acodec", "aac", "-b:a", "128k",
            str(output)
        ]
        ui.run_ffmpeg_with_progress(path.name, cmd, f"compressing '{path.name}' to quality {quality_percent}", f"{path.name} compressed to {quality_percent}% quality")

        # Vorschaufunktion: Vergleich Original vs. Komprimiert (nur wenn Kitty-Terminal)
        if is_kitty:
            with tempfile.TemporaryDirectory() as tmpdir:
                original_preview = Path(tmpdir) / "original_preview.jpg"
                compressed_preview = Path(tmpdir) / "compressed_preview.jpg"

                def extract_middle_frame(video_path, output_path):
                    probe = subprocess.run([
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                        str(video_path)
                    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

                    try:
                        duration = float(probe.stdout.strip())
                        midpoint = duration / 2
                    except Exception:
                        midpoint = 1.0

                    subprocess.run([
                        "ffmpeg", "-v", "error", "-y", "-ss", str(midpoint), "-i", str(video_path),
                        "-frames:v", "1", str(output_path)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                extract_middle_frame(path, original_preview)
                extract_middle_frame(output, compressed_preview)

                print("\n  Preview comparison:")
                subprocess.run(f"kitty +kitten icat {original_preview}", shell=True)
                print("[original] ⤴                     [compressed] ⤵ ")
                subprocess.run(f"kitty +kitten icat {compressed_preview}", shell=True)
        else:
            print("\n  (Preview skipped – not running in Kitty terminal)")

        original_size = path.stat().st_size / (1024 * 1024)
        compressed_size = output.stat().st_size / (1024 * 1024)
        print(f"  Size: original = {original_size:.2f} MB → compressed = {compressed_size:.2f} MB")
        ui.print_finished("compressing")



def gif(args):

    files = ui.prepare_files("creating gif", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return 

    # Impact font detection
    impact_path = shutil.which("fc-match")
    system_impact = ""
    if impact_path:
        try:
            result = subprocess.run(["fc-match", "-f", "%{file}\n", "Impact"],
                                    capture_output=True, text=True)
            if result.returncode == 0 and Path(result.stdout.strip()).exists():
                system_impact = result.stdout.strip()
        except Exception:
            pass

    if system_impact:
        fontfile = system_impact
    else:
        fontfile = str(Path(__file__).parent / "impact.ttf")
        if not Path(fontfile).exists():
            ui.print_error("'Impact' font not found system-wide and fallback impact.ttf missing.")
            return

    # Font size handling
    font_sizes = {"small": 16, "medium": 27, "large": 45}

    font_size_label = args.font_size

    for file in files:
        path = Path(file)
        output = path.with_suffix(".gif")
        filters = ["fps=10", "scale=320:-1:flags=lanczos"]
        drawtext_filters = []

        top_text = args.text_top
        bottom_text = args.text_bottom


        if top_text is None and bottom_text is None:
            print("  Press ENTER to skip text.")
            top_text = input("  Meme text on top: ").strip()
            bottom_text = input("  Meme text on bottom: ").strip()



        if not args.files:  # interactive mode: prompt
            print("  Choose font size: [s]mall / [m]edium (default) / [l]arge")
            user_input = input("  Font size: ").strip().lower()
            if user_input.startswith("s"):
                font_size_label = "small"
            elif user_input.startswith("l"):
                font_size_label = "large"
            else:
                font_size_label = "medium"
        else:
            font_size_label = font_size_label or "medium"

        fontsize = font_sizes[font_size_label]

        if top_text:
            drawtext_filters.append(
                f"drawtext=fontfile='{fontfile}':text='{top_text}':"
                f"fontcolor=white:fontsize={fontsize}:x=(w-text_w)/2:y=10"
            )
        if bottom_text:
            drawtext_filters.append(
                f"drawtext=fontfile='{fontfile}':text='{bottom_text}':"
                f"fontcolor=white:fontsize={fontsize}:x=(w-text_w)/2:y=h-th-10"
            )


        filter_string = ",".join(filters + drawtext_filters)
        cmd = ["ffmpeg", "-i", str(path), "-vf", filter_string, "-loop", "0", str(output)]

        ui.run_ffmpeg_with_progress(path.name, cmd, f"creating gif from '{path.name}'", f"{str(output)} created")
        subprocess.run(["xdg-open", str(output)])
        ui.print_finished("gif creation")



def extract(args):
    files = ui.prepare_files("extracting", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return 

    # Interactive fallback prompt if no flags given
    if not (args.audio or args.subtitle or args.frame or args.video) and not args.files:
        print("No extraction type given. Choose one or more: [a]udio, [s]ubtitle, [f]rame, [v]ideo")
        choice = input("Enter letters (e.g. as): ").strip().lower()
        args.audio = 'a' in choice
        args.subtitle = True if 's' in choice else None
        args.frame = '' if 'f' in choice else None
        args.video = 'v' in choice

    for file in files:
        path = Path(file)

        # Audio extraction (all tracks)
        if getattr(args, 'audio', False):
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                "stream=index", "-of", "json", str(path)
            ], capture_output=True, text=True)
            audio_streams = json.loads(probe.stdout).get("streams", [])
            for stream in audio_streams:
                idx = stream['index']
                output = path.with_stem(f"{path.stem}_audio{idx}").with_suffix(".mp3")
                cmd = ["ffmpeg", "-y", "-i", str(path), "-map", f"0:a:{idx}", "-q:a", "0", str(output)]
                ui.run_ffmpeg_with_progress(path.name, cmd, f"extracting audio track {idx}", f"{output.name} created")

        # Subtitle extraction
        if getattr(args, 'subtitle', None) is not None:
            subtitle_format = getattr(args, 'format', None)
            if not subtitle_format and not args.files:
                subtitle_format = input("Subtitle format (srt|ass|vtt) [default: srt]: ").strip().lower() or 'srt'
            if subtitle_format not in {'srt', 'ass', 'vtt'}:
                subtitle_format = 'srt'

            probe = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                "stream=index:stream_tags=title", "-of", "json", str(path)
            ], capture_output=True, text=True)
            streams = json.loads(probe.stdout).get("streams", [])

            if not streams:
                print(f"  No subtitle tracks found in '{path.name}'.")
            elif getattr(args, 'subtitle') is True:
                for s in streams:
                    idx = s['index']
                    title = s.get('tags', {}).get('title', f"sub{idx}")
                    output = path.with_stem(path.stem + f"_{title}").with_suffix(f".{subtitle_format}")
                    cmd = ["ffmpeg", "-y", "-i", str(path), "-map", f"0:s:{idx}", str(output)]
                    ui.run_ffmpeg_with_progress(path.name, cmd, f"extracting subtitle #{idx}", f"{output.name} created")
            else:
                name_filter = args.subtitle
                match = None
                for s in streams:
                    title = s.get("tags", {}).get("title", "").lower()
                    if name_filter.lower() in title:
                        match = s["index"]
                        break
                if match is not None:
                    output = path.with_suffix(f".{subtitle_format}")
                    cmd = ["ffmpeg", "-y", "-i", str(path), "-map", f"0:s:{match}", str(output)]
                    ui.run_ffmpeg_with_progress(path.name, cmd, f"extracting subtitle #{match} from '{path.name}'", f"{output.name} created")

        # Frame extraction
        if getattr(args, 'frame', None) is not None:
            time = args.frame
            if not time and not args.files:
                time = input("Time to extract frame (HH:MM:SS or seconds) [default: middle]: ").strip()
                time = time if time else None
            if not time:
                probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True)
                seconds = float(probe.stdout.strip())
                time = str(int(seconds / 2))
            output = path.with_stem(path.stem + "_frame").with_suffix(".png")
            cmd = ["ffmpeg", "-y", "-i", str(path), "-ss", time, "-vframes", "1", str(output)]
            ui.run_ffmpeg_with_progress(path.name, cmd, f"extracting frame from '{path.name}'", f"{output.name} created")
            ui.display_image_in_kitty(output)

        # Video only (remove audio + subs)
        if getattr(args, 'video', False):
            output = path.with_stem(path.stem + "_video")
            final_path = output.with_suffix(path.suffix)
            cmd = ["ffmpeg", "-y", "-i", str(path), "-an", "-sn", "-c:v", "copy", str(final_path)]
            ui.run_ffmpeg_with_progress(path.name, cmd, f"removing audio/subtitles from '{path.name}'", f"{final_path.name} created")

    ui.print_finished("extracting")




def metadata(args):
    """Edit or add metadata tags without removing embedded thumbnails.

    Works for MP4/MOV as well as MKV files.  For MP4 containers the function
    searches for streams with the disposition ``attached_pic`` and explicitly
    restores that flag after copying all streams so that the cover/thumbnail
    is preserved.
    """

    files = ui.prepare_files("metadata", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return

    for file in files:
        path = Path(file)
        print(f"\nMetadata for {path.name}:")

        # ------------------------------------------------------------------
        # 1. Videostream‑Infos anzeigen (Codec, Auflösung, FPS)
        # ------------------------------------------------------------------
        stream_info = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate",
            "-of", "json", str(path)
        ], capture_output=True, text=True)

        check_thumbnail(path)

        try:
            stream = json.loads(stream_info.stdout)["streams"][0]
            print("\n  \033[38;5;117m [Video Info]\033[0m")
            print(f"   Resolution: {stream['width']}x{stream['height']}")
            print(f"   Codec:      {stream['codec_name']}")
            print(f"   FPS:        {stream['r_frame_rate']}")
        except Exception:
            print("  Could not retrieve video stream info.")

        # ------------------------------------------------------------------
        # 2. Vorhandene Container‑Tags einlesen
        # ------------------------------------------------------------------
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags",
            "-of", "json",
            str(path)
        ], capture_output=True, text=True)

        try:
            metadata_json = json.loads(result.stdout)
            tags_raw = metadata_json.get("format", {}).get("tags", {})
        except json.JSONDecodeError:
            tags_raw = {}

        tags = {k.lower(): v for k, v in tags_raw.items()}
        protected_tags = {k: tags[k] for k in tags if k in defin.PROTECTED_KEYS}
        editable_tags = {k: tags[k] for k in tags if k not in defin.PROTECTED_KEYS}

        if tags:
            print("\n  \033[38;5;214m [Protected Metadata Tags]\033[0m")
            for key in sorted(protected_tags):
                print(f"   {key}: {protected_tags[key]}")

            print("\n  \033[38;5;70m [Editable Metadata Tags]\033[0m")
            for key in sorted(editable_tags):
                print(f"   {key}: {editable_tags[key]}")
        else:
            print("  No metadata found.")

        original_tags = tags.copy()
        deleted_keys = set()

        # ------------------------------------------------------------------
        # 3. Flags aus CLI sammeln
        # ------------------------------------------------------------------
        metadata_flags = {
            "title": getattr(args, "title", None),
            "show": getattr(args, "show", None),
            "season_number": getattr(args, "season_number", None),
            "episode_id": getattr(args, "episode_id", None),
            "director": getattr(args, "director", None),
            "production_year": getattr(args, "production_year", None),
            "network": getattr(args, "network", None),
            "artist": getattr(args, "artist", None),
            "album": getattr(args, "album", None),
            "date": getattr(args, "date", None),
            "track": getattr(args, "track", None),
            "composer": getattr(args, "composer", None),
            "publisher": getattr(args, "publisher", None),
            "genre": getattr(args, "genre", None),
            "language": getattr(args, "language", None),
            "comment": getattr(args, "comment", None),
            "keywords": getattr(args, "keywords", None),
        }

        # ------------------------------------------------------------------
        # 4. Interaktiver Editor (wenn keine Flags übergeben wurden)
        # ------------------------------------------------------------------
        if not any(v for v in metadata_flags.values()):
            choice = input("\n  Do you want to edit metadata? [y/N]: ").strip().lower()
            if choice != 'y':
                # Keine Änderungen gewünscht
                continue

            while True:
                tag_order = list(defin.TAG_DESCRIPTIONS.keys())
                all_keys = [k for k in tag_order if k not in defin.PROTECTED_KEYS]
                set_tags = [k for k in all_keys if k in tags]
                unset_tags = [k for k in all_keys if k not in tags]
                indexed_keys = set_tags + unset_tags

                print("\n  \033[38;5;75mEditable Tags (Set):\033[0m")
                for i, key in enumerate(set_tags, 1):
                    desc = defin.TAG_DESCRIPTIONS.get(key, "")
                    print(f"   \033[1;34m[{i:2}] {key:<16}= {tags[key]:<22} # {desc}\033[0m")

                offset = len(set_tags)
                print("\n  \033[38;5;70mEditable Tags (Unset):\033[0m")
                for j, key in enumerate(unset_tags, 1):
                    desc = defin.TAG_DESCRIPTIONS.get(key, "")
                    print(f"   \033[1;32m[{offset + j:2}] {key:<16}  {'':<22} # {desc}\033[0m")

                print("\n \033[1;33m  [#] Set thumbnail \033[0m")
                print("\n   [0] 🚪 Exit and save")
                print("   [?] Help")

                user_input = input("\n  Select tag number to edit or delete: ").strip()

                if user_input == '?':
                    print("\n  Help:")
                    print("   Tags marked as 'protected' are container‑level identifiers")
                    print("   essential for playback compatibility and are excluded here.")
                    continue
                elif user_input == '#':
                    set_thumbnail(file)
                    continue  # Thumbnail gesetzt, Schleife weiter

                try:
                    sel = int(user_input)
                except ValueError:
                    print("  Invalid input. Please enter a number or '?' for help.")
                    continue

                if sel == 0:
                    if tags == original_tags and not deleted_keys:
                        print("\n  Quit video Manager – no changes.")
                        return
                    break

                if 1 <= sel <= len(indexed_keys):
                    key = indexed_keys[sel - 1]
                    new_val = input(f"  Enter new value for '{key}' (leave empty to delete): ").strip()
                    if new_val:
                        tags[key] = new_val
                        deleted_keys.discard(key)
                    elif key in tags:
                        del tags[key]
                        deleted_keys.add(key)
                        print(f"  Tag '{key}' deleted.")
                else:
                    print("  Invalid selection.")
        else:
            # Flags über CLI anwenden
            for k, v in metadata_flags.items():
                if v:
                    tags[k] = v

        # ------------------------------------------------------------------
        # 5. Dateien mit Cover‑Stream ermitteln (nur relevant für MP4/MOV)
        # ------------------------------------------------------------------
        cover_vidx: list[int] = []
        if path.suffix.lower().lstrip('.') in {"mp4", "m4v", "mov"}:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                capture_output=True, text=True
            )

            try:
                all_streams = json.loads(probe.stdout).get("streams", [])
                video_only   = [s for s in all_streams if s["codec_type"] == "video"]

                for v_idx, s in enumerate(video_only):
                    if s.get("disposition", {}).get("attached_pic") == 1:
                        cover_vidx.append(v_idx)          # <-- video-lokaler Index!
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 6. Temporäre Ausgabedatei & FFmpeg‑Kommando bauen
        # ------------------------------------------------------------------
        meta_file = path.with_stem(path.stem + "_meta")

        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-map", "0", "-map_metadata", "0",
            "-c", "copy"
        ]

        for k, v in tags.items():
            if k not in defin.PROTECTED_KEYS:
                cmd += ["-metadata", f"{k}={v}"]
        cmd.append(str(meta_file))

        ui.run_ffmpeg_with_progress(
            file, cmd,
            f"Writing metadata to {meta_file}",
            f"Added metadata to {path.name}"
        )
        if not meta_file.exists():
            ui.print_error("  Failed to update metadata.")
            continue


    # ----------------------------------------------------------------
        # 2) Falls MP4 & Cover vorhanden → Cover extrahieren und erneut
        #    mit _embed_mp4 einbetten.  Ergebnis = *_temp.{ext}
        # ----------------------------------------------------------------
        final_file = path.with_stem(path.stem + "_temp")
        if path.suffix.lower().lstrip(".") in {"mp4", "m4v", "mov"}:
            has_cover, cov_idx = _has_mp4_cover(path)
            if has_cover and cov_idx is not None:
                cover_jpg = _extract_mp4_cover(path, cov_idx)
                _embed_mp4(meta_file, cover_jpg, final_file)
            else:
                # kein Cover → einfach Kopieren
                meta_file.rename(final_file)
        else:
            # MKV etc.: Cover bleibt ohnehin erhalten
            meta_file.rename(final_file)

        # ----------------------------------------------------------------
        # 3) Original überschreiben
        # ----------------------------------------------------------------
        if final_file.exists():
            path.unlink()
            final_file.rename(path)
            print(f"  Successfully updated metadata in {path.name}\n")
        else:
            ui.print_error("  Failed to finalise updated file.")



def merge(args):
    """Merge videos using two‑step approach for matching resolution & fps.

    Step 1  (prep):  Re‑encode each input to shared parameters → temp files.
    Step 2  (mux) :  Concatenate via concat demuxer (‑c copy) – audio stays sync.

    Flags
    ------
    --target-res   smallest | average | largest   (default: largest)
    --pause TIME   pause (black + silent audio) in seconds / MM:SS / HH:MM:SS
    --offset TIME  delay second input if it is audio or subtitle
    """

    # 1) gather files / interactive select

    if hasattr(args, 'files') and args.files:
        files = ui.expand_glob_patterns(args.files)
    else:
        files = ui.select_files_interactively(defin.VIDEO_EXTENSIONS,defin.AUDIO_EXTENSIONS,defin.SUB_EXTENSIONS)

    if not files:
        print('No files selected.');return

    
    # ────────────────── interactive offset prompt ──────────────────
    offset_in = getattr(args,'offset',None)
    if offset_in is None and not args.files:  # menu mode
        offset_in = input("Offset for audio/sub/video (sec or MM:SS) [Enter = 0]: ").strip()
    args.offset = offset_in if offset_in else None
    offset_sec = he.s2sec(args.offset)


    # ─────────────── two‑file shortcuts (video+audio/sub) ────────────────
    fpaths = [Path(f) for f in files]
    if len(fpaths) == 2:
        f1, f2 = fpaths
        if f1.suffix.lower() in defin.AUDIO_EXTENSIONS and f2.suffix.lower() in defin.VIDEO_EXTENSIONS:
            f1, f2 = f2, f1
        if f1.suffix.lower() in defin.SUB_EXTENSIONS and f2.suffix.lower() in defin.VIDEO_EXTENSIONS:
            f1, f2 = f2, f1

        if f1.suffix.lower() in defin.VIDEO_EXTENSIONS and f2.suffix.lower() in defin.AUDIO_EXTENSIONS:
            out = f1.with_stem(f1.stem + '_merged_audio')
            cmd = ['ffmpeg', '-y', '-i', str(f1)]
            if offset_sec:
                cmd += ['-itsoffset', str(offset_sec)]
            cmd += [
                '-i', str(f2), '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(out)
            ]
            ui.run_ffmpeg_with_progress(f1, cmd, f'Merging audio into {f1.name} with offset {offset_in}', f'Created {out.name} with audio offset {offset_in}')
            return

        if f1.suffix.lower() in defin.VIDEO_EXTENSIONS and f2.suffix.lower() in defin.SUB_EXTENSIONS:
            out = f1.with_stem(f1.stem + '_subtitled')
            cmd = ['ffmpeg', '-y', '-i', str(f1)]
            if offset_sec:
                cmd += ['-itsoffset', str(offset_sec)]
            cmd += ['-i', str(f2)]
            if getattr(args, 'burn_subtitle', False):
                cmd += ['-vf', f'subtitles={f2}']
            else:
                cmd += ['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text']
                subtitle_name = getattr(args, 'subtitle_name', None)
                if not subtitle_name and not args.files:
                    subtitle_name=input(f"Subtitle name [Enter = no name]: ").strip()
                if subtitle_name:
                    cmd += ['-metadata:s:s:0', f'title={subtitle_name}']

            cmd += [str(out)]
            ui.run_ffmpeg_with_progress(f1, cmd, f"{'Burning' if args.burn_subtitle else 'Adding'} subtitles to {f1.name} {f'with name {subtitle_name}' if subtitle_name else ''}", f'Created {out.name}')
            return


    # ────────────────── interactive pause prompt ──────────────────
    pause_in = getattr(args,'pause',None)
    if pause_in is None and not args.files:
        pause_in  = input("Pause between clips (sec or MM:SS) [Enter = 0]: ").strip()
    args.pause  = pause_in  if pause_in  else None
    pause_sec = he.s2sec(args.pause)


    # prompt target‑resolution in menu‑mode
    tgt = getattr(args, 'target_res', None)
    if tgt is None and not args.files:
        print("Target resolution: [n]o-scale / [s]mallest / [a]verage / [l]argest (default n)")
        choice = input('> ').strip().lower()
        tgt = 'smallest' if choice.startswith('s') else 'average' if choice.startswith('a') else 'largest' if choice.startswith('l') else 'no-scale'
    tgt = tgt or 'no-scale'
    if tgt not in {'smallest', 'average', 'largest', 'no-scale'}:
        print('target-res must be no-scale|smallest|average|largest'); return

    output_path = getattr(args, 'output', None)
    if not output_path and not args.files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"merged_output_{timestamp}.mp4"
        output_input = input(f"Output filename [Enter = {default_name}]: ").strip()
        output_path = output_input if output_input else default_name
    output_path = Path(output_path or f"merged_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    if output_path.suffix == '':
        output_path = output_path.with_suffix('.mp4')


    # Promt all settings an Files for information
    print("\n === Start merging Video files ===")
    if offset_sec > 0:
        print(f"\nOffset = {offset_in}")
    if pause_sec > 0:
        print(f"Pause = {pause_in}")
    print("\nVideo files:")
    for fil in files:
        print(f" - {fil}")
    print(f"\nOutput name = {output_path}")
    print(" ")

    # compute target WxH (from streams)
    info=[]
    for f in files:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', f], capture_output=True, text=True)
        data = json.loads(probe.stdout)['streams'][0]
        info.append((int(data['width']), int(data['height'])))

    if tgt == 'smallest':
        W, H = min(info)[0], min(info)[1]
    elif tgt == 'largest':
        W, H = max(info)[0], max(info)[1]
    elif tgt == 'average':
        W, H = sum(w for w, _ in info) // len(info), sum(h for _, h in info) // len(info)
    else:  # no-scale
        W, H = max(info)[0], max(info)[1]

    # step 1: transcode each video to match W,H,fps=25, h264/aac
    tmp_dir=Path(tempfile.mkdtemp(prefix='mergeprep_'))
    prep_cache = {}
    pause_cache = {}
    prep_files=[]

    # ───── schwarzes Offset-Bild am Anfang einfügen (nur bei Video-Merge) ─────
    if len(files) > 2 and offset_sec > 0:
        start_black = tmp_dir / f'start_offset_{offset_sec}s.mp4'
        if not start_black.exists():
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i', f'color=black:s={W}x{H}:d={offset_sec}',
                '-f', 'lavfi', '-i', 'anullsrc=cl=stereo:r=48000',
                '-shortest', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-c:a', 'aac', str(start_black)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        prep_files.append(start_black)

    for i, f in enumerate(files):
        f_path = Path(f).resolve()
        if f_path in prep_cache:
            out = prep_cache[f_path]
        else:
            out = tmp_dir / f_path.with_suffix('.prep.mp4').name
            vfilt = f'scale={W}:{H},setsar=1:1' if tgt != 'no-scale' else f"scale=w=iw:h=ih:flags=neighbor,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1:1"
            ui.run_ffmpeg_with_progress(
                f, ['ffmpeg', '-y', '-i', f, '-vf', vfilt, '-r', '25',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(out)],
                f"preparing {f}", f"{f} prepared")
            prep_cache[f_path] = out
        prep_files.append(out)

        if pause_sec > 0 and i < len(files) - 1:
            if pause_sec in pause_cache:
                pause = pause_cache[pause_sec]
            else:
                pause = tmp_dir / f'pause_{pause_sec}s.mp4'
                subprocess.run([
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', f'color=black:s={W}x{H}:d={pause_sec}',
                    '-f', 'lavfi', '-i', 'anullsrc=cl=stereo:r=48000', '-shortest',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-c:a', 'aac',
                    '-movflags', '+faststart', str(pause)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pause_cache[pause_sec] = pause
            prep_files.append(pause)

    # step 2: concat demuxer – write listfile
    listfile=tmp_dir/'list.txt'
    with listfile.open('w') as lf:
        for pf in prep_files:
            lf.write(f"file '{pf.resolve()}'\n")
    concat_cmd=['ffmpeg','-y','-f','concat','-safe','0','-i',str(listfile),'-c','copy',str(output_path)]
    # Anzahl eingeschobener Pausen = Videos-1  (nur wenn pause_sec > 0)
    pause_count = max(0, len(files) - 1) if pause_sec > 0 else 0
    overall = he.total_sec(prep_files, pause_len=pause_sec, pause_count=pause_count)
    ui.run_ffmpeg_with_progress(prep_files[0],concat_cmd,'Concatenating pre‑processed clips',f'Merged to {output_path.name}',total_duration=overall)

    # cleanup tmp dir silently
    shutil.rmtree(tmp_dir,ignore_errors=True)




def crop_pad(args):

    files = ui.prepare_files("crop-pad", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return  # nichts zu tun / Fehlermeldung kam schon aus prepare_files

    resolution = getattr(args, 'resolution', None)
    if resolution is None and not args.files:
        resolution = input("Target resolution (e.g. 1920x1080): ").strip()

    try:
        target_w, target_h = map(int, resolution.lower().split('x'))
    except:
        print("Invalid resolution format. Use WxH, e.g. 1920x1080.")
        return

    offset_x = getattr(args, 'offset_x', None)
    offset_y = getattr(args, 'offset_y', None)

    if offset_x is None and not args.files:
        offset_x = input("X-offset from the center: ").strip()
    if offset_y is None and not args.files:
        offset_y = input("Y-offset from the center: ").strip()

    try:
        offset_x = int(offset_x) if offset_x else 0
        offset_y = int(offset_y) if offset_y else 0
    except ValueError:
        print("Offset must be an integer.")
        return

    for file in files:
        path = Path(file)
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height', '-of', 'json', str(path)
        ], capture_output=True, text=True)


        output_name = getattr(args, 'output', None)
        if not output_name and not args.files:
            if '.' in file:
                name_only = file.rsplit('.', 1)[0]
            else:
                name_only = file
            default_name = f"{name_only}_{target_w}x{target_h}.mp4"
            output_name = input(f"Output filename [Enter = {default_name}]: ").strip() or default_name
        if output_name and not Path(output_name).suffix:
            output_name += ".mp4"


        data = json.loads(probe.stdout)['streams'][0]
        in_w = int(data['width'])
        in_h = int(data['height'])

        # determine filter chain
        if in_w > target_w or in_h > target_h:
            # crop center + offset
            crop_w = min(target_w, in_w)
            crop_h = min(target_h, in_h)
            cx = max((in_w - crop_w) // 2 + offset_x, 0)
            cy = max((in_h - crop_h) // 2 + offset_y, 0)
            if cx < 0 or cy < 0 or cx + crop_w > in_w or cy + crop_h > in_h:
                print(f"Offset too large for '{path.name}' – crop area is out of bounds.")
                continue
            vfilt = f"crop={crop_w}:{crop_h}:{cx}:{cy}"
            prgress_promt=f"Cropping '{path.name}' to resolution {target_w}x{target_h}"
            finished_promt=f"{str(output_name)} croped to resolution {target_w}x{target_h} with offset {offset_x,offset_y}"
        elif in_w < target_w or in_h < target_h:
            # pad center + offset
            pad_x = max((target_w - in_w) // 2 + offset_x, 0)
            pad_y = max((target_h - in_h) // 2 + offset_y, 0)
            vfilt = f"pad={target_w}:{target_h}:{pad_x}:{pad_y}:black"
            prgress_promt=f"Padding '{path.name}' to resolution {target_w}x{target_h}"
            finished_promt=f"{str(output_name)} padded to resolution {target_w}x{target_h} with offset {offset_x,offset_y}"
        else:
            print(f"  Skipping '{path.name}' (already {target_w}x{target_h})")
            continue

        cmd = [
            'ffmpeg', '-y', '-i', str(path),
            '-vf', vfilt,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'copy', str(output_name)
        ]
        print("")
        ui.run_ffmpeg_with_progress(path.name, cmd, prgress_promt, finished_promt)

    ui.print_finished("crop-pad")



# =============================================================================
# Top‑level wrapper – applies cleaning filters to one or many videos
# =============================================================================

def filters(args):
    """Batch‑apply clean‑up filters to selected videos.

    ▸ Stützt sich auf vorhandene UI‑Helferfunktionen:
      • `ui.prepare_files()`   – Dateiauswahl / Globs / Drag‑&‑Drop
      • `ui.get_parameter()`   – Menü mit festen Optionen (z. B. `preset`)
      • `ui.print_error()`     – Schönes Fehlermeldungs‑Layout
      • `ui.run_ffmpeg_with_progress()` – Fortschrittsbalken um FFmpeg
    """

    # ---------------------------------------------------------------------
    # 1) Dateiliste zusammenstellen
    # ---------------------------------------------------------------------
    files = ui.prepare_files("filters", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return

    # ---------------------------------------------------------------------
    # 2) Interaktive / CLI‑basierte Parameterabfrage
    # ---------------------------------------------------------------------

    # --- Preset (Nur feste Auswahl) --------------------------------------
    preset_choice = ui.get_parameter(
        "preset",
        args,
        "Select x264 preset target [ENTER = medium]",
        list(defin.PRESETS.keys()),
        defin.PRESET_DESCRIPTIONS,
        "casual",
        True,
    )

    # --- CRF → über Prozent‑Quality abfragen -----------------------------
    default_quality_str = "75"  # entspricht ca. CRF 21
    input_quality = getattr(args, "quality", None) or input(
        f"Filter quality (0 = max compression, 100 = best quality) [ENTER = {default_quality_str}]: "
    ).strip()
    input_quality = input_quality or default_quality_str

    try:
        quality_percent = int(input_quality)
        if not (0 <= quality_percent <= 100):
            raise ValueError
    except ValueError:
        ui.print_error("Invalid input. Please enter a number between 0 and 100.")
        return

    # Map 0‑100 → CRF 30‑18 (empfohlen für SD/PAL)
    crf_value = 30 - (quality_percent / 100 * 12)
    crf_value = round(crf_value, 1)

    # --- Crop Noise -------------------------------------------------------
    crop_noise = getattr(args, "crop_noise", None) or input(
        "Pixels to crop from bottom to remove VHS head‑switching noise [ENTER = 16]: "
    ).strip()
    crop_noise = int(crop_noise) if crop_noise else 16

    # ---------------------------------------------------------------------
    # 3) Batch‑Verarbeitung
    # ---------------------------------------------------------------------
    for src_path in files:
        src_path = Path(src_path)
        ext = src_path.suffix if src_path.suffix else ".mp4"
        dst_path = Path.cwd() / f"{src_path.stem}_filtered{ext}"

        cmd = vf._build_filter_cmd(
            input_file=src_path,
            output_file=dst_path,
            crop_noise=crop_noise,
            crf=crf_value,
            preset='slow',
        )

        # subprocess muss Strings bekommen → str‑Casting + None‑Filter
        cmd = [str(c) for c in cmd if c is not None]
        print("cmd ===")
        print(cmd)
        print("---------------------")
        subprocess.run(cmd)

#        ui.run_ffmpeg_with_progress(
#            src_path.name,
#            cmd,
#            f"Filtering '{src_path.name}' …",
#            f"{dst_path.name} created",
#        )

    ui.print_finished("filters")


########################################################################
#  Real-ESRGAN Video Upscaling Helper                                   #
#  • Works with CUDA GPU (torch‑based) or falls back to CPU / NCNN       #
#  • Ensures compatible torch/torchvision for Real‑ESRGAN (≤Python 3.11) #
########################################################################

_MIN_PY_OK = (3, 7)
_MAX_PY_OK = (3, 11)  # Real‑ESRGAN wheels not yet built for 3.12+


def _ensure_torch_packages(venv_python: Path, cuda_suffix: str = "cu118"):
    """Verify torch + torchvision imports. Install compatible versions if missing.

    For Python 3.12+, official wheels are not available. In that case we bail out
    early with a clear error message so the user can create a 3.11 (or lower)
    virtual environment.  Otherwise we attempt to install the reference wheels
    (torch 1.13.1 / torchvision 0.14.1) that Real‑ESRGAN upstream relies on.
    """
    py_ver = tuple(map(int, _run_py_out(venv_python, "import sys, json; print(json.dumps(sys.version_info[:3]))").strip("[]\n").split(", ")))  # e.g. (3, 12, 2)

    # Hard fail for unsupported Python versions (>3.11)
    if py_ver > _MAX_PY_OK:
        raise RuntimeError(
            f"Real‑ESRGAN wheels are not available for Python {py_ver[0]}.{py_ver[1]}. "
            "Please recreate your venv with Python 3.11 or lower (e.g. `python3.11 -m venv …`).")

    # Quick check: can we already import?
    if _run_py(venv_python, "from torchvision.transforms.functional import rgb_to_grayscale; import torch"):  # noqa
        return  # OK

    print("[upscale] Installing Real‑ESRGAN‑compatible torch/torchvision …")
    torch_ver = "1.13.1+" + cuda_suffix
    tv_ver    = "0.14.1+" + cuda_suffix

    # Uninstall incompatible bits
    _pip(venv_python, "uninstall", "-y", "torch", "torchvision")

    # Install reference wheels
    _pip(venv_python, "install", f"torch=={torch_ver}", f"torchvision=={tv_ver}",
         "--index-url", "https://download.pytorch.org/whl/cu118")

    # Verify again
    if not _run_py(venv_python, "from torchvision.transforms.functional import rgb_to_grayscale"):
        raise RuntimeError("Torchvision still incompatible after installation – aborting upscaling.")


# Helper wrappers ---------------------------------------------------------

def _run_py(venv_python: Path, code: str) -> bool:
    return subprocess.run([str(venv_python), "-c", code]).returncode == 0

def _run_py_out(venv_python: Path, code: str) -> str:
    res = subprocess.run([str(venv_python), "-c", code], capture_output=True, text=True)
    return res.stdout

def _pip(venv_python: Path, *args):
    subprocess.run([str(venv_python), "-m", "pip", *args], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------------



def upscale_real(args):
    files = ui.prepare_files("upscale", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return

    #upscale_factor = getattr(args, "factor", 4)
    upscale_factor = 4
    temp_dir = Path("temp_upscale")
    temp_dir.mkdir(exist_ok=True)

    # Paths
    esrgan_dir   = Path("~/syscripts/videoManager/Real-ESRGAN").expanduser()
    venv_python  = Path("~/syscripts/videoManager/venv/bin/python").expanduser()

    # Ensure env compatibility
    try:
        _ensure_torch_packages(venv_python)
    except RuntimeError as err:
        ui.print_error(str(err))
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    for file in files:
        path = Path(file)
        if not path.exists():
            ui.print_error(f"File not found: {path}")
            continue

        # 1️⃣ Extract frames ------------------------------------------------
        frame_pattern = temp_dir / "frame_%06d.png"
        extract_cmd   = ["ffmpeg", "-y", "-i", str(path), str(frame_pattern)]
        subprocess.run(extract_cmd, check=True)

        # 2️⃣ Real‑ESRGAN processing --------------------------------------
        upscale_out_dir = temp_dir / "upscaled"
        upscale_out_dir.mkdir(exist_ok=True)

        esrgan_cmd = [
            str(venv_python), "inference_realesrgan.py",
            "-n", f"realesrgan-x{upscale_factor}plus",
            "-i", str(temp_dir),
            "-o", str(upscale_out_dir)
        ]
        proc = subprocess.run(esrgan_cmd, cwd=esrgan_dir)
        if proc.returncode != 0:
            ui.print_error("Real‑ESRGAN failed – see log above.")
            continue

        # 3️⃣ Re‑encode with original FPS + audio -------------------------
        fps = _run_py_out(Path("/usr/bin/python3"), f"import ffmpeghelper, sys; print(ffmpeghelper.detect_fps('{path}'))") or "25"  # fallback helper optional

        upscaled_pattern = upscale_out_dir / "frame_%06d.png"
        output           = path.with_stem(f"{path.stem}_upscaled")
        cmd = [
            "ffmpeg", "-y", "-framerate", fps,
            "-i", str(upscaled_pattern),
            "-i", str(path),  # original audio
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(output)
        ]

        ui.run_ffmpeg_with_progress(path.name, cmd,
            f"upscaling '{path.name}' x{upscale_factor}",
            f"{path.name} x{upscale_factor} upscaled to {output}")

    shutil.rmtree(temp_dir, ignore_errors=True)
    ui.print_finished("upscaling")




# -----------------------------------------------------------------------------
# Farbkorrektur mit Vorschau‑Schleife (Kitty‑tauglich)
# -----------------------------------------------------------------------------
# Zwei intuitive Slider‑Werte (0‑100):
#     • warmth – Farbtemperatur   (0 = kühler, 50 = neutral, 100 = wärmer)
#     • tint   – Grün/Magenta‑Stich (0 = magenta, 50 = neutral, 100 = grün)
#
# Workflow
# --------
# 1. Benutzer gibt warmth/tint an (oder übergibt via CLI).
# 2. Aus der Mitte des ersten Videos werden zwei Vorschau‑Frames erstellt
#    (original & farbkorrigiert). Für große Videos wird vorher auf ≤640px
#    herunterskaliert, um IO zu minimieren.
# 3. Wenn das Programm in einem Kitty‑Terminal läuft, werden die beiden Bilder
#    direkt inline mit `kitty +kitten icat` angezeigt. In anderen Terminals
#    werden nur die Pfade ausgegeben.
# 4. Benutzer bestätigt die Einstellungen oder passt sie erneut an.
# 5. Nach Bestätigung werden alle Videos mit dem `colorbalance`‑Filter und den
#    gewünschten Parametern verarbeitet.
# -----------------------------------------------------------------------------

def color_correction(args):
    """Interaktive Farbkorrektur aller übergebenen Videos.

    *args* kann die Attribute `warmth` und `tint` enthalten (0‑100).
    """

    # ---------------------------------------------------------------------
    # Hilfsfunktionen
    # ---------------------------------------------------------------------
    def _is_kitty() -> bool:
        """True, wenn das Programm in einem Kitty‑Terminal läuft."""
        return os.environ.get("TERM", "").lower() == "xterm-kitty" or "KITTY_WINDOW_ID" in os.environ

    def _kitty_show(path: Path):
        """Bild inline im Kitty‑Terminal anzeigen (falls möglich)."""
        try:
            subprocess.run([
                "kitty", "+kitten", "icat", "--transfer-mode", "stream", str(path)
            ], check=True)
        except Exception:
            # Fällt silently zurück, wenn icat nicht verfügbar ist
            pass

    def _read_percent(attr: str, prompt: str, default: int = 50) -> int:
        val = getattr(args, attr, None)
        if val is None:
            val = input(prompt).strip() or str(default)
        try:
            val_int = int(val)
            if not (0 <= val_int <= 100):
                raise ValueError
            return val_int
        except ValueError:
            ui.print_error("Invalid input. Please enter a number between 0 and 100.")
            raise

    def _get_duration_seconds(path: Path) -> float:
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration", "-of",
                "default=nokey=1:noprint_wrappers=1", str(path)
            ])
            return float(out.decode().strip())
        except Exception:
            return 60.0

    def _extract_frame(path: Path, filter_chain: str | None, out_path: Path):
        duration = _get_duration_seconds(path)
        timestamp = duration / 2.0
        scale = "scale='min(640,iw)':-2"
        vf = scale if not filter_chain else f"{filter_chain},{scale}"
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{timestamp}", "-i", str(path),
            "-frames:v", "1", "-vf", vf, "-q:v", "2", str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def _percent_to_scale(v: int) -> float:
        return (v - 50) / 50.0

    def _build_colorbalance_filter(warmth: int, tint: int) -> str:
        ws = _percent_to_scale(warmth)
        ts = _percent_to_scale(tint)
        params = {}
        for suf in ("s", "m", "h"):
            if ws:
                params[f"r{suf}"] = ws
                params[f"b{suf}"] = -ws
            if ts:
                params[f"g{suf}"] = ts
                params.setdefault(f"r{suf}", 0.0)
                params.setdefault(f"b{suf}", 0.0)
                params[f"r{suf}"] += -ts * 0.2
                params[f"b{suf}"] += -ts * 0.2
        parts = [f"{k}={max(-1.0, min(1.0, v)):.3f}" for k, v in params.items() if abs(v) > 1e-3]
        return "colorbalance" if not parts else f"colorbalance={':'.join(parts)}"

    # ---------------------------------------------------------------------
    # 1) Eingangs‑Videos sammeln
    # ---------------------------------------------------------------------
    files = ui.prepare_files("color correction", args, defin.VIDEO_EXTENSIONS)
    if not files:
        return

    # ---------------------------------------------------------------------
    # 2) Interaktive Schleife
    # ---------------------------------------------------------------------
    while True:
        try:
            warmth_pct = _read_percent("warmth", "Warmth (0=cool, 50=neutral, 100=warm): ", 50)
            tint_pct   = _read_percent("tint",   "Tint   (0=magenta, 50=neutral, 100=green): ", 50)
        except Exception:
            return

        filter_str = _build_colorbalance_filter(warmth_pct, tint_pct)

        sample_path = Path(files[0])
        with tempfile.TemporaryDirectory(prefix="preview_") as tmp:
            orig_img = Path(tmp) / (sample_path.stem + "_orig.jpg")
            prev_img = Path(tmp) / (sample_path.stem + "_preview.jpg")
            try:
                _extract_frame(sample_path, None, orig_img)
                _extract_frame(sample_path, filter_str, prev_img)
            except subprocess.CalledProcessError:
                ui.print_error("Could not extract preview frames.")
                return

            if _is_kitty():
                ui.print_info("Showing preview inline (Kitty terminal detected)…")
                _kitty_show(orig_img)
                _kitty_show(prev_img)
            else:
                ui.print_info(f"Preview created: {orig_img} (original) | {prev_img} (corrected)")

            if input("Accept these settings? [y/n]: ").strip().lower().startswith("y"):
                break  # Einstellungen akzeptiert

    # ---------------------------------------------------------------------
    # 3) Verarbeitung
    # ---------------------------------------------------------------------
    for file in files:
        path = Path(file)
        output = path.with_stem(path.stem + "_colorbalanced")
        cmd = [
            "ffmpeg", "-y", "-i", str(path), "-vf", filter_str, "-c:a", "copy", str(output)
        ]
        ui.run_ffmpeg_with_progress(
            path.name, cmd, f"color correcting '{path.name}' ", f"{path.name} color corrected"
        )

    ui.print_finished("color corrected")





def filter_menu():
    print("\033[1;44m         🎬 Filter Menu for post processing      \033[0m")
    print(" ");
    foptions = [
        ("color-correction", "Adjustment of brightness, contrast, saturation and hue", "🔁"),
        ("noise-reduction", "Reduces the noise in the video", "🗜️"),
        ("stabilization", "Stabilizes blurred shots", "✂️"),
        ("upscale-real", "Upscale real videos with AI", "📐"),
        ("upscale-art", "Upscale anime or cartoons with AI", "📐")
    ]

    for i, (cmd, desc, symb) in enumerate(foptions, start=1):
        print(f"    [{i}] {symb} {cmd.ljust(16)} - {desc}")
    print("    [0] 🚪 Back")
    print(" ")
    choice = input("\033[1;33m     Select an action: \033[0m")
    if choice == "0":
        sys.exit(0)
    try:
        selected = foptions[int(choice)-1][0]
                # Call the corresponding function with default args

        if selected == "color-correction": color_correction(default_args)
        elif selected == "noise-reduction": filters(default_args)
        elif selected == "stabilization": record(default_args)
        elif selected == "upscale-real": upscale_real(default_args)
        elif selected == "upscale-art": convert(default_args)
        else: show_info()
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        sys.exit(1)




def show_info(subcommand=None):
    if subcommand == "screencast":
        infofile = defin.SCREENCAST_INFO
    else:
        base = os.path.splitext(os.path.abspath(__file__))[0]
        infofile = f"{base}.{subcommand}.info" if subcommand else f"{base}.info"
    subprocess.run(["bash", "-c", f"source ~/syscripts/functions/scripting.sh && show_infofile \"{infofile}\""])
    sys.exit(0)

def interactive_menu():
    print("\033[1;44m         🎬 VIDEO MANAGER - Tool for processing videos      \033[0m")
    print(" ");
    options = [
        ("convert", "Convert videos to another format", "🔁"),
        ("compress", "Compress video file", "🗜️"),
        ("trim", "Cut part of a video", "✂️"),
        ("crop-pad", "Crop or pad to target resolution", "📐"),
        ("interpolate", "Increase video framerate", "🌀"),
        ("screencast", "Start screen recording", "🖥️"),
        ("record", "Recording an external video via USB", "📹"),
        ("merge", "Merge audio, subtitle or multiple videos", "🔗"),
        ("extract", "Extract audio,subtitle,frame or video from video", "🧲"),
        ("filters", "A bunch of different video filters and effects", "🪄"),
        ("gif", "Convert to animated GIF", "🖼️"),
        ("metadata", "Show video metadata", "🏷️")
    ]

    for i, (cmd, desc, symb) in enumerate(options, start=1):
        print(f"    [{i}] {symb} {cmd.ljust(16)} - {desc}")
    print("    [0] 🚪 Exit")
    print(" ")
    choice = input("\033[1;33m     Select an action: \033[0m")
    if choice == "0":
        sys.exit(0)
    try:
        selected = options[int(choice)-1][0]
                # Call the corresponding function with default args
 #       default_args = argparse.Namespace(
 #           files=[], format=None, preset=None, resolution=None, framerate=None,
 #           factor=None, start=None, duration=None, quality=None,
 #           title=None, artist=None, comment=None, album=None, genre=None,
 #           date=None, track=None, composer=None, publisher=None, language=None,
 #           show=None, season_number=None, episode_id=None, network=None,
 #           director=None, production_year=None, keywords=None, text_top=None,
 #           text_bottom=None, font_size=None, offset=None, pause=None, output=None,
 #           burn_subtitle=None, subtitle_name=None, target_res=None,
 #           offset_x=None, offset_y=None, audio=None, subtitle=None, frame=None,
 #           video=None, codec=None, crf=None, threads=None
 #       )
        if selected == "convert": convert(default_args)
        elif selected == "screencast": screencast(default_args)
        elif selected == "record": record(default_args)
        elif selected == "interpolate": interpolate(default_args)
        elif selected == "trim": trim(default_args)
        elif selected == "compress": compress(default_args)
        elif selected == "gif": gif(default_args)
        elif selected == "extract": extract(default_args)
        elif selected == "metadata": metadata(default_args)
        elif selected == "merge": merge(default_args)
        elif selected == "filters": filter_menu()
        elif selected == "crop-pad": crop_pad(default_args)
        else: show_info()
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        sys.exit(1)

def main():

    parser = argparse.ArgumentParser(description="Video Manager CLI", usage="video <command> [<args>]", add_help=False)
    parser.add_argument("--help", "-h", action="store_true", help="Show this help message and exit")
    parser.add_argument("--list-tags", action="store_true", help="List all metadata tags and exit")
    parser.add_argument("--list-tags-json", action="store_true", help="Output all metadata tags as JSON and exit")

    tag_keys = [
        "title", "artist", "comment", "album", "genre", "date", "track",
        "composer", "publisher", "language", "show", "season_number", "episode_id",
        "network", "director", "production_year", "keywords"
    ]
    for tag in tag_keys:
        parser.add_argument(f"--list-tag-{tag}", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("files", nargs="*")
    convert_parser.add_argument("--format", "-f", choices=defin.FORMATS, default=None)
    convert_parser.add_argument("--preset", "-p", choices=defin.PRESETS.keys(), default=None)
    convert_parser.add_argument("--resolution", "-r", choices=defin.RESOLUTIONS, default=None)
    convert_parser.add_argument("--framerate", "-fr", choices=defin.FRAMERATES, default=None)
    convert_parser.add_argument("--codec","-c", choices=defin.CODECS, default=None)

    subparsers.add_parser("screencast")

    subparsers.add_parser("record")

    interp_parser = subparsers.add_parser("interpolate")
    interp_parser.add_argument("files", nargs="*")
    interp_parser.add_argument("--factor")

    trim_parser = subparsers.add_parser("trim")
    trim_parser.add_argument("files", nargs="*")
    trim_parser.add_argument("--start")
    trim_parser.add_argument("--duration")

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("files", nargs="*")
    compress_parser.add_argument("--quality")

    gif_parser = subparsers.add_parser("gif")
    gif_parser.add_argument("files", nargs="*")
    gif_parser.add_argument("--text-top", help="Add meme text at the top")
    gif_parser.add_argument("--text-bottom", help="Add meme text at the bottom")
    gif_parser.add_argument("--font-size", choices=["small", "medium", "large"], default="medium", help="Font size for meme text")



    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("files", nargs="*")
    extract_parser.add_argument("--audio")
    extract_parser.add_argument("--subtitle")
    extract_parser.add_argument("--format",choices=["srt", "ass", "vtt"],default="srt", help="Format for the subtitle extraction")
    extract_parser.add_argument("--frame")
    extract_parser.add_argument("--video")

    meta_parser = subparsers.add_parser("metadata")
    meta_parser.add_argument("files", nargs="*")

    # → Flags für das metadata-Kommando direkt beim metadata-Parser registrieren
    meta_parser.add_argument("--list-tags", action="store_true", help="List all metadata tags and exit")
    meta_parser.add_argument("--list-tags-json", action="store_true", help="Output all metadata tags as JSON and exit")
    for tag in tag_keys:
        meta_parser.add_argument(f"--{tag}")
        meta_parser.add_argument(f"--list-tag-{tag}", action="store_true")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("files", nargs="*")
    merge_parser.add_argument("--offset")
    merge_parser.add_argument("--pause")
    merge_parser.add_argument("--output")
    merge_parser.add_argument("--burn-subtitle")
    merge_parser.add_argument("--subtitle-name")
    merge_parser.add_argument("--target-res", choices=["no-scale","smallest","average","largest"], default="no-scale", help="Method for same resolutions") 

    croppad_parser = subparsers.add_parser("crop-pad")
    croppad_parser.add_argument("files", nargs="*")
    croppad_parser.add_argument("--resolution")
    croppad_parser.add_argument("--offset-x")
    croppad_parser.add_argument("--offset-y")
    croppad_parser.add_argument("--output")

    filters_parser = subparsers.add_parser("filters", description="One‑shot VHS cleanup using FFmpeg filters.")
    filters_parser.add_argument("input", help="input video file")
    filters_parser.add_argument("output", help="output video file")
    filters_parser.add_argument("--crf", type=int, default=18, help="libx264 CRF quality (lower is better)")
    filters_parser.add_argument("--preset", default="medium", help="x264 preset (veryfast–placebo)")
    filters_parser.add_argument("--threads", type=int, default=0, help="x264 threads (0 = auto)")



    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            show_info()
        elif len(sys.argv) > 2 and sys.argv[2] in ["--help", "-h"]:
            show_info(sys.argv[1])

    args, unknown = parser.parse_known_args()

    # Check for tag listing mode
    if args.command == "metadata" and (args.list_tags or args.list_tags_json or any(getattr(args, f"list_tag_{k}") for k in tag_keys)):
        if not args.files:
            sys.exit(1)
        path = Path(args.files[0])
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags",
            "-of", "json",
            str(path)
        ], capture_output=True, text=True)

        try:
            metadata_json = json.loads(result.stdout)
            tags = metadata_json.get("format", {}).get("tags", {})

            if args.list_tags_json:
                print(json.dumps(tags, indent=2))
            elif args.list_tags:
                for k, v in tags.items():
                    print(f"{k}: {v}")
            else:
                for k in tag_keys:
                    if getattr(args, f"list_tag_{k}", False):
                        if k in tags:
                            print(tags[k])
                        break
        except Exception:
            pass
        sys.exit(0)

    if not args.command:
        return interactive_menu()

    if args.command == "convert": convert(args)
    elif args.command == "screencast": screencast(args)
    elif args.command == "record": record(args)
    elif args.command == "interpolate": interpolate(args)
    elif args.command == "trim": trim(args)
    elif args.command == "compress": compress(args)
    elif args.command == "gif": gif(args)
    elif args.command == "extract": extract(args)
    elif args.command == "metadata": metadata(args)
    elif args.command == "merge": merge(args)
    elif args.command == "filters": filters(args)
    elif args.command == "crop-pad": crop_pad(args)
    else:
        show_info()


if __name__ == "__main__":
    main()

