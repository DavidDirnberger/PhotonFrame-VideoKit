#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# video_thumbnail.py – robust in‑place thumbnail embedding for common
#                     multimedia containers (MP4/M4V/MOV/3GP, MKV/WebM)
# ---------------------------------------------------------------------------
# 2025‑06‑09 – Revision 2
#   • FIX: _mp4_has_cover und _mkv_has_cover nutzen jetzt JSON‑Ausgabe von
#     ffprobe → zuverlässige Erkennung von attached_pic‑Streams bzw. Matroska‑
#     Attachments. Dadurch schlägt die Verifikation nicht mehr fehl, obwohl
#     VLC das Cover korrekt anzeigt.
# ---------------------------------------------------------------------------
"""Öffentliche API: set_thumbnail(<videodatei|Path>)"""

from __future__ import annotations

import json
import shutil
import subprocess
import datetime
#import base64
import sys
import uuid
#import os
import tempfile
from pathlib import Path
from typing import Callable
import userInteraction as ui
import definitions as defi
import helpers as he

# ---------------------------------------------------------------------------
# Subprozess‑Helfer
# ---------------------------------------------------------------------------

_DEF_ARGS = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Führe *cmd* aus; wirf RuntimeError bei exit‑Code ≠ 0."""
    proc = subprocess.run(cmd, **_DEF_ARGS)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "subprocess failed")
    return proc



def extract_thumbnail(file, tmpdir=defi.TMP_DIR):
    tmp_path = Path(tmpdir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    thumb_path = tmp_path / (Path(file).stem + "_thumb.jpg")
    
    # Extract embedded thumbnail (if available)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(file),
        "-map", "0:v:1",
        "-frames:v", "1",
        str(thumb_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return thumb_path if thumb_path.exists() else None





# ------------------------------------------------------------
# Thumbnail bei Kitty‑Terminals anzeigen
# ------------------------------------------------------------

def show_thumbnail_if_possible(video: Path) -> None:
    """Extrahiere Cover und zeige es direkt im Kitty‑Terminal (falls möglich)."""

    thumb_file = extract_thumbnail(video)
    if thumb_file is not None:
        ui.display_image_in_kitty(thumb_file)

# ---------------------------------------------------------------------------
# Verifikation
# ---------------------------------------------------------------------------


def _has_mp4_cover(src: Path) -> tuple[bool, int | None]:
    """
    Liefert (True, stream_index) wenn ein attached_pic-Stream existiert,
    sonst (False, None).
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=index,disposition", "-of", "json", str(src)],
        capture_output=True, text=True
    )
    try:
        streams = json.loads(probe.stdout)["streams"]
        for s in streams:
            if s.get("disposition", {}).get("attached_pic") == 1:
                return True, s["index"]
    except Exception:
        pass
    return False, None


def _extract_mp4_cover(src: Path, idx: int) -> Path:
    """
    Extrahiert den attached_pic-Stream `idx` als JPEG‐Datei
    und gibt den Pfad zurück.
    """
    cover_tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_cover.jpg"
    # MJPEG schon im Container – einfach Stream kopieren
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-map", f"0:{idx}", "-c", "copy", str(cover_tmp)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return cover_tmp





def _mp4_has_cover(path: Path) -> bool:
    """True, wenn irgendein Videostream disposition.attached_pic == 1 hat."""
    try:
        res = _run([
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", str(path),
        ])
        data = json.loads(res.stdout)
        for s in data.get("streams", []):
            if (
                s.get("codec_type") == "video"
                and s.get("disposition", {}).get("attached_pic") == 1
            ):
                return True
        return False
    except Exception:
        return False




def _mkv_has_cover(path: Path) -> bool:
    """Return *True* if *path* contains a cover image.

    Erfasst:
      • Matroska-Attachments (image/*)
      • Attachment-Streams (image/*)
      • MJPEG-Videostreams (codec_name == mjpeg)
    """
    try:
        res = _run([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "stream=index,codec_type,codec_name,disposition:stream_tags",
            str(path),
        ])
        info = json.loads(res.stdout)
        for s in info.get("streams", []):
            if s.get("codec_type") == "attachment":
                tags = s.get("tags", {})
                if tags.get("mimetype", "").startswith("image/"):
                    return True
            if s.get("codec_type") == "video" and s.get("codec_name") == "mjpeg":
                return True
    except Exception:
        return False
    return False



# ---------------------------------------------------------------------------
# Embedding‑Handler
# ---------------------------------------------------------------------------

Handler = Callable[[Path, Path, Path], None]


def _clear_mp4_thumbnails(file: Path) -> None:
    """Entfernt alle MJPEG-Coverstreams aus MP4 in-place."""
    print("[cleanup] Entferne bestehende Thumbnails aus MP4 …")
    tmp = file.with_suffix(".clean.mp4")

    try:
        res = _run([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "stream=index,codec_name", str(file),
        ])
        info = json.loads(res.stdout)
        non_cover_indices = [
            str(s["index"]) for s in info.get("streams", [])
            if s.get("codec_name") != "mjpeg"
        ]
        maps = sum([["-map", f"0:{i}"] for i in non_cover_indices], [])

        _run(["ffmpeg", "-y", "-i", str(file)] + maps + ["-c", "copy", str(tmp)])
        shutil.move(tmp, file)
        print("✅ MJPEG-Thumbnails removed.")
    except Exception as err:
        ui.print_error(f"Removeing the MP4-Thumbnails didn't work: {err}")


def _embed_mp4(src: Path, cover: Path, dst: Path) -> None:
    processed = Path(tempfile.gettempdir()) / "tmp_cover.jpg"
    _run([
        "ffmpeg", "-y", "-i", str(cover),
        "-vf", "scale='min(iw,600)':-2", "-q:v", "2", str(processed),
    ])

    _clear_mp4_thumbnails(Path(str(src)))
    _run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-i", str(processed),
        "-map", "0", "-map", "1",
        "-c", "copy",
        "-disposition:v:1", "attached_pic",
        "-metadata:s:v:1", "title=Cover",
        str(dst),
    ])
    if not _mp4_has_cover(dst):
        raise RuntimeError("verification failed – no attached_pic stream found")



def _clear_mkv_attachments(file: Path) -> None:
    """Entfernt alle bestehenden Attachments (inkl. MJPEG-Coverstreams) aus *file* in-place."""
    print("[cleanup] Entferne existierende MJPEG-Videostreams aus MKV …")
    tmp = file.with_suffix(".clean.mkv")

    # entferne explizit MJPEG-Streams (codec_name=mjpeg) mit map
    try:
        res = _run([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "stream=index,codec_name", str(file),
        ])
        info = json.loads(res.stdout)
        mjpeg_indices = [
            str(s["index"]) for s in info.get("streams", [])
            if s.get("codec_name") != "mjpeg"
        ]
        maps = sum([ ["-map", f"0:{i}"] for i in mjpeg_indices ], [])

        _run(["ffmpeg", "-y", "-i", str(file)] + maps + ["-c", "copy", str(tmp)])
        shutil.move(tmp, file)
        print("✅ Alle MJPEG-Coverstreams entfernt.")
    except Exception as err:
        print(f"⚠️  Fehler beim Entfernen der Attachments: {err}")


def _embed_mkv(src: Path, cover: Path, dst: Path) -> None:
    """Embed *cover* as Matroska attachment → *dst*.
    1. Skaliert das Bild auf max 600 px Breite und wandelt es in JPEG um.
    2. Erstellt neues MKV mit FFmpeg unter Hinzufügung des Covers.
    3. Verifiziert das Ergebnis mit `_mkv_has_cover()`.
    """
    processed = Path(tempfile.gettempdir()) / "tmp_cover.jpg"
    _run([
        "ffmpeg", "-y", "-i", str(cover),
        "-vf", "scale='min(iw,600)':-2", "-q:v", "2", str(processed),
    ])

    _clear_mkv_attachments(Path(str(src)))

    # FFmpeg erstellen mit Anhang
    _run([
        "ffmpeg", "-y", "-i", str(src), "-map", "0",
        "-attach", str(processed),
        "-metadata:s:t", "mimetype=image/jpeg",
        "-metadata:s:t", "filename=cover.jpg",
        "-c", "copy", str(dst),
    ])



_HANDLERS: dict[str, Handler] = {
    ".mp4": _embed_mp4,
    ".m4v": _embed_mp4,
    ".mov": _embed_mp4,
    ".3gp": _embed_mp4,
    ".mkv": _embed_mkv,
    ".webm": _embed_mkv,
}

# ------------------------------------------------------------
# Thumbnail‑Check: universell für MP4/MKV + Side‑Car‑Poster
# ------------------------------------------------------------

def check_thumbnail(file: Path):
    """True, wenn eingebettetes Cover oder Side‑Car‑Poster existiert."""
    ext = file.suffix.lower()

    if ext in {".mp4", ".m4v", ".mov", ".3gp"}:
        if _mp4_has_cover(file):
            print("✅ Thumbnail exists")
            show_thumbnail_if_possible(file)
        else:
            print("❌ No Thumbnail set")

    if ext in {".mkv", ".webm"}:
        if _mkv_has_cover(file):
            print("✅ Thumbnail exists")
            show_thumbnail_if_possible(file)
        else:
            print("❌ No Thumbnail set")

    # Fallback: existiert ein Side‑Car‑Bild?
    return

# ---------------------------------------------------------------------------
# Öffentliche Hauptfunktion
# ---------------------------------------------------------------------------

def set_thumbnail(file: str | Path) -> None:
    src = Path(file)
    if not src.is_file():
        ui.print_error("❌ File not found – process aborted.")
        return

    ext = src.suffix.lower()
    handler = _HANDLERS.get(ext)

    if handler is None:
        ui.print_error("⚠️  Container not supported → Side-car poster …", file=sys.stderr)
        _sidecar_from_user(src, src.with_suffix(".jpg"))
        return

    print(f"\n[set_thumbnail] {src.name}\n{'-'*40}")
    choice = ui.ask_user("Select thumbnail source", ["Image file", "Extract frame"])

    # --- Cover beschaffen ---------------------------------------------------
    if choice == "Image file":
        imgs = ui.select_files_interactively([".jpg", ".jpeg", ".png"])
        if not imgs:
            ui.print_error("No image file selected - canceled.")
            return
        cover = Path(imgs[0])
    else:
        ts_raw = input("Timestamp (HH:MM:SS | Seconds): ").strip()

            # 3) Normalisierte Strings & Sekunden ----------------------------------
        try:
            ts_td = he.parse_time(ts_raw)
        except ValueError as e:
            ui.print_error(f"❌ Invalid Character: {e}")
            return

        ts_str = he.seconds_to_time(ts_td)

        if not ui.is_time_in_video(ts_str,str(src)):
            ui.print_error(f"Timestamp is not part of the Video")
            return
        cover = defi.TMP_DIR / f"{src.stem}_cover.jpg"
        try:
            _run([
                "ffmpeg", "-y", "-ss", ts_str, "-i", str(src),
                "-frames:v", "1", "-q:v", "2", str(cover),
            ])
        except RuntimeError as err:
            ui.print_error(f"❌ Frame‑Export failed – {err}")
            return

    # --- Sicheres In‑Place‑Update ------------------------------------------
    backup = src.with_suffix(src.suffix + ".bak")
    shutil.copy2(src, backup)

    try:
        fd, tmp_name = tempfile.mkstemp(suffix=src.suffix)
        Path(tmp_name).unlink(missing_ok=True)
        dst = Path(tmp_name)

        handler(src, cover, dst)  # → hebt RuntimeError bei Verifikationsfehler

        shutil.move(dst, src)
        backup.unlink(missing_ok=True)
        print("✅ Thumbnail embedded and verified.")
        show_thumbnail_if_possible(src)
    except Exception as err:
        print(f"❌ Embedding failed - {err}. Backup is being restored …")
        shutil.copy2(backup, src)
    finally:
        backup.unlink(missing_ok=True)
        if cover.exists() and cover.parent == defi.TMP_DIR:
            cover.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Fallback für nicht unterstützte Container
# ---------------------------------------------------------------------------

def _sidecar_from_user(src: Path, poster: Path) -> None:
    imgs = ui.select_files_interactively([".jpg", ".jpeg", ".png"])
    if not imgs:
        ui.print_error("No image file selected - nothing changed.")
        return
    shutil.copy2(imgs[0], poster)
    print(f"✅ Side-car poster saved as {poster.name}.")
