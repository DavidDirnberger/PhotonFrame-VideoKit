#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# video_thumbnail.py – robust in-place thumbnail embedding for common
#                     multimedia containers (MP4/M4V/MOV/3GP, MKV/WebM)
# ---------------------------------------------------------------------------
# 2025-06-09 – Revision 5 (fix: don't shadow i18n._)
# ---------------------------------------------------------------------------
"""Öffentliche API: set_thumbnail(<videodatei|Path>)"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, cast

import consoleOutput as co
import definitions as defi
import helpers as he
import process_wrappers as pw
import userInteraction as ui

# local modules
from i18n import _
from imageDisplayer import ImageDisplayer

THUMBNAIL_SOURCE_SELECTION: Tuple[str, str] = ("extract_frame", "image_file")

displayer = ImageDisplayer()


# ---------------------------------------------------------------------------
# Subprozess-Helfer
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Führe *cmd* aus; wirf RuntimeError bei exit-Code ≠ 0."""
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "subprocess failed")
    return proc


def extract_thumbnail(file: Path, tmpdir: Path = defi.TMP_DIR) -> Optional[Path]:
    """
    Extrahiert das eingebettete Cover:
      • MP4/MOV: attached_pic-Video-Stream
      • MKV/WebM: (optional) falls MJPEG-Video vorhanden; Attachments weglassen
    """
    tmpdir.mkdir(parents=True, exist_ok=True)
    thumb_path = tmpdir / (file.stem + "_thumb.jpg")

    ext = file.suffix.lower()
    if ext in {".mp4", ".m4v", ".mov", ".3gp"}:
        ok, idx = mp4_has_cover(file)
        if not ok or idx is None:
            return None
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-stats_period",
            "0.5",
            "-i",
            str(file),
            "-map",
            f"0:{idx}",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(thumb_path),
        ]

        pw.run_ffmpeg_with_progress(
            file.name,
            cmd,
            _("extracting_cover"),
            _("cover_extracted"),
            output_file=thumb_path,
            BATCH_MODE=True,
            force_overwrite=True,
        )
        return thumb_path if thumb_path.exists() else None

    # Für MKV/WebM optional: MJPEG-Video-Stream als Vorschau nehmen
    if ext == ".mkv":
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=index,codec_name",
                "-of",
                "json",
                str(file),
            ],
            capture_output=True,
            text=True,
        )
        try:
            j = json.loads(probe.stdout or "{}")
            for s in j.get("streams", []):
                if s.get("codec_name") == "mjpeg":
                    idx = int(s["index"])
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-stats",
                        "-stats_period",
                        "0.5",
                        "-i",
                        str(file),
                        "-map",
                        f"0:{idx}",
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(thumb_path),
                    ]
                    pw.run_ffmpeg_with_progress(
                        file.name,
                        cmd,
                        _("extracting_cover"),
                        _("cover_extracted"),
                        output_file=thumb_path,
                        BATCH_MODE=True,
                        force_overwrite=True,
                    )
                    return thumb_path if thumb_path.exists() else None
        except Exception:
            pass

    return None


# ------------------------------------------------------------
# Thumbnail bei Kitty-Terminals anzeigen
# ------------------------------------------------------------


def show_thumbnail_if_possible(video: Path) -> None:
    thumb_file = extract_thumbnail(video)
    if thumb_file is not None and thumb_file.exists():
        try:
            displayer.show_image(str(thumb_file))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Verifikation
# ---------------------------------------------------------------------------


def mp4_has_cover(src: Path) -> Tuple[bool, Optional[int]]:
    """
    Robust: prüft alle Streams und findet jeden Video-Stream mit disposition.attached_pic == 1.
    """
    try:
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(src),
            ],
            capture_output=True,
            text=True,
        )
        streams = json.loads(pr.stdout or "{}").get("streams", [])
        for s in streams:
            if (
                s.get("codec_type") == "video"
                and s.get("disposition", {}).get("attached_pic") == 1
            ):
                return True, int(s.get("index"))
    except Exception:
        pass
    return False, None


def extract_mp4_cover(
    src: Path,
    idx: int,
) -> Path:
    """
    Extrahiert den attached_pic-Stream `idx` als JPEG‐Datei und gibt den Pfad zurück.
    """
    defi.TMP_DIR.mkdir(parents=True, exist_ok=True)
    cover_tmp = defi.TMP_DIR / f"{uuid.uuid4().hex}_cover.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-map",
            f"0:{idx}",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(cover_tmp),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cover_tmp


def _mkv_has_cover(path: Path) -> bool:
    """Return *True* if *path* contains a cover image.

    Erfasst:
      • Matroska-Attachments (image/*)
      • Attachment-Streams (image/*)
      • MJPEG-Videostreams (codec_name == mjpeg)
    """
    try:
        res = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "stream=index,codec_type,codec_name,disposition:stream_tags",
                str(path),
            ]
        )
        info = json.loads(res.stdout or "{}")
        for s in info.get("streams", []):
            if s.get("codec_type") == "attachment":
                tags = s.get("tags", {})
                if str(tags.get("mimetype", "")).startswith("image/"):
                    return True
            if s.get("codec_type") == "video" and s.get("codec_name") == "mjpeg":
                return True
    except Exception:
        return False
    return False


def _parse_frame_time_to_seconds(raw: str, duration_s: Optional[float]) -> float:
    if raw is None:
        raise ValueError("empty")
    s = raw.strip().lower().replace(" ", "")
    if s in {"mid", "middle", "center", "centre"}:
        if duration_s is None:
            raise ValueError("cannot resolve 'middle' without duration")
        return duration_s / 2.0
    if s.endswith("%") or s.endswith("p"):
        if duration_s is None:
            raise ValueError("percentage requires known duration")
        val = float(s[:-1].replace(",", "."))
        if not (0.0 <= val <= 100.0):
            raise ValueError(_("read_percent_error"))
        return duration_s * (val / 100.0)
    if s.endswith("s"):
        return float(s[:-1].replace(",", "."))
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            hh, mm, ss = parts
        elif len(parts) == 2:
            hh, mm, ss = "0", parts[0], parts[1]
        else:
            hh, mm, ss = "0", "0", parts[0]
        try:
            h = int(hh)
            m = int(mm)
            sec = float(ss.replace(",", "."))
        except ValueError:
            raise ValueError(_("invalid_time_input"))
        if not (0 <= m < 60 and 0 <= sec < 60):
            raise ValueError("MM/SS must be in [0,60)")
        return h * 3600 + m * 60 + sec
    return float(s.replace(",", "."))


def _extract_frame_to_jpeg(
    video: Path, time_spec: str, out: Path, BATCH_MODE: bool = False
) -> Optional[Path]:
    """Exportiert ein Einzelbild wie in 'extract' (präzise, Qualität)."""
    dur = he.probe_duration_sec(video)
    try:
        t = _parse_frame_time_to_seconds(time_spec, dur)
    except ValueError:
        co.print_error(_("invalid_time_input"))
        return None
    t = he.clamp_seek_time(t, dur)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    pw.run_ffmpeg_with_progress(
        video.name,
        cmd,
        _("extract_frame"),
        _("frame_extracted"),
        output_file=out,
        BATCH_MODE=BATCH_MODE,
        force_overwrite=True,
    )
    return out if out.exists() else None


def _scale_to_jpeg(
    src_img: Path, out: Path, max_w: int = 600, BATCH_MODE: bool = False
) -> Optional[Path]:
    """Skaliert ein Bild nach JPEG (max Breite), hochwertig, mit Progress."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-i",
        str(src_img),
        "-vf",
        f"scale='min(iw,{max_w})':-2",
        "-q:v",
        "2",
        str(out),
    ]
    pw.run_ffmpeg_with_progress(
        src_img.name,
        cmd,
        _("cover_image_progress"),
        _("cover_image_done"),
        output_file=out,
        BATCH_MODE=BATCH_MODE,
        force_overwrite=True,
    )
    if out.exists():
        return out
    return None


# ---------------------------------------------------------------------------
# Embedding-Handler
# ---------------------------------------------------------------------------

Handler = Callable[[Path, Path, Path], None]


def _clear_mp4_thumbnails(file: Path, BATCH_MODE: bool = False) -> bool:
    """
    Entfernt *alle* attached_pic-Streams sicher und versions-robust,
    indem per ffprobe deren Indizes ermittelt und dann gezielt herausgemappt werden.
    Gibt True zurück, wenn neu geschrieben wurde (d. h. mindestens ein Cover existierte).
    """
    try:
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(file),
            ],
            capture_output=True,
            text=True,
        )
        streams = json.loads(pr.stdout or "{}").get("streams", [])
        cover_idx = [
            int(s["index"])
            for s in streams
            if s.get("codec_type") == "video"
            and s.get("disposition", {}).get("attached_pic") == 1
        ]
    except Exception:
        cover_idx = []

    co.print_info(_("cleaning_mp4_stream"))
    tmp = file.with_name(file.stem + ".clean" + file.suffix)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-i",
        str(file),
        "-map",
        "0",
    ]
    for idx in cover_idx:
        cmd += ["-map", f"-0:{idx}"]
    cmd += ["-c", "copy", str(tmp)]

    out_path = cast(
        Optional[Path],
        pw.run_ffmpeg_with_progress(
            file.name,
            cmd,
            _("removing_cover"),
            _("cover_removed"),
            output_file=tmp,
            BATCH_MODE=BATCH_MODE,
            force_overwrite=True,
        ),
    )

    if out_path and out_path.exists():
        shutil.move(str(out_path), str(file))
        co.print_success(_("mp4_cover_removed"))
        return True

    co.print_warning(_("could_not_remove_mp4_cover"))
    return False


def _clear_mkv_attachments(file: Path, BATCH_MODE: bool = False) -> bool:
    """
    Entfernt *nur* Matroska-Attachments (z.B. alte cover.jpg / Fonts).
    Video-/Audio-/Subtitle-Streams bleiben unangetastet.
    """
    try:
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "stream=index,codec_type",
                str(file),
            ],
            capture_output=True,
            text=True,
        )
        streams = json.loads(pr.stdout or "{}").get("streams", [])
        has_attachments = any(s.get("codec_type") == "attachment" for s in streams)
        if not has_attachments:
            return False

        tmp = file.with_name(file.stem + ".clean" + file.suffix)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-stats_period",
            "0.5",
            "-i",
            str(file),
            "-map",
            "0",
            "-map",
            "-0:t",
            "-c",
            "copy",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            str(tmp),
        ]

        pw.run_ffmpeg_with_progress(
            file.name,
            cmd,
            _("removing_cover_attachments"),
            _("removed"),
            output_file=tmp,
            BATCH_MODE=BATCH_MODE,
            force_overwrite=True,
        )

        if not tmp.exists():
            return False

        shutil.move(str(tmp), str(file))
        return True

    except Exception as e:
        co.print_warning(f"MKV cleanup failed: {e}")
        return False


def delete_thumbnail(file: str | Path, BATCH_MODE: bool = False) -> bool:
    """Öffentliche API: Thumbnail aus Container entfernen (falls vorhanden)."""
    p = Path(file)
    ext = p.suffix.lower()
    if ext in {".mp4", ".m4v", ".mov", ".3gp"}:
        ok = _clear_mp4_thumbnails(p, BATCH_MODE=BATCH_MODE)
    elif ext in {".mkv", ".webm"}:
        ok = _clear_mkv_attachments(p, BATCH_MODE=BATCH_MODE)
    else:
        co.print_info(_("container_not_support_cover"))
        return False
    co.print_success(_("thumbnail_removed") if ok else _("no_thumbnail_found"))
    return ok


def embed_mp4(src: Path, cover: Path, dst: Path, BATCH_MODE: bool = False) -> None:
    """
    Robustes Einbetten:
      • ermittelt alle *nicht*-attached_pic Videostreams → 'main_videos'
      • mappt diese + alle Audio/Subtitle/Data Streams explizit
      • hängt das Cover als letzten Video-Stream an (wird dann v:<len(main_videos)>)
      • setzt 'mjpeg' + 'attached_pic' exakt auf diesen Zielstream
    """
    defi.TMP_DIR.mkdir(parents=True, exist_ok=True)
    # 1) Cover vorbereiten (Skalierung + Vorschau)
    processed = defi.TMP_DIR / "tmp_cover.jpg"
    if _scale_to_jpeg(cover, processed, max_w=600, BATCH_MODE=BATCH_MODE) is None:
        raise RuntimeError(_("failed_to_prepare_cover"))

    # 2) vorhandene Cover-Streams entfernen (in-place, robust)
    _clear_mp4_thumbnails(src, BATCH_MODE=BATCH_MODE)

    # 3) Streams der Quelle untersuchen und die gewünschten Indizes einsammeln
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(src)],
        capture_output=True,
        text=True,
    )
    info = json.loads(pr.stdout or "{}")
    streams = info.get("streams", [])

    main_video_idx = [
        int(s["index"])
        for s in streams
        if s.get("codec_type") == "video"
        and s.get("disposition", {}).get("attached_pic") != 1
    ]
    audio_idx = [int(s["index"]) for s in streams if s.get("codec_type") == "audio"]
    sub_idx = [int(s["index"]) for s in streams if s.get("codec_type") == "subtitle"]
    data_idx = [int(s["index"]) for s in streams if s.get("codec_type") == "data"]

    # 4) Deterministische Mapping-Reihenfolge
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-i",
        str(src),
        "-i",
        str(processed),
    ]
    for idx in main_video_idx:
        cmd += ["-map", f"0:{idx}"]
    for idx in audio_idx:
        cmd += ["-map", f"0:{idx}"]
    for idx in sub_idx:
        cmd += ["-map", f"0:{idx}"]
    for idx in data_idx:
        cmd += ["-map", f"0:{idx}"]

    # Cover anhängen
    cmd += ["-map", "1:v:0"]

    cover_v_out_index = len(main_video_idx)  # 0-basiert in der v:-Menge
    cmd += [
        "-c",
        "copy",
        f"-c:v:{cover_v_out_index}",
        "mjpeg",
        f"-disposition:v:{cover_v_out_index}",
        "attached_pic",
        f"-metadata:s:v:{cover_v_out_index}",
        "title=Cover",
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    out_path = cast(
        Optional[Path],
        pw.run_ffmpeg_with_progress(
            src.name,
            cmd,
            _("embedding_thumbnail_progress"),
            _("embedding_thumbnail_done"),
            output_file=dst,
            BATCH_MODE=BATCH_MODE,
            force_overwrite=True,
        ),
    )
    if not out_path or not out_path.exists():
        raise RuntimeError(_("embedding_failed"))

    # 6) Verifizieren
    assert isinstance(out_path, Path)
    ok, _cover_idx = mp4_has_cover(out_path)
    if not ok:
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_streams",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            co.print_warning(_("thumbnail_verification_failed_ffprobe"))
            co.print_line((probe.stdout or "")[:2000])
        except Exception:
            pass
        raise RuntimeError(_("thumbnail_verification_failed_no_stream"))


def _embed_mkv(src: Path, cover: Path, dst: Path, BATCH_MODE: bool = False) -> None:
    """Fügt *cover* als Matroska-Attachment ein, verifiziert."""
    defi.TMP_DIR.mkdir(parents=True, exist_ok=True)
    processed = defi.TMP_DIR / f"{uuid.uuid4().hex}_cover.jpg"
    if not _scale_to_jpeg(cover, processed, max_w=600, BATCH_MODE=BATCH_MODE):
        raise RuntimeError(_("cover_preparation_failed"))

    _clear_mkv_attachments(src, BATCH_MODE=BATCH_MODE)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-i",
        str(src),
        "-map",
        "0",
        "-attach",
        str(processed),
        "-metadata:s:t",
        "mimetype=image/jpeg",
        "-metadata:s:t",
        "filename=cover.jpg",
        "-c",
        "copy",
        str(dst),
    ]
    pw.run_ffmpeg_with_progress(
        src.name,
        cmd,
        _("embedding_thumbnail_progress"),
        _("embedding_thumbnail_done"),
        output_file=dst,
        BATCH_MODE=BATCH_MODE,
        force_overwrite=True,
    )
    if not _mkv_has_cover(dst):
        raise RuntimeError(_("thumbnail_verification_failed_no_attachment"))


_HANDLERS: Dict[str, Handler] = {
    ".mp4": embed_mp4,
    ".m4v": embed_mp4,
    ".3gp": embed_mp4,
    ".mkv": _embed_mkv,
}


# ------------------------------------------------------------
# Thumbnail-Check: universell für MP4/MKV + Side-Car-Poster
# ------------------------------------------------------------


def check_thumbnail(file: Path, silent: bool = False) -> bool:
    ext = file.suffix.lower()
    has = False
    if ext in {".mp4", ".m4v", ".3gp"}:
        has, _cover_idx = mp4_has_cover(file)
    elif ext == ".mkv":
        has = _mkv_has_cover(file)
    else:
        if not silent:
            co.print_info(_("container_not_support_cover"))
        return False  # immer bool zurückgeben

    if silent:
        return has

    if has:
        co.print_info(f"✅ {_('thumbnail_exists')}")
        show_thumbnail_if_possible(file)
    else:
        co.print_info(f"❌ {_('thumbnail_not_set')}")
    return has


# ---------------------------------------------------------------------------
# Öffentliche Hauptfunktion
# ---------------------------------------------------------------------------


def set_thumbnail(
    file: str | Path, value: Optional[str] = None, BATCH_MODE: bool = False
) -> None:
    """
    Setzt ein Thumbnail.
    - value ist None  → interaktiv (Bild wählen oder Framezeit eingeben)
    - value ist Pfad  → dieses Bild verwenden
    - value ist Zeit  → Frame extrahieren (z. B. '01:23', '12.5s', '33%', 'middle')
    Zeigt vor dem Einbetten eine Vorschau.
    """
    src = Path(file)
    if not src.is_file():
        co.print_error(_("file_not_found"))
        return

    ext = src.suffix.lower()
    handler = _HANDLERS.get(ext)

    # Unsupported-/Sidecar-Logik
    unsupported_container = (ext in {".mov", ".avi", ".webm"}) or (handler is None)
    if unsupported_container:
        poster = src.with_suffix(".jpg")

        if BATCH_MODE and value is None:
            co.print_info(_("container_not_support_cover"))
            return

        if value is None:
            _sidecar_from_user(src, poster)
            return

        cand = Path(value)
        if cand.exists() and cand.is_file():
            try:
                shutil.copy2(cand, poster)
                co.print_success(_("side_car_poster_saved").format(file=poster.name))
            except Exception as e:
                co.print_error(f"Failed to copy poster: {e}")
            return
        else:
            # value ist Zeitangabe → Frame extrahieren
            tmp_img = defi.TMP_DIR / f"{src.stem}_cover.jpg"
            got = _extract_frame_to_jpeg(src, value, tmp_img, BATCH_MODE=BATCH_MODE)
            if not got:
                co.print_error(_("frame_extraction_failed"))
                return
            try:
                shutil.copy2(got, poster)
                co.print_success(_("side_car_poster_saved").format(file=poster.name))
            except Exception as e:
                co.print_error(f"Failed to create sidecar: {e}")
            return

    # Unterstützte Container
    img: Optional[Path] = None

    while True:
        # Quelle bestimmen
        if value is None:
            if BATCH_MODE:
                break
            selection_keys = list(THUMBNAIL_SOURCE_SELECTION)
            sel_options = [_(k) for k in selection_keys]
            choice = ui.ask_user(
                _("select_thumbnail_source"),
                selection_keys,
                display_labels=sel_options,
                default=0,
            )
            if choice == selection_keys[1]:
                imgs = ui.select_files_interactively([".jpg", ".jpeg", ".png"])
                if not imgs:
                    co.print_error(_("no_image_selected_canceled"))
                    return
                img = Path(imgs[0])
            else:
                ts_raw = input(co.return_promt(_("enter_frame_time")) + ": ").strip()
                tmp_img = defi.TMP_DIR / f"{src.stem}_cover.jpg"
                got = _extract_frame_to_jpeg(
                    src, ts_raw, tmp_img, BATCH_MODE=BATCH_MODE
                )
                if not got:
                    co.print_error(_("frame_extraction_failed"))
                    return
                img = got
        else:
            cand = Path(value)
            if cand.exists() and cand.is_file():
                img = cand
            else:
                tmp_img = defi.TMP_DIR / f"{src.stem}_cover.jpg"
                got = _extract_frame_to_jpeg(src, value, tmp_img, BATCH_MODE=BATCH_MODE)
                if not got:
                    co.print_error(_("frame_extraction_failed"))
                    return
                img = got

        if BATCH_MODE:
            break

        # Vorschau
        try:
            displayer.show_image(str(img))
            ans = ui.ask_yes_no(_("apply_thumbnail"), default=True, back_option=False)
            if ans:
                break
            else:
                continue
        except Exception:
            break

    if img is None or not img.is_file():
        return

    # Sicheres In-Place-Update (Backup)
    backup = src.with_suffix(src.suffix + ".bak")
    shutil.copy2(src, backup)

    # Temp-Ausgabedatei verwalten
    fd: Optional[int] = None
    dst: Optional[Path] = None

    try:
        # Erzeuge Tempziel in Systemtemp (/tmp). Wenn du lieber defi.TMP_DIR willst:
        # fd, tmp_name = tempfile.mkstemp(suffix=src.suffix, dir=str(defi.TMP_DIR))
        fd, tmp_name = tempfile.mkstemp(suffix=src.suffix)
        os.close(fd)  # WICHTIG: Descriptor sofort schließen
        dst = Path(tmp_name)  # NICHT vorab unlinken – ffmpeg soll hierhin schreiben

        if ext in {".mp4", ".m4v", ".mov", ".3gp"}:
            embed_mp4(src, img, dst, BATCH_MODE=BATCH_MODE)
        else:
            _embed_mkv(src, img, dst, BATCH_MODE=BATCH_MODE)

        shutil.move(
            str(dst), str(src)
        )  # verschiebt Ergebnis → Temp-Datei existiert danach nicht mehr
        backup.unlink(missing_ok=True)
        if not BATCH_MODE:
            co.print_success(_("thumbnail_verified"))
            show_thumbnail_if_possible(src)
    except Exception as err:
        co.print_error(_("embedding_failed_restore").format(error=err))
        shutil.copy2(backup, src)
    finally:
        backup.unlink(missing_ok=True)
        # Cleanup: Temp-Output aus mkstemp sicher entfernen (falls noch vorhanden)
        try:
            if dst is not None and dst.exists():
                dst.unlink(missing_ok=True)
        except Exception:
            pass
        # Cleanup: evtl. temporär extrahiertes Cover in defi.TMP_DIR löschen
        try:
            if (
                img is not None
                and img.parent == defi.TMP_DIR
                and img.name.endswith("_cover.jpg")
            ):
                img.unlink(missing_ok=True)
        except Exception:
            pass


def show_thumbnail(file: str | Path) -> None:
    """Flag-tauglich: zeigt vorhandenes Thumbnail (falls extrahierbar)."""
    p = Path(file)
    if not p.exists():
        co.print_error(_("file_not_found").format(file=p.name))
        return
    show_thumbnail_if_possible(p)


# ---------------------------------------------------------------------------
# Fallback für nicht unterstützte Container
# ---------------------------------------------------------------------------


def _sidecar_from_user(src: Path, poster: Path) -> None:
    imgs = ui.select_files_interactively([".jpg", ".jpeg", ".png"])
    if not imgs:
        co.print_error(_("no_image_selected_no_change"))
        return
    shutil.copy2(imgs[0], poster)
    co.print_success(_("side_car_poster_saved").format(file=poster.name))
