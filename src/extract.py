#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import consoleOutput as co
import definitions as defin
import helpers as he
import process_wrappers as pw
import userInteraction as ui
from ffmpeg_perf import autotune_final_cmd

# local modules
from i18n import _, tr
from imageDisplayer import ImageDisplayer

displayer = ImageDisplayer()


# ---------------- Dataclass ----------------
@dataclass
class ExtractArgs:
    files: List[str] = field(default_factory=list)
    frame: Optional[str] = (
        None  # None=aus, ''=Mitte/Frage, sonst 'HH:MM:SS' oder Sekunden
    )
    audio: Optional[bool] = None  # True=alle Audio-Tracks extrahieren (MP3)
    video: Optional[bool] = None  # True=Video-only (Audio/Subs entfernen)
    subtitle: Optional[str] = (
        None  # None=aus, ''=alle, sonst Teilstring-Filter für title-Tag
    )
    format: Optional[str] = None  # srt/ass/vtt ODER Bild-Format (png/jpg/…)
    output: Optional[str] = None  # optional: Zielpfad/Name


#    languages: Optional[List[str]] | None = None


# ---------------- Helper ----------------


def _no_flags_selected(args: Any) -> bool:
    return not (
        bool(getattr(args, "audio", False))
        or (getattr(args, "subtitle", None) is not None)
        or (getattr(args, "frame", None) is not None)
        or bool(getattr(args, "video", False))
    )


def _is_subtitle_ext(ext: str | None) -> bool:
    return (ext or "").lower().lstrip(".") in getattr(
        defin, "SUB_EXTENSIONS", {"srt", "ass", "vtt", "sup", "idx", "sub"}
    )


def _is_image_ext(ext: str | None) -> bool:
    return (ext or "").lower().lstrip(".") in getattr(
        defin, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg", "webp", "bmp"}
    )


def _validate_sub_format(ext: str | None) -> str | None:
    e = (ext or "").lower().lstrip(".")
    return e if _is_subtitle_ext(e) and e in {"srt", "ass", "vtt"} else None


def _validate_image_format(ext: str | None) -> str | None:
    e = (ext or "").lower().lstrip(".")
    return e if _is_image_ext(e) else None


def _auto_flags_from_output_or_format(args: Any, input_files: List[str]) -> None:
    """
    Wenn keine Modi gesetzt sind, versuche Modus/Defaults aus --format/--output abzuleiten.
    - Sprachen → Subtitles aktivieren, wenn gesetzt
    - --format nutzt SUB_EXTS für subtitle, IMG_EXTS für frame
    - --output-Endung kann Modus implizieren
    - Bei GENAU 1 Datei + keine Flags → audio + subs + frame
    """

    # Format ableiten (unterstützt jetzt Listen wie "ass,png;webp")
    fmt_raw = getattr(args, "format", None) or ""
    fmt_tokens: List[str] = []
    for tok in fmt_raw.replace(";", ",").split(","):
        t = tok.strip().lower().lstrip(".")
        if t:
            fmt_tokens.append(t)

    if fmt_tokens:
        # Irgendein Sub-Format → Subtitles aktivieren (falls noch nicht gesetzt)
        if getattr(args, "subtitle", None) is None and any(
            _is_subtitle_ext(t) for t in fmt_tokens
        ):
            args.subtitle = ""  # alle Untertitel

        # Irgendein Bild-Format → Frame aktivieren (falls noch nicht gesetzt)
        if getattr(args, "frame", None) is None and any(
            _is_image_ext(t) for t in fmt_tokens
        ):
            args.frame = ""  # später Default-Zeit

    # Output-Endung auswerten
    out = getattr(args, "output", None)
    if out and _no_flags_selected(args):
        ext = Path(out).suffix.lower().lstrip(".")
        if _is_subtitle_ext(ext):
            args.subtitle = ""
            args.format = ext
        elif _is_image_ext(ext):
            args.frame = ""
            if not getattr(args, "format", None):
                args.format = ext

    # Falls immer noch nichts gewählt: bei genau 1 Datei → "alles was geht"
    if _no_flags_selected(args) and len(input_files) == 1:
        args.audio = True
        args.subtitle = ""
        args.frame = "middle"  # explizit Mitte


def norm_lang_iso3(lang: str) -> str:
    """
    Normalisiert beliebige Sprachangaben (z.B. 'de', 'de-DE', 'german')
    auf ISO-639-3-Codes gemäß LANG_ISO3.

    Rückgabe:
        - ISO3-Code, z.B. 'deu', 'eng'
        - '' bei unbekannter/blacklist-Sprache
    """
    s = (lang or "").strip().lower()
    if not s:
        return ""
    # explizit Blacklist (ffprobe-Standardwerte etc.)
    if s in defin.BLACKLIST_LANG:
        return ""

    s = s.replace("_", "-")

    # 1) Direkter Treffer in der Mapping-Tabelle
    iso3 = defin.LANG_ISO3.get(s)
    if iso3:
        return iso3

    # 2) Region-Codes 'xx-YY' → Basis 'xx' versuchen
    if "-" in s:
        base = s.split("-", 1)[0]
        iso3 = defin.LANG_ISO3.get(base)
        if iso3:
            return iso3

    # 3) Wenn schon ein 3-Buchstaben-Code: akzeptieren, falls bekannt
    if len(s) == 3 and s.isalpha():
        if s in defin.LANG_DISPLAY or s in defin.LANG_ISO3.values():
            return s

    # 4) Fallback: erste zwei Buchstaben als ISO-2 versuchen
    if len(s) >= 2 and s[:2].isalpha():
        base2 = s[:2]
        iso3 = defin.LANG_ISO3.get(base2)
        if iso3:
            return iso3

    return ""


def _norm_lang(s: str) -> str:
    """
    Normalisiert auf ISO-639-3 (z.B. 'deu','eng') mithilfe von defin.norm_lang_iso3.
    Fällt im Notfall auf die alte 2-Letter-Heuristik zurück (für ältere definitions-Versionen).
    """
    # bevorzugt zentrale Logik aus definitions

    iso = norm_lang_iso3(s)
    if iso:
        return iso

    # Fallback (sollte in deinem Setup eigentlich nie greifen)
    s = (s or "").strip().lower().replace("_", "-")
    if s in getattr(defin, "BLACKLIST_LANG", set()):
        return ""

    iso_map = getattr(defin, "LANG_ISO3", {})

    if s in iso_map:
        return iso_map[s]

    if "-" in s:
        base = s.split("-", 1)[0]
        if base in iso_map:
            return iso_map[base]

    if len(s) >= 2 and s[:2].isalpha():
        base2 = s[:2]
        if base2 in iso_map:
            return iso_map[base2]

    return ""


def _parse_lang_list(user_input: str) -> List[str]:
    """
    Parst eine Komma-/Semikolon-Liste in *ISO-3* Codes (z.B. 'de, en' -> ['deu','eng']).
    Dubletten werden entfernt, Reihenfolge beibehalten.
    """
    out: List[str] = []
    seen: set[str] = set()
    for tok in (user_input or "").replace(";", ",").split(","):
        code = _norm_lang(tok)
        if code and code not in seen:
            out.append(code)
            seen.add(code)
    return out


def _sub_label(st: Dict[str, Any]) -> str:
    lang = st.get("_norm_lang", "")  # ← jetzt ISO-3
    disp = st.get("disposition") or {}
    codec = (st.get("codec_name") or "").lower()

    parts: List[str] = []
    if lang:
        parts.append(lang)
    if disp.get("forced", 0):
        parts.append("forced")
    if (
        disp.get("hearing_impaired", 0)
        or disp.get("visual_impaired", 0)
        or disp.get("captions", 0)
    ):
        parts.append("sdh")
    if codec in {"eia_608", "eia_708"}:
        parts.append("cc")

    return "_".join(parts) or "subs"


def _probe_first_vcodec(path: Path) -> str | None:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        c = (r.stdout or "").strip().lower()
        return c or None
    except Exception:
        return None


VIDEO_CONTAINERS = defin.FORMATS


def _pick_video_container(
    format_opt: str | None, input_path: Path, default: str = "mp4"
) -> str:
    ext_opt = (format_opt or "").lower().lstrip(".")
    in_ext = input_path.suffix.lower().lstrip(".")
    if ext_opt in VIDEO_CONTAINERS:
        return ext_opt
    if in_ext in VIDEO_CONTAINERS:
        return in_ext
    return default


def _container_for_video(
    codec: str | None, desired_ext: str, input_ext: str | None = None
) -> str:
    if desired_ext not in VIDEO_CONTAINERS:
        desired_ext = input_ext if (input_ext in VIDEO_CONTAINERS) else "mp4"
    if desired_ext == "mp4":
        if codec not in {"h264", "hevc", "h265", "av1", "mpeg4"}:
            return "mkv"
    return desired_ext


def _resolve_out_path(
    base_output: str | None,
    input_path: Path,
    desired_ext: str,
    extra: str = "",
    multi_count: int = 1,
) -> Path:
    desired_ext = desired_ext.lower()
    if not desired_ext.startswith("."):
        desired_ext = "." + desired_ext

    in_stem = input_path.stem

    if not base_output:
        return input_path.with_stem(in_stem + extra).with_suffix(desired_ext)

    base = Path(base_output)
    if multi_count == 1:
        if base.suffix.lower() != desired_ext:
            base = base.with_suffix(desired_ext)
        return base

    if base.exists() and base.is_dir():
        return base / f"{in_stem}{extra}{desired_ext}"

    parent = base.parent if base.parent != Path("") else Path(".")
    out_stem = base.stem or "out"
    return (parent / f"{in_stem}_{out_stem}{extra}").with_suffix(desired_ext)


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
        num = s[:-1].replace(",", ".")
        val = float(num)
        if not (0.0 <= val <= 100.0):
            raise ValueError("percent out of range")
        return duration_s * (val / 100.0)

    if s.endswith("s"):
        num = s[:-1].replace(",", ".")
        return float(num)

    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            hh, mm, ss = parts
        elif len(parts) == 2:
            hh, mm, ss = "0", parts[0], parts[1]
        elif len(parts) == 1:
            hh, mm, ss = "0", "0", parts[0]
        else:
            raise ValueError("too many ':'")
        try:
            h = int(hh)
            m = int(mm)
            sec = float(ss.replace(",", "."))
        except ValueError:
            raise ValueError("invalid time tokens")
        if not (0 <= m < 60 and 0 <= sec < 60):
            raise ValueError("MM/SS must be in [0,60)")
        return h * 3600 + m * 60 + sec

    return float(s.replace(",", "."))


# ---------------- Hauptfunktion ----------------
def extract(args: ExtractArgs | Namespace | Any):
    """Extract audio/subtitles/frame or produce video-only, batch & interactive, consistent with `trim` style."""

    # Prepare inputs (BATCH_MODE depends entirely on provided files)
    BATCH_MODE, files = he.prepare_inputs(args)
    co.print_start(_("extract_method"))

    # ensure args.files reflects expanded list for downstream logic
    try:
        args.files = list(files)
    except Exception:
        pass

    # Initialize possibly-unbound variables for Pylance
    extraction_choice: Optional[str] = None
    do_audio = do_video = do_frame = do_subs = False
    frame_time: Optional[str] = None
    frame_format: Optional[str] = None
    subtitle_format: Optional[str] = None
    preferred_langs: Optional[List[str]] = None
    frame_times_choice: Optional[str] = None

    # If no explicit mode flags were given, try to infer from --format/--output
    if _no_flags_selected(args):
        _auto_flags_from_output_or_format(args, files)

    # --------- Format-Liste aus --format parsen (neu) ---------
    def _parse_format_list(fmt_raw: Optional[str]) -> List[str]:
        if not fmt_raw:
            return []
        out = []
        for tok in fmt_raw.replace(";", ",").split(","):
            tt = tok.strip().lower().lstrip(".")
            if tt:
                out.append(tt)
        return out

    # --- Batch/Tasks vorbereiten ---
    if BATCH_MODE:
        do_audio = bool(getattr(args, "audio", False))
        do_video = bool(getattr(args, "video", False))
        do_frame = getattr(args, "frame", None) is not None
        do_subs = getattr(args, "subtitle", None) is not None

        if len(files) == 1 and not (do_audio or do_video or do_frame or do_subs):
            do_audio = True
            do_subs = True
            do_frame = True

        # --- Formate aus --format einmal global klassifizieren ---
        fmt_list_all = _parse_format_list(getattr(args, "format", None))

        fmt_sub: Optional[str] = None
        fmt_img: Optional[str] = None
        unknown_formats: List[str] = []

        for f in fmt_list_all:
            sub_norm = _validate_sub_format(f)  # 'srt','ass','vtt' oder None
            img_norm = _validate_image_format(f)  # z.B. 'png','jpg','webp' oder None

            is_sub = sub_norm is not None
            is_img = img_norm is not None

            if is_sub and fmt_sub is None:
                fmt_sub = sub_norm  # normalisierte Schreibweise
            if is_img and fmt_img is None:
                fmt_img = img_norm

            if not is_sub and not is_img:
                unknown_formats.append(f)

        if unknown_formats:
            co.print_warning(
                _("unknown_format_entry") + ", ".join(f"'{x}'" for x in unknown_formats)
            )

        if do_frame:
            frame_time = getattr(args, "frame", None)

            # Primär: erstes gültiges Bildformat aus --format
            frame_format = fmt_img or "jpg"

            # Falls kein Bildformat gewählt wurde → ggf. aus --output ableiten
            if fmt_img is None:
                out_opt = getattr(args, "output", None)
                if out_opt:
                    out_ext = Path(out_opt).suffix.lower().lstrip(".")
                    if _is_image_ext(out_ext):
                        frame_format = out_ext

        if do_subs:
            # Erstes gültiges Subtitle-Format aus --format, sonst Default
            subtitle_format = fmt_sub or "srt"

        # --------- Subtitle-Sprachauswertung direkt hinter --subtitle ---------
        raw_sub = getattr(args, "subtitle", None)

        # Fälle:
        #   None        → keine Subtitles
        #   ""          → alle Subtitles (nur Flag)
        #   "all"       → alle Subtitles
        #   "de,eng"    → gefilterte ISO-Manifest

        if raw_sub is None:
            preferred_langs = None  # subtitles off
        else:
            tok = raw_sub.strip().lower()

            # nur Flag → alle
            if tok == "":
                preferred_langs = None

            # explizit "all"
            elif tok == "all":
                preferred_langs = None

            else:
                # Sprache(n)-Liste parsen → ISO-3 konvertieren
                preferred_langs = _parse_lang_list(tok)

                # Nichts erkannt → Fehler + abbrechen
                if not preferred_langs:
                    co.print_fail(_("invalid_subtitle_language").format(langs=raw_sub))
                    preferred_langs = []  # leere Liste = extrahiere nichts

    else:
        # Interaktiv
        if not files:
            co.print_finished(_("extract_method"))
            return

        # Auswahl: Audio/Subtitles/Frame/Video-only
        extraction_type_keys = list(getattr(defin, "EXTRACT_MODE", {}).keys())
        extraction_type_descriptions = [
            tr((getattr(defin, "EXTRACT_MODE", {}).get(k) or {}).get("description", ""))
            for k in extraction_type_keys
        ]
        extraction_type_labels = [
            (getattr(defin, "EXTRACT_MODE", {}).get(k) or {}).get("name", k)
            for k in extraction_type_keys
        ]
        extraction_choice = ui.ask_user(
            _("select_extract_type"),
            extraction_type_keys,
            extraction_type_descriptions,
            2,
            extraction_type_labels,
        )

        if extraction_choice == "subtitles":
            sub_tbl = getattr(defin, "SUBTITLE_FORMATS", {})
            subtitle_keys = list(sub_tbl.keys())
            subtitle_descriptions = [
                tr((sub_tbl.get(k) or {}).get("description", "")) for k in subtitle_keys
            ]
            subtitle_labels = [
                (sub_tbl.get(k) or {}).get("name", k) for k in subtitle_keys
            ]
            subtitle_format = ui.ask_user(
                _("select_subtitle_format"),
                subtitle_keys,
                subtitle_descriptions,
                0,
                subtitle_labels,
            )
            if subtitle_format is None:
                co.print_finished(_("extract_method"))
                return

            pref_tbl = getattr(defin, "EXTRACT_SUBTITLE_OPTIONS", {})
            pref_keys = list(pref_tbl.keys())
            pref_labels = [tr(pref_tbl[k]) for k in pref_keys]
            pref_choice = ui.ask_user(
                _("select_preferred_subtitle_languages"),
                pref_keys,
                default=2,
                display_labels=pref_labels,
            )
            if pref_choice is None:
                co.print_finished(_("extract_method"))
                return

            if pref_choice == "german":
                preferred_langs = ["deu"]
            elif pref_choice == "english":
                preferred_langs = ["eng"]
            elif pref_choice == "german+english":
                preferred_langs = ["deu", "eng"]
            elif pref_choice == "all":
                preferred_langs = None
            else:
                raw = input(co.return_promt(_("enter_languages_comma") + ": "))
                preferred_langs = _parse_lang_list(raw)
                if not preferred_langs:
                    co.print_info(_("lang_fallback_all"))
                    preferred_langs = None

        elif extraction_choice == "frame":
            frame_mode = getattr(defin, "EXTRACT_MODE", {}).get("frame") or {}
            frame_formats_map = cast(Dict[str, Any], frame_mode.get("formats", {}))
            frame_format_keys = list(frame_formats_map.keys())
            frame_format = ui.ask_user(
                _("select_frame_format"), frame_format_keys, default=1
            )
            if frame_format is None:
                co.print_finished(_("extract_method"))
                return

            ft_tbl = getattr(defin, "FRAME_TIMES_PRESET", {})
            frame_times_keys = list(ft_tbl.keys())
            frame_times_descriptions = [
                tr((ft_tbl.get(k) or {}).get("description", ""))
                for k in frame_times_keys
            ]
            frame_times_labels = [
                tr((ft_tbl.get(k) or {}).get("name", k)) for k in frame_times_keys
            ]
            frame_times_choice = ui.ask_user(
                _("select_frame_time_preset"),
                frame_times_keys,
                frame_times_descriptions,
                4,
                frame_times_labels,
            )
            if frame_times_choice is None:
                co.print_finished(_("extract_method"))
                return
            elif frame_times_choice == "custom":
                frame_time = ui.read_frame_time()
                if frame_time is None:
                    co.print_finished(_("extract_method"))
                    return

        elif extraction_choice in ("audio", "video_only"):
            pass
        else:
            sys.exit(1)

    # ------------- Verarbeiten -------------
    for file in files:
        path = Path(file)

        # ---- AUDIO ----
        if (not BATCH_MODE and extraction_choice == "audio") or (
            BATCH_MODE and do_audio
        ):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=index:stream_tags=language",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )

            try:
                data = json.loads(probe.stdout or "{}")
            except json.JSONDecodeError:
                data = {}

            audio_streams = data.get("streams", [])
            if not audio_streams:
                co.print_fail(_("no_audio_track").format(filename=path.name))
                continue

            n_streams = len(audio_streams)

            # ISO-3-normalisierte Sprachen für "unterscheidbare" Suffixe
            langs_all: List[str] = []
            for s in audio_streams:
                raw_lang = (s.get("tags") or {}).get("language") or ""
                norm = _norm_lang(raw_lang)  # → z.B. 'deu', 'eng', '' bei unbekannt
                langs_all.append(norm)

            has_distinct_langs = len({lang for lang in langs_all if lang}) > 1

            for ai, s in enumerate(audio_streams):
                global_idx = s.get("index", ai)
                raw_lang = (s.get("tags") or {}).get("language") or ""
                norm_lang = _norm_lang(raw_lang)

                suffix_parts: List[str] = []
                if n_streams > 1:
                    suffix_parts.append(f"audio{ai}")
                    if has_distinct_langs and norm_lang:
                        # Dateiname bekommt ISO-3: z.B. "..._audio0_deu.mp3"
                        suffix_parts.append(norm_lang)
                else:
                    suffix_parts.append("audio")

                extra_tag = "_" + "_".join(suffix_parts)

                out = _resolve_out_path(
                    getattr(args, "output", None),
                    path,
                    "mp3",
                    extra=extra_tag,
                    multi_count=max(len(files), n_streams),
                )
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
                    str(path),
                    "-map",
                    f"0:{global_idx}",
                    "-vn",
                    "-sn",
                    "-c:a",
                    "libmp3lame",
                    "-q:a",
                    "0",
                    "-id3v2_version",
                    "3",
                    str(out),
                ]
                cmd = autotune_final_cmd(path, cmd)
                pw.run_ffmpeg_with_progress(
                    path.name,
                    cmd,
                    _("extracting_audio_progress"),
                    _("extracting_audio_done"),
                    output_file=out,
                    BATCH_MODE=BATCH_MODE,
                )

        # ---- SUBTITLES ----
        if (not BATCH_MODE and extraction_choice == "subtitles") or (
            BATCH_MODE and do_subs
        ):
            fmt_key: str = cast(str, subtitle_format or "srt")
            text_codec_map = {"srt": "srt", "ass": "ass", "vtt": "webvtt"}
            ext_map = {
                "srt": "srt",
                "ass": "ass",
                "vtt": "vtt",
                "sup": "sup",
                "vobsub": "idx",
            }

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "s",
                    "-show_entries",
                    "stream=index,codec_name,disposition:stream_tags=language",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )

            try:
                data = json.loads(probe.stdout or "{}")
            except json.JSONDecodeError:
                data = {}
            streams = data.get("streams", [])

            if not streams:
                co.print_fail(_("no_subtitle_tracks").format(filename=path.name))
                continue

            for s in streams:
                raw = (s.get("tags") or {}).get("language") or ""
                s["_norm_lang"] = _norm_lang(raw)

            if preferred_langs is None:
                # alle
                selected = streams

            elif len(preferred_langs) == 0:
                # Benutzer hat explizit "falsche Sprache" eingegeben → nichts extrahieren
                selected = []

            else:
                # gefiltert
                want = set(preferred_langs)
                selected = [s for s in streams if s.get("_norm_lang") in want]

            labels = [_sub_label(st) for st in selected]
            counts = Counter(labels)
            ord_map: Dict[str, int] = {}

            n_streams = len(selected)

            for si, s in enumerate(selected):
                global_idx = s.get("index", si)
                codec = (s.get("codec_name") or "").lower()
                is_bitmap = codec in {
                    "hdmv_pgs_subtitle",
                    "pgs",
                    "dvd_subtitle",
                    "dvb_subtitle",
                    "xsub",
                    "vobsub",
                }

                out_fmt = fmt_key
                out_codec = text_codec_map.get(fmt_key, "copy")
                mux_args: List[str] = []

                if is_bitmap and fmt_key in {"srt", "ass", "vtt"}:
                    co.print_warning(
                        _("subtitle_is_bitmap").format(trackname=s, codec=codec)
                    )
                    if codec in {"hdmv_pgs_subtitle", "pgs"}:
                        out_fmt = "sup"
                        out_codec = "copy"
                        mux_args = ["-f", "sup"]
                    else:
                        out_fmt = "vobsub"
                        out_codec = "copy"
                        mux_args = ["-f", "vobsub"]

                base = labels[si]
                suffix_parts = [base]
                if counts[base] > 1:
                    ord_map[base] = ord_map.get(base, 1)
                    suffix_parts.append(str(ord_map[base]))
                    ord_map[base] += 1
                extra = "_" + "_".join(suffix_parts)

                out = _resolve_out_path(
                    getattr(args, "output", None),
                    path,
                    ext_map[out_fmt],
                    extra=extra,
                    multi_count=max(len(files), n_streams),
                )

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
                    str(path),
                    "-map",
                    f"0:{global_idx}",
                    "-vn",
                    "-an",
                    "-c:s",
                    out_codec,
                    *mux_args,
                    str(out),
                ]
                cmd = autotune_final_cmd(path, cmd)
                pw.run_ffmpeg_with_progress(
                    path.name,
                    cmd,
                    _("extracting_subtitle_progress"),
                    _("extracting_subtitle_done"),
                    output_file=out,
                    BATCH_MODE=BATCH_MODE,
                )

        # ---- FRAME (Einzelbild) ----
        if (not BATCH_MODE and extraction_choice == "frame") or (
            BATCH_MODE and do_frame
        ):
            duration_s = he.probe_duration_sec(path)

            if BATCH_MODE:
                t_raw = frame_time  # '' oder String
                fmt = frame_format or "jpg"
                if t_raw == "":
                    t_seconds = 1.0 if duration_s is None else (duration_s / 3.0)
                else:
                    if not t_raw:
                        t_seconds = (
                            (duration_s / 2.0) if duration_s is not None else 1.0
                        )
                    else:
                        try:
                            t_seconds = _parse_frame_time_to_seconds(t_raw, duration_s)
                        except ValueError:
                            co.print_fail(_("invalid_time_input"))
                            continue
            else:
                fmt = frame_format or "jpg"
                if frame_times_choice == "custom":
                    t_raw = frame_time
                else:
                    t_raw = frame_times_choice

            if not t_raw:
                if duration_s is None:
                    co.print_error(_("no_duration_fallback"))
                    t_seconds = 1.0
                else:
                    t_seconds = duration_s / 2.0
            else:
                try:
                    t_seconds = _parse_frame_time_to_seconds(
                        cast(str, t_raw), duration_s
                    )
                except ValueError:
                    co.print_fail(_("invalid_time_input"))
                    continue

            t_seconds = he.clamp_seek_time(t_seconds, duration_s)

            if BATCH_MODE and len(files) == 1 and getattr(args, "output", None):
                out = Path(cast(str, args.output))
                if out.suffix.lower().lstrip(".") != fmt:
                    out = out.with_suffix("." + fmt)
            else:
                out = path.with_stem(path.stem + "_frame").with_suffix("." + fmt)

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
                f"{t_seconds:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
            ]
            if fmt in {"jpg", "jpeg"}:
                cmd += ["-q:v", "2"]
            cmd += [str(out)]
            cmd = autotune_final_cmd(path, cmd)
            out_tmp = pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("extracting_frame_progress"),
                _("extracting_frame_done"),
                output_file=out,
                BATCH_MODE=BATCH_MODE,
            )
            # Sicherer Umgang mit optionalem Rückgabewert/Typ
            if out_tmp is not None and isinstance(out_tmp, (str, Path)):
                out = Path(out_tmp)

            try:
                displayer.show_image(str(out))  # show_image erwartet str
            except Exception:
                pass

        # ---- VIDEO-ONLY (Audio/Subs entfernen; Video stream-copy) ----
        if (not BATCH_MODE and extraction_choice == "video_only") or (
            BATCH_MODE and do_video
        ):
            format_opt = cast(Optional[str], getattr(args, "format", None))
            desired_ext = _pick_video_container(format_opt, path, default="mp4")
            vcodec = _probe_first_vcodec(path)
            out_ext = _container_for_video(
                vcodec, desired_ext, input_ext=path.suffix.lstrip(".").lower()
            )

            extra = "_video"  # auch bei 1 Datei – klarer Output
            out = _resolve_out_path(
                getattr(args, "output", None),
                path,
                out_ext,
                extra=extra,
                multi_count=len(files),
            )

            map_args = ["-map", "0:v:0?"]

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
                str(path),
                *map_args,
                "-map",
                "-0:a",
                "-map",
                "-0:s",
                "-map",
                "-0:d",
                "-map",
                "-0:t",
                "-c:v",
                "copy",
            ]

            if out_ext == "mp4":
                cmd += ["-movflags", "+faststart"]
                # if vcodec in {"hevc", "h265"}:
                #    cmd += ["-tag:v", "hvc1"]

            cmd += [str(out)]
            cmd = autotune_final_cmd(path, cmd)

            pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("extracting_videoonly_progress"),
                _("extracting_videoonly_done"),
                output_file=out,
                BATCH_MODE=BATCH_MODE,
            )

    co.print_finished(_("extract_method"))
