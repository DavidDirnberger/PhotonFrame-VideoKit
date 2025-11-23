#!/usr/bin/env python3
# trim.py
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import consoleOutput as co
import definitions as defin
import fileSystem as fs
import helpers as he
import process_wrappers as pw
import userInteraction as ui
import video_thumbnail as vt
import VideoEncodersCodecs as vec
from ffmpeg_perf import autotune_final_cmd

# local modules
from i18n import _, tr

# ---------- kleine, getypte Helfer für Eingaben ----------


def _list_str_factory() -> List[str]:
    return []


def _to_str_list(v: object) -> List[str]:
    """
    Normalisiert beliebige File-Argumente strikt zu List[str].
    Vermeidet 'Unknown' in Comprehensions.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        out: List[str] = []
        for elem in cast(Iterable[object], v):
            if elem is not None:
                out.append(str(elem))
        return out
    return [str(v)]


def _prepare_inputs_typed(args: Any) -> Tuple[bool, List[str]]:
    bm_raw, fl_any = he.prepare_inputs(args)
    return bool(bm_raw), _to_str_list(fl_any)


@dataclass
class TrimArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    start: Optional[str] = "1:00"
    duration: Optional[str] = "1:00"  # Default bleibt erhalten für CLI-Kompat.
    end: Optional[str] = None  # absolute Endzeit optional
    output: Optional[str] = None
    precise: Optional[bool] = False
    quality: Optional[str] = None


def _read_len_or_end_prompt() -> tuple[Optional[str], Optional[str]]:
    raw = input(co.return_promt(_("trim_len_or_end_prompt") + " ")).strip()
    if not raw:
        return None, None
    if raw == "0":
        co.print_line("   ➜ " + _("exiting"))
        sys.exit(0)

    low = raw.lower()

    # nacktes '-…' wird als *Endzeit* interpretiert (interaktiver Modus)
    if low.startswith("-"):
        return None, raw

    if low.startswith("+"):
        return raw[1:].strip(), None
    if low.startswith("d:") or low.startswith("dur:") or low.startswith("duration:"):
        return raw.split(":", 1)[1].strip(), None
    if low.startswith("end:") or low.startswith("@"):
        return None, raw.split(":", 1)[1].strip() if ":" in raw else raw[1:].strip()
    return None, raw


def _resolve_range_for_file(
    start_raw: str,
    duration_raw: Optional[str],
    end_raw: Optional[str],
    video_path: Path,
) -> tuple[str, str, str]:
    """
    Unterstützt:
      • Zeitangaben & Prozent (…%, …p/P) via he.time_or_percent_to_seconds(...)
      • Vorzeichen-Minus: Wert wird von der *Gesamtzeit* abgezogen
        - start  : start = T - |x|
        - end    : end   = T - |x|
        - duration: end  = T - |x|  → dur = end - start
    """
    vid_sec = float(he.get_duration_seconds(video_path) or 0.0)

    # --- START ---
    try:
        s_neg = start_raw.strip().startswith("-")
        s_base = (
            he.time_or_percent_to_seconds(start_raw.lstrip("+-").strip(), vid_sec)
            if s_neg
            else he.time_or_percent_to_seconds(start_raw, vid_sec)
        )
        start_sec = max(0.0, (vid_sec - s_base) if s_neg else s_base)
    except Exception:
        raise ValueError(_("invalid_start").format(val=start_raw))

    if vid_sec > 0 and start_sec >= vid_sec:
        start_sec = max(vid_sec - 1.0, 1e-3)

    # --- ENDE ODER DAUER ---
    if end_raw is not None and (duration_raw is None):
        # Endzeitpfad
        try:
            e_neg = end_raw.strip().startswith("-")
            e_base = (
                he.time_or_percent_to_seconds(end_raw.lstrip("+-").strip(), vid_sec)
                if e_neg
                else he.time_or_percent_to_seconds(end_raw, vid_sec)
            )
            end_sec = (vid_sec - e_base) if e_neg else e_base
        except Exception:
            raise ValueError(_("invalid_end").format(val=end_raw))

        if vid_sec > 0:
            end_sec = max(1.0, min(end_sec, max(vid_sec - 1e-3, 1.0)))
        if end_sec <= start_sec:
            raise ValueError(
                _("end_must_be_gt_start").format(
                    start=start_raw, end=he.sec_to_time_str(end_sec)
                )
            )
        dur_sec = end_sec - start_sec

    else:
        # Dauerpfad
        if duration_raw is None:
            raise ValueError(_("need_duration_or_end"))

        d_neg = duration_raw.strip().startswith("-")
        if d_neg:
            # Negative Dauer bedeutet: Endzeit = T - |d|, daraus Dauer ableiten
            try:
                d_base = he.time_or_percent_to_seconds(
                    duration_raw.lstrip("+-").strip(), vid_sec
                )
            except Exception:
                raise ValueError(_("invalid_duration").format(val=duration_raw))

            end_sec = vid_sec - d_base
            if vid_sec > 0:
                end_sec = max(1.0, min(end_sec, max(vid_sec - 1e-3, 1.0)))

            if end_sec <= start_sec:
                # kein gültiger Bereich
                raise ValueError(
                    _("end_must_be_gt_start").format(
                        start=start_raw, end=he.sec_to_time_str(end_sec)
                    )
                )

            dur_sec = max(1.0, end_sec - start_sec)

        else:
            # Positive/gewöhnliche Dauer
            try:
                dur_sec = he.time_or_percent_to_seconds(duration_raw, vid_sec)
            except Exception:
                raise ValueError(_("invalid_duration").format(val=duration_raw))
            if dur_sec <= 0:
                raise ValueError(_("duration_must_be_pos"))
            end_sec = start_sec + dur_sec
            if vid_sec > 0:
                end_sec = max(1.0, min(end_sec, max(vid_sec - 1e-3, 1.0)))
                dur_sec = max(1.0, end_sec - start_sec)

    start_str = he.sec_to_time_str(start_sec)
    duration_str = he.sec_to_time_str(dur_sec)
    end_str = he.sec_to_time_str(end_sec)
    return start_str, duration_str, end_str


# --- Container-spezifische Copy-/Mux-Flags (verlustfrei) ---
def _container_copy_flags(ext: str) -> tuple[list[str], list[str]]:
    ext = (ext or "").lower()
    input_flags: list[str] = []
    output_flags: list[str] = []

    # ► PTS/TS-Erzeugung bereits beim Einlesen
    input_flags += ["-fflags", "+genpts"]

    if ext == "mkv":
        output_flags += [
            "-muxpreload",
            "0",
            "-muxdelay",
            "0",
            "-avoid_negative_ts",
            "make_zero",
            "-copyinkf",  # ← WICHTIG: kein "1" dahinter!
        ]
    elif ext in ("mp4", "m4v", "mov"):
        output_flags += [
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            "-copyinkf",  # ← WICHTIG: kein "1" dahinter!
        ]
    return input_flags, output_flags


def _build_copy_trim_cmd(
    path: Path, s: str, e: str, out_ext: str, output: Path
) -> list[str]:
    in_flags, out_flags = _container_copy_flags(out_ext)

    # Container-aware stream mapping (inkl. Subtitle-Handhabung):
    # - MP4: textbasierte Subs → mov_text
    # - MKV: Subs → copy
    maps = vec.build_stream_mapping_args(out_ext, input_path=path)

    # Kein globales "-c copy", sonst kollidiert MP4 mit Nicht-mov_text Subs
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "0.5",
        "-ss",
        s,
        "-to",
        e,
        *in_flags,
        "-i",
        str(path),
        *maps,
        "-dn",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        *out_flags,
        str(output),
    ]


# --- Lossless/Near-lossless Fallback (optional HW) ---
def _build_lossless_reencode_cmd(
    path: Path, s: str, e: str, out_ext: str, output: Path, prefer_hw: bool
) -> list[str]:
    # Dauer aus Start/Ende berechnen und als -t nutzen (präziser als -to)
    s_sec = he.parse_time_relaxed_seconds(s)
    e_sec = he.parse_time_relaxed_seconds(e)
    dur_sec = max(0.01, e_sec - s_sec)
    dur_str = he.sec_to_time_str(dur_sec)

    # Source-Infos
    src_codec = (vec.probe_video_codec(path) or "h264").lower()
    src_w, src_h, _ = vec.ffprobe_geometry(path)
    src_fps = he.probe_src_fps(path)

    # codec_key an Container anpassen
    codec_key = vec.normalize_codec_key(src_codec) or src_codec
    if not vec.container_allows_codec(out_ext, codec_key):
        codec_key = vec.suggest_codec_for_container(out_ext)

    # Lossless-Plan bauen
    plan = vec.build_transcode_plan(
        input_path=path,
        target_container=out_ext,
        preset_name="lossless",
        codec_key=codec_key,
        preferred_encoder=None,
        req_scale=None,
        src_w=src_w,
        src_h=src_h,
        src_fps=src_fps,
        user_fps_rational=None,
        preset_max_fps=None,
        force_key_at_start=True,
    )

    cmd: list[str] = cast(list[str], vec.assemble_ffmpeg_cmd(plan, output))

    # Präziser Schnitt: -ss NACH -i + -t <Dauer> einfügen
    try:
        i_idx = cmd.index("-i")
        insert_at = i_idx + 2
        cmd[insert_at:insert_at] = ["-ss", s, "-t", dur_str]
    except ValueError:
        cmd += ["-ss", s, "-t", dur_str]

    return cmd


def _build_precise_reencode_cmd(
    path: Path,
    start_str: str,
    dur_str: str,
    output: Path,
    *,
    mode: str,
    preset: Optional[str],
) -> list[str]:
    """
    mode: 'preset' | 'lossless'
    - 'preset': nutzt Plan-API + gewünschtes Preset (z.B. 'cinema')
    - 'lossless': nutzt lossless-Args + All-Intra (wie oben)
    """
    out_ext = output.suffix.lstrip(".").lower() or (
        vec.detect_container_from_path(path) or "mkv"
    )

    if mode == "lossless":
        end_sec = he.parse_time_relaxed_seconds(
            start_str
        ) + he.parse_time_relaxed_seconds(dur_str)
        end_str = he.sec_to_time_str(end_sec)
        return _build_lossless_reencode_cmd(
            path, start_str, end_str, out_ext, output, prefer_hw=True
        )

    # --- preset-Modus über Plan-API ---
    src_codec = (vec.probe_video_codec(path) or "h264").lower()
    src_w, src_h, _ = vec.ffprobe_geometry(path)
    src_fps = he.probe_src_fps(path)

    codec_key = vec.normalize_codec_key(src_codec) or src_codec
    if not vec.container_allows_codec(out_ext, codec_key):
        codec_key = vec.suggest_codec_for_container(out_ext)

    enc_maps = vec.prepare_encoder_maps(
        [str(path)],
        out_ext,
        defin.CONVERT_FORMAT_DESCRIPTIONS,
        prefer_hw=True,
        ffmpeg_bin="ffmpeg",
    )
    enc_map = enc_maps.get(out_ext, {}) if isinstance(enc_maps, dict) else {}
    preferred = enc_map.get(codec_key) if codec_key != "copy" else None

    plan = vec.build_transcode_plan(
        input_path=path,
        target_container=out_ext,
        preset_name=(preset or "cinema"),
        codec_key=codec_key,
        preferred_encoder=preferred,
        req_scale=None,
        src_w=src_w,
        src_h=src_h,
        src_fps=src_fps,
        user_fps_rational=None,
        preset_max_fps=defin.CONVERT_PRESET.get(preset or "cinema", {}).get("max_fps"),
        force_key_at_start=True,
    )
    cmd: list[str] = cast(list[str], vec.assemble_ffmpeg_cmd(plan, output))

    # Falls der Plan wider Erwarten copy gewählt hat → SW-Encoder wählen
    try:
        j = cmd.index("-c:v")
        if cmd[j + 1].lower() == "copy":
            fallback = vec.encoder_for_codec(
                vec.normalize_codec_key(plan.codec_key) or "h264"
            )
            cmd[j + 1] = fallback
    except Exception:
        pass

    # -ss nach -i, -t <dauer>
    try:
        i_idx = cmd.index("-i")
        insert_at = i_idx + 2
        cmd[insert_at:insert_at] = ["-ss", start_str, "-t", dur_str]
    except ValueError:
        cmd += ["-ss", start_str, "-t", dur_str]

    # Knappe Stream-Synchronität sicherstellen
    if "-shortest" not in cmd:
        cmd += ["-shortest"]

    return cast(list[str], vec.apply_container_codec_quirks(cmd, out_ext, codec_key))


# --- Hauptfunktion ---
def trim(args: Any) -> None:
    """Trim videos using start + duration OR end time.
    Interaktiv wie convert: ohne Dateien → UI, mit Dateien → Batch.
    """
    BATCH_MODE, files = _prepare_inputs_typed(args)
    co.print_start(_("trimming_method"))

    # 1) PARAMETER ERMITTELN
    if BATCH_MODE:
        start_raw = cast(str, getattr(args, "start", "0:00") or "0:00")
        dur_opt: Optional[str] = getattr(args, "duration", None)
        end_opt: Optional[str] = getattr(args, "end", None)
        USE_PRECISE: bool = bool(getattr(args, "precise", False))
        quality_preset: Optional[str] = getattr(args, "quality", None)

        if dur_opt and end_opt:
            co.print_warning(_("duration_and_end_given_use_duration"))
        dur_raw = dur_opt
        end_raw = None if dur_opt else end_opt

        per_file: Dict[str, tuple[str, str, str]] = {}
        start_list: List[str] = []
        dur_list: List[str] = []
        end_list: List[str] = []
        try:
            for f in files:
                s, d, e = _resolve_range_for_file(start_raw, dur_raw, end_raw, Path(f))
                per_file[f] = (s, d, e)
                start_list.append(s)
                dur_list.append(d)
                end_list.append(e)
        except ValueError as e:
            co.print_error(str(e))
            return

        params_table: Dict[str, object] = {
            "files": files,
            "start_time": start_list[0],
            "duration": dur_list[0],
            "end_time": end_list[0],
            "precision": (_("precise") if USE_PRECISE else _("fast-lossless")),
            "quality": (
                _("copy-lossless")
                if quality_preset is None
                else defin.CONVERT_PRESET[quality_preset]["name"]
            ),
        }

        co.print_selected_params_table(params_table)

    else:
        # Interaktiv: Eingabe + Vorschau + Moduswahl
        while True:
            start_raw = input(
                co.return_promt("\n   " + _("trim_start_prompt") + ": ")
            ).strip()
            if not start_raw:
                start_raw = "0:00"

            print(" ")
            dur_raw, end_raw = _read_len_or_end_prompt()
            if dur_raw is None and end_raw is None:
                dur_raw = "1:00"  # Default

            mode_keys = ["fast", "precise"]
            mode_desc = [
                _("trim_mode_fast_desc"),
                _("trim_mode_precise_desc"),
            ]
            mode_labels = {
                "fast": {
                    "de": "Schnell (lossless, Keyframes)",
                    "en": "Fast (lossless, keyframes)",
                },
                "precise": {
                    "de": "Präzise (Re-Encode, langsam)",
                    "en": "Precise (re-encode, slow)",
                },
            }
            model_options = [tr(mode_labels[k]) for k in mode_keys]
            sel = ui.ask_user(
                _("choose_trim_mode"),
                mode_keys,
                descriptions=mode_desc,
                default=0,
                display_labels=model_options,
            )
            if sel is None:
                continue
            quality_preset: Optional[str] = None
            USE_PRECISE = sel == "precise"
            if USE_PRECISE:
                quality_keys = list(defin.CONVERT_PRESET)
                quality_options = [
                    defin.CONVERT_PRESET[k]["name"] for k in quality_keys
                ]
                quality_descriptions = [
                    tr(defin.CONVERT_PRESET[k]["description"]) for k in quality_keys
                ]
                quality_preset = ui.ask_user(
                    _("choose_quality_preset"),
                    quality_keys,
                    descriptions=quality_descriptions,
                    default=4,
                    display_labels=quality_options,
                )
                if quality_preset is None:
                    continue

            per_file: Dict[str, tuple[str, str, str]] = {}
            start_list: List[str] = []
            dur_list: List[str] = []
            end_list: List[str] = []
            try:
                for f in files:
                    s, d, e = _resolve_range_for_file(
                        start_raw, dur_raw, end_raw, Path(f)
                    )
                    per_file[f] = (s, d, e)
                    start_list.append(s)
                    dur_list.append(d)
                    end_list.append(e)
            except ValueError as e:
                co.print_fail(str(e))
                continue

            params_table: Dict[str, object] = {
                "files": files,
                "start_time": start_list[0],
                "duration": dur_list[0],
                "end_time": end_list[0],
                "precision": (_("precise") if USE_PRECISE else _("fast-lossless")),
                "quality": (
                    _("copy-lossless")
                    if quality_preset is None
                    else defin.CONVERT_PRESET[quality_preset]["name"]
                ),
            }
            co.print_selected_params_table(params_table)

            try:
                sample = Path(files[0])
                s0, d0, _end_ignored = (
                    per_file[str(sample)]
                    if str(sample) in per_file
                    else per_file[files[0]]
                )
                targets: List[Path] = [Path(f) for f in files]
                ui.preview_trim_range(targets, s0, d0)
            except Exception as ex:
                co.print_warning(_("trim_preview_skipped") + f": {ex}")

            if ui.ask_yes_no(_("confirm_apply_trim"), default=True, back_option=False):
                break

    total = len(files)
    PREFER_HW = True  # falls du das später als Flag ausbaust

    for i, file in enumerate(files, start=1):
        path = Path(file)
        if not path.exists():
            co.print_warning(_("file_not_found").format(file=path.name))
            continue
        s, d, e = per_file[file]

        # ggf. vorhandenes Thumbnail konservieren
        preserved_cover: Path | None = None
        try:
            if vt.check_thumbnail(path, silent=True):
                preserved_cover = vt.extract_thumbnail(path)
        except Exception:
            preserved_cover = None

        output = fs.build_output_path(
            input_path=path,
            output_arg=getattr(args, "output", None),
            default_suffix="_trimmed",
            idx=i,
            total=total,
            target_ext=None,
        )
        out_ext = output.suffix.lstrip(".").lower() or (
            vec.detect_container_from_path(path) or "mkv"
        )

        if not USE_PRECISE:
            # 1) Lossless-GOP Copy
            cmd: list[str] = _build_copy_trim_cmd(path, s, e, out_ext, output)

            rc = pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("trimming_file_progress"),
                _("trimming_file_done"),
                output,
                BATCH_MODE=BATCH_MODE,
                total_duration=he.to_seconds(d),
                mode=1,
            )

            if rc != 0 or (not output.exists() or output.stat().st_size == 0):
                co.print_warning(_("trim_copy_failed_fallback_reencode"))
                cmd = _build_lossless_reencode_cmd(
                    path, s, e, out_ext, output, prefer_hw=PREFER_HW
                )
                cmd = autotune_final_cmd(path, cmd)

                pw.run_ffmpeg_with_progress(
                    path.name,
                    cmd,
                    _("trimming_file_progress"),
                    _("trimming_file_done"),
                    output,
                    BATCH_MODE=BATCH_MODE,
                    total_duration=he.to_seconds(d),
                )
            else:
                co.print_line(
                    "\n  "
                    + _("trimming_file_done").replace(
                        "#ofilename", "\n " + str(output.name) + "\n"
                    ),
                    color="light_green",
                )

        else:
            # 2) Präzise Methode (Plan-API, Re-Encode)
            if quality_preset is None or quality_preset not in defin.CONVERT_PRESET:
                quality_preset = "cinema"

            mode = "preset"
            if defin.CONVERT_PRESET.get(quality_preset, {}).get("lossless"):
                mode = "lossless"

            cmd = _build_precise_reencode_cmd(
                path,
                start_str=s,
                dur_str=d,
                output=output,
                mode=mode,
                preset=quality_preset if mode == "preset" else None,
            )

            cmd = autotune_final_cmd(path, cmd)
            pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("trimming_file_progress"),
                _("trimming_file_done"),
                output,
                BATCH_MODE=BATCH_MODE,
                total_duration=he.to_seconds(d),
            )

        # Thumbnail wieder einbetten (falls vorhanden)
        try:
            if output.exists() and preserved_cover and preserved_cover.exists():
                vt.set_thumbnail(output, value=str(preserved_cover), BATCH_MODE=True)
            else:
                co.print_info(_("no_thumbnail_found"))
        except Exception as e:
            co.print_warning(_("embedding_skipped") + f": {e}")
        try:
            if preserved_cover and preserved_cover.exists():
                preserved_cover.unlink(missing_ok=True)
        except Exception:
            pass

    co.print_finished(_("trimming_method"))
