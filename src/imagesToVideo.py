# imagesToVideo.py
# -----------------------------------------------------------------------------
# Bilder → Video
# -----------------------------------------------------------------------------
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import consoleOutput as co

# local modules
import definitions as defin
import ImagesToVideo_helpers as itvh
import process_wrappers as pw
import userInteraction as ui
import VideoEncodersCodecs as vec
from i18n import _, tr

# ----------------------- Argumente & Defaults -----------------------


def _empty_str_list() -> List[str]:
    """Typed Default-Factory für dataclass (verhindert list[Unknown])."""
    return []


@dataclass
class ImagesToVideoArgs:
    files: List[str] = field(
        default_factory=_empty_str_list
    )  # Einzelbilder ODER Ordner (per he.prepare_inputs)
    preset: Optional[str] = "casual"  # aus defin.CONVERT_PRESET
    format: Optional[str] = "mp4"  # aus defin.CONVERT_FORMAT_DESCRIPTIONS
    resolution: Optional[str] = "original"  # aus defin.RESOLUTIONS
    framerate: Optional[str] = "25"  # "fps" oder Dauer/Bild: "2s", "150ms"
    codec: Optional[str] = "h264"  # Key, z.B. 'h264','hevc','av1', …
    output: Optional[str] = None
    scale: Optional[bool] = None
    duration: Optional[str] = None


def _merge_with_images_defaults(args: Any) -> ImagesToVideoArgs:
    cfg = ImagesToVideoArgs()
    for f in (
        "files",
        "preset",
        "format",
        "resolution",
        "framerate",
        "codec",
        "scale",
        "duration",
    ):
        if hasattr(args, f):
            v = getattr(args, f)
            if v is not None and not (isinstance(v, list) and v == []):
                setattr(cfg, f, v)
    return cfg


def _framerate_was_explicit(original_args: Any, default_val: str = "25") -> bool:
    """
    Versucht zu erkennen, ob die FPS vom Nutzer gesetzt wurde.
    - Bevorzugt Flag original_args.framerate_given (falls dein CLI das setzt).
    - Sonst: Falls Wert != Default ('25'), als explizit werten.
    """
    if hasattr(original_args, "framerate_given"):
        try:
            return bool(getattr(original_args, "framerate_given"))
        except Exception:
            pass
    val = getattr(original_args, "framerate", None)
    if val is None:
        return False
    s = str(val).strip().lower()
    # Wenn exakt Default und kein "…_given"-Flag vorhanden → als NICHT explizit werten
    if s == str(default_val).strip().lower():
        return False
    return s != ""


# ----------------------- Hauptfunktion -----------------------


def images_to_video(args):
    co.print_start(_("images_to_video_method"))

    # 0) Inputs wie bei convert (he.prepare_inputs – aber für Bilder)
    _inp: Tuple[bool, List[str]] = itvh.prepare_inputs_images(args)
    BATCH_MODE: bool = _inp[0]
    image_files: List[str] = _inp[1]

    if BATCH_MODE:
        if not image_files:
            co.print_error(_("no_file_passed"))
            sys.exit(1)
    else:
        if not image_files:
            co.print_warning(_("no_file_passed"))
            return

    num_frames = max(1, len(image_files))

    # 1) Parameter ermitteln (analog zu convert)
    if BATCH_MODE:
        cfg = _merge_with_images_defaults(args)
        preset_choice = cfg.preset or "casual"
        format_choice = cfg.format or "mp4"
        resolution = cfg.resolution or "original"
        fr_raw = str(cfg.framerate or "25")
        codec_key_def = vec.normalize_codec_key(cfg.codec or "h264") or "h264"

        scale_exact = (
            bool(cfg.scale) if getattr(cfg, "scale", None) is not None else False
        )

        # '--resolution WxH'
        resolution, resolution_scale = itvh.parse_resolution_arg_images(resolution)

        # Sonderwerte min/avg/max → "custom"
        _res_token = (resolution or "").strip().lower()
        if _res_token in {"min", "minimal", "small", "smallest"}:
            mw, mh, aw, ah, xw, xh = itvh.compute_image_size_stats(image_files)
            if mw and mh:
                resolution, resolution_scale = "custom", f"{mw}:{mh}"
        elif _res_token in {"avg", "average", "middle"}:
            mw, mh, aw, ah, xw, xh = itvh.compute_image_size_stats(image_files)
            if aw and ah:
                resolution, resolution_scale = "custom", f"{aw}:{ah}"
        elif _res_token in {"max", "maximal", "maximum", "large", "largest"}:
            mw, mh, aw, ah, xw, xh = itvh.compute_image_size_stats(image_files)
            if xw and xh:
                resolution, resolution_scale = "custom", f"{xw}:{xh}"

        # FPS/Dauer interpretieren
        fr_mode, fps_rational, dur_seconds = itvh.parse_fps_or_duration(fr_raw)

        # Optional: explizite Gesamtdauer parsen
        total_seconds: Optional[float] = None
        if getattr(cfg, "duration", None):
            try:
                total_seconds = itvh.parse_total_duration_to_seconds(str(cfg.duration))
            except Exception as e:
                co.print_warning(
                    _("duration_unparsable").format(value=cfg.duration, error=e)
                )

        # Wenn Totaldauer angegeben und FPS nicht explizit → VFR bevorzugen
        enforce_cfr = True
        fr_explicit = _framerate_was_explicit(args)
        if (total_seconds is not None) and (not fr_explicit):
            fr_mode = "dur"
            fps_rational = None
            enforce_cfr = False  # VFR

        # Encoder-Maps
        encoder_maps_by_container = (
            vec.prepare_encoder_maps(
                image_files,
                format_choice,
                defin.CONVERT_FORMAT_DESCRIPTIONS,
                prefer_hw=True,
                ffmpeg_bin="ffmpeg",
            )
            or {}
        )

        codec_choice_by_container: Dict[str, str] = {}
        for c, enc_map in encoder_maps_by_container.items():
            enc_keys = list((enc_map or {}).keys())
            if not enc_keys:
                continue
            if codec_key_def == "copy":
                codec_choice_by_container[c] = (
                    "h264" if "h264" in enc_keys else enc_keys[0]
                )
            elif codec_key_def in enc_keys:
                codec_choice_by_container[c] = codec_key_def
            elif "h264" in enc_keys:
                codec_choice_by_container[c] = "h264"
            else:
                codec_choice_by_container[c] = enc_keys[0]

        # Totaldauer *immer* berechnen (mit max_fps-Cap)
        preset_max_fps = (
            defin.CONVERT_PRESET.get(preset_choice or "casual", {}) or {}
        ).get("max_fps")
        if total_seconds is None:
            if fr_mode == "fps":
                fps_val = itvh.safe_parse_rational(fps_rational or "25")
                if preset_max_fps and fps_val > float(preset_max_fps):
                    fps_val = float(preset_max_fps)
                total_seconds = float(num_frames) / max(0.001, fps_val)
            elif fr_mode == "dur":
                total_seconds = float(num_frames) * float(dur_seconds or 0.04)

        # VFR-Flag für Tabellenausgabe
        is_vfr = (fr_mode in ("dur", "total")) and not (
            fr_mode == "total" and enforce_cfr
        )

        params, labels = itvh.build_params_for_table_images(
            files=image_files,
            preset_choice=preset_choice,
            format_choice=format_choice,
            resolution_key=resolution,
            resolution_scale=resolution_scale,
            codec_choice_by_container=codec_choice_by_container,
            presets=defin.CONVERT_PRESET,
            res_defs=defin.RESOLUTIONS,
            fr_mode=fr_mode,
            fps_rational=fps_rational,
            dur_seconds=dur_seconds,
            scale_exact=scale_exact,
            total_duration=(
                itvh.pretty_hms(total_seconds) if total_seconds is not None else None
            ),
            is_vfr=is_vfr,
        )

    else:
        # ► Interaktiv
        while True:
            preset_keys = [p for p in list(defin.CONVERT_PRESET) if p != "lossless"]
            preset_descriptions = [
                tr((defin.CONVERT_PRESET[k] or {}).get("description", ""))
                for k in preset_keys
            ]
            preset_labels = [
                (defin.CONVERT_PRESET[k] or {}).get("name", k) for k in preset_keys
            ]
            preset_choice = ui.ask_user(
                _("choose_quality_preset"),
                preset_keys,
                preset_descriptions,
                3,
                preset_labels,
                back_button=False,
            )

            _exclude = {"keep", "original"}
            format_keys = [
                k
                for k in defin.CONVERT_FORMAT_DESCRIPTIONS
                if k.lower() not in _exclude
            ]
            format_descriptions = [
                tr((defin.CONVERT_FORMAT_DESCRIPTIONS[k] or {}).get("description", ""))
                for k in format_keys
            ]
            format_labels = [
                tr((defin.CONVERT_FORMAT_DESCRIPTIONS[k] or {}).get("name", k))
                for k in format_keys
            ]
            format_choice = ui.ask_user(
                _("choose_format"), format_keys, format_descriptions, 0, format_labels
            )
            if format_choice is None:
                continue

            encoder_maps_by_container = (
                vec.prepare_encoder_maps(
                    image_files,
                    format_choice,
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                )
                or {}
            )

            codec_choice_by_container: Dict[str, str] = {}
            ans: Optional[str] = None
            for c in list(encoder_maps_by_container.keys()):
                encoders_codec_keys = list((encoder_maps_by_container[c] or {}).keys())
                codec_descriptions = [
                    tr((defin.VIDEO_CODECS.get(e, {}) or {}).get("description", ""))
                    for e in encoders_codec_keys
                ]
                codec_labels = [
                    (defin.VIDEO_CODECS.get(e, {}) or {}).get("name", e)
                    for e in encoders_codec_keys
                ]
                ans = ui.ask_user(
                    _("choose_codec").format(format=c),
                    encoders_codec_keys,
                    codec_descriptions,
                    0,
                    codec_labels,
                )
                if ans is None:
                    break
                if ans == "copy":
                    ans = (
                        "h264"
                        if "h264" in encoders_codec_keys
                        else (encoders_codec_keys[0] if encoders_codec_keys else "h264")
                    )
                codec_choice_by_container[c] = ans
            if ans is None:
                continue
            print()
            co.print_info(_("analysing_source_images"))

            # Auflösung filtern (wie convert)
            min_src_w, min_src_h = vec.probe_min_source_geometry(image_files)
            resolution_keys = list(defin.RESOLUTIONS)
            res_mapping = cast(Mapping[str, Mapping[str, str]], defin.RESOLUTIONS)

            auto_resolution = False
            mw, mh, aw, ah, xw, xh = itvh.compute_image_size_stats(image_files)
            co.delete_last_n_lines(3)
            if mw <= 0 or mh <= 0:
                co.print_warning(_("no_valid_frames_found"))
            else:
                auto_resolution = mw != xw or mh != xh

            extra_keys = ["smallest", "average", "largest"] if auto_resolution else []
            extra_labels = [_(ek) for ek in extra_keys] if auto_resolution else []
            extra_desc = (
                [
                    _("smallest_resolution"),
                    _("average_resolution"),
                    _("largest_resolution"),
                ]
                if auto_resolution
                else []
            )

            filtered_keys = vec.filter_resolution_keys_for_source(
                res_mapping, resolution_keys, min_src_w, min_src_h
            )
            resolution_descriptions = [
                tr((defin.RESOLUTIONS[k] or {}).get("description", ""))
                for k in filtered_keys
            ] + extra_desc
            resolution_labels = [
                (defin.RESOLUTIONS[k] or {}).get("name", k) for k in filtered_keys
            ] + extra_labels
            filtered_keys = filtered_keys + extra_keys
            resolution = ui.ask_user(
                _("choose_resolution"),
                filtered_keys,
                resolution_descriptions,
                0,
                resolution_labels,
            )
            if resolution is None:
                continue

            resolution_scale = None
            if resolution == "custom":
                while True:
                    candidate = ui.ask_two_ints(_("enter_custom_resolution"))  # "W:H"
                    ok, msg = vec.validate_custom_scale_leq(
                        min_src_w,
                        min_src_h,
                        candidate,
                        require_even=True,
                        ignore_boundaries=True,
                    )
                    if ok:
                        resolution_scale = msg
                        break
                    co.print_fail(msg)

            if auto_resolution and resolution in extra_keys:
                if resolution == "smallest" and mw and mh:
                    resolution, resolution_scale = "custom", f"{mw}:{mh}"
                elif resolution == "average" and aw and ah:
                    resolution, resolution_scale = "custom", f"{aw}:{ah}"
                elif resolution == "largest" and xw and xh:
                    resolution, resolution_scale = "custom", f"{xw}:{xh}"

            scale_exact = False
            if resolution != "original":
                scale_options = ("cmd_crop_pad", "cmd_scale")
                scale_labels = [_(op) for op in scale_options]
                scale_descs = [_("crop_or_pad_to_target"), _("scale_to_target")]
                ans = ui.ask_user(
                    _("croppad_or_scale"),
                    scale_options,
                    scale_descs,
                    display_labels=scale_labels,
                )
                if ans is None:
                    continue
                if ans == scale_options[1]:
                    scale_exact = True

            # FPS / Dauer pro Frame / Gesamtdauer (neue Heuristik)
            answer = ui.ask_time_or_other_number(
                _("enter_fps_or_duration"),
                suffix="fps",
                default="25",
                return_raw_if_nonnumber=True,
            )
            if answer is None:
                continue

            # answer liefert (is_time, input_value), aber wir parsen selbst nach den neuen Regeln:
            is_time, raw_value = answer
            mode, fps_rational, dur_per_frame, total_seconds = (
                itvh.parse_interactive_fps_dur_total(str(raw_value))
            )

            enforce_cfr = False
            if mode == "total":
                yn = ui.ask_yes_no(
                    _("enforce_cfr_question"), explanation=_("enforce_cfr_explanation")
                )
                if yn is None:
                    continue
                enforce_cfr = yn

                if enforce_cfr:
                    # FPS erfragen (default 25)
                    ans2 = ui.ask_time_or_other_number(
                        _("choose_cfr_fps"), suffix=None, default="25"
                    )
                    if ans2 is None:
                        continue
                    _tmp, val2 = ans2
                    fps_num = float(val2)
                    fps_rational = (
                        str(int(fps_num))
                        if float(fps_num).is_integer()
                        else str(fps_num)
                    )
                    mode = "fps"  # Wir erzwingen CFR
                else:
                    # VFR-Total: wir rechnen später dur_each = total_seconds / N
                    fps_rational = None
                    # mode bleibt "total"

            if mode == "fps":
                fr_mode, dur_seconds = "fps", None
            elif mode == "dur":
                fr_mode, dur_seconds = "dur", dur_per_frame
            else:  # "total"
                fr_mode, dur_seconds = (
                    "total",
                    total_seconds,
                )  # dur_each wird später aus total_seconds/N berechnet

            # max_fps-Cap berücksichtigen
            preset_max_fps = (
                defin.CONVERT_PRESET.get(preset_choice or "casual", {}) or {}
            ).get("max_fps")

            # Totaldauer *immer* ableiten
            if total_seconds is None:
                if fr_mode == "fps":
                    fps_val = itvh.safe_parse_rational(fps_rational or "25")
                    if preset_max_fps and fps_val > float(preset_max_fps):
                        fps_val = float(preset_max_fps)
                    total_seconds = float(num_frames) / max(0.001, fps_val)
                elif fr_mode == "dur":
                    total_seconds = float(num_frames) * float(dur_seconds or 0.04)

            # VFR-Flag
            is_vfr = (fr_mode == "dur") or (fr_mode == "total")

            params, labels = itvh.build_params_for_table_images(
                files=image_files,
                preset_choice=preset_choice,
                format_choice=format_choice,
                resolution_key=resolution,
                resolution_scale=resolution_scale,
                codec_choice_by_container=codec_choice_by_container,
                presets=defin.CONVERT_PRESET,
                res_defs=defin.RESOLUTIONS,
                fr_mode=fr_mode,
                fps_rational=fps_rational,
                dur_seconds=dur_seconds,
                scale_exact=scale_exact,
                total_duration=(
                    itvh.pretty_hms(total_seconds)
                    if total_seconds is not None
                    else None
                ),
                is_vfr=is_vfr,
            )

            break  # interaktiver Block erfolgreich

    # 2) Status-Tabelle
    co.print_selected_params_table(params, labels=labels)

    # 3) ffconcat-Dauerberechnung
    if total_seconds is not None:
        dur_each = max(0.001, float(total_seconds) / float(num_frames))
        fr_mode_effective = (
            "fps" if (("enforce_cfr" in locals()) and enforce_cfr) else "total"
        )
    else:
        if fr_mode == "fps":
            fps_val = itvh.safe_parse_rational(fps_rational or "30")
            preset_cap = (
                defin.CONVERT_PRESET.get(preset_choice or "casual", {}) or {}
            ).get("max_fps")
            if preset_cap and fps_val > float(preset_cap):
                fps_val = float(preset_cap)
                fps_rational = str(preset_cap)
            dur_each = 1.0 / max(0.001, float(fps_val))
            fr_mode_effective = "fps"
        else:
            dur_each = max(0.001, float(dur_seconds or 0.04))  # default ~25fps
            fr_mode_effective = "dur"

    # total_seconds final (für spätere Nutzung/Debug)
    total_seconds = float(num_frames) * float(dur_each)

    # 4) Gemeinsamer Ordner + Ausgabepfad (respektiert optional args.output)
    common_dir = itvh.get_common_dir(image_files)

    target_container = itvh.target_container(format_choice or "mp4", image_files)
    base_name = common_dir.name or "slideshow"
    out_default = common_dir / f"{base_name}_slideshow.{target_container}"

    out_arg = getattr(args, "output", None)
    if out_arg:
        out_path = Path(str(out_arg)).expanduser()
        if out_path.is_dir():
            out_path = out_path / out_default.name
        elif out_path.suffix.lower().lstrip(".") != target_container:
            out_path = out_path.with_suffix("." + target_container)
    else:
        out_path = out_default

    # 5) Encoder/Container-Setup
    encoder_maps_by_container = (
        vec.prepare_encoder_maps(
            image_files,
            format_choice or "mp4",
            defin.CONVERT_FORMAT_DESCRIPTIONS,
            prefer_hw=True,
            ffmpeg_bin="ffmpeg",
        )
        or {}
    )
    if (format_choice or "mp4") != "keep":
        encoder_map = (
            encoder_maps_by_container.get(target_container, OrderedDict()) or {}
        )
        chosen_codec_key = codec_choice_by_container.get(target_container, "h264")
    else:
        chosen_codec_key = next(
            iter(encoder_maps_by_container.get(target_container, {"h264": None}).keys())
        )
        encoder_map = (
            encoder_maps_by_container.get(target_container, OrderedDict()) or {}
        )

    if chosen_codec_key == "copy":
        chosen_codec_key = (
            "h264"
            if "h264" in encoder_map
            else (next(iter(encoder_map.keys()), "h264"))
        )

    # Container/Codec prüfen
    if not vec.container_allows_codec(target_container, chosen_codec_key):
        sugg = vec.suggest_codec_for_container(target_container)
        co.print_warning(
            _("unsuitable_codec").format(
                codec=chosen_codec_key, container=target_container, sugg=sugg
            )
        )
        chosen_codec_key = sugg

    preferred_encoder = (encoder_map or {}).get(chosen_codec_key)

    # 6) gewünschte Skalierung ableiten
    if resolution == "custom" and resolution_scale:
        req_scale = resolution_scale
    elif resolution != "original":
        req_scale = (defin.RESOLUTIONS.get(resolution or "", {}) or {}).get("scale")
    elif (defin.CONVERT_PRESET.get(preset_choice or "casual", {}) or {}).get("scale"):
        req_scale = (defin.CONVERT_PRESET.get(preset_choice or "casual", {}) or {}).get(
            "scale"
        )
    else:
        req_scale = None

    # 7) ffconcat-Datei schreiben (Helfer, relative Pfade)
    try:
        tmp_file = itvh.write_ffconcat_file(
            image_files,
            dur_each,
            base_dir=common_dir,
            filename_prefix="img2vid_",
            use_relative=True,
        )
    except Exception as e:
        co.print_error(
            _("could_not_write_tempfile").format(error=str(e))
            if _("could_not_write_tempfile")
            else f"Fehler beim Schreiben der temporären Liste: {e}"
        )
        return

    # 8) Plan-API verwenden (scale_exact steuert, ob *skaliert* oder *croppad* angewandt wird)
    min_w, min_h = vec.probe_min_source_geometry(image_files)
    spec = defin.CONVERT_PRESET.get(preset_choice or "casual", {}) or {}
    preset_max_fps = spec.get("max_fps")

    user_fps_rat = fps_rational if (fr_mode_effective == "fps") else None
    src_fps_float = itvh.safe_parse_rational(user_fps_rat) if user_fps_rat else None

    # Wenn "scale_exact": hart auf Zielauflösung skalieren (kein AR)
    # Sonst: croppad (AR erhalten) -> req_scale=None im Plan, VF-Chain nachträglich injizieren.
    plan_req_scale = req_scale if (req_scale and scale_exact) else None
    plan_preserve_ar = False if (req_scale and scale_exact) else True

    plan = vec.build_transcode_plan(
        input_path=tmp_file,
        target_container=target_container,
        preset_name=str(preset_choice or "casual"),
        codec_key=str(chosen_codec_key),
        preferred_encoder=preferred_encoder,
        req_scale=plan_req_scale,
        src_w=int(min_w or 0),
        src_h=int(min_h or 0),
        src_fps=src_fps_float,
        user_fps_rational=user_fps_rat,
        preset_max_fps=int(preset_max_fps) if preset_max_fps is not None else None,
        force_key_at_start=True,
        preserve_ar=plan_preserve_ar,
    )
    cmd = vec.assemble_ffmpeg_cmd(plan, out_path)

    # Croppad-Pfad: VF-Filterkette anhängen (identische Achs-Logik wie croppad)
    if req_scale and not scale_exact:
        tw, th = itvh.parse_wh_from_scale(req_scale)  # "W:H" -> (W,H)
        transparent_pad = itvh.detect_alpha_from_first(image_files)
        vf = itvh.build_croppad_vf_images(
            target_w=tw,
            target_h=th,
            offset_x=0,
            offset_y=0,
            transparent_pad=transparent_pad,
        )
        itvh.ensure_merge_vf(cmd, out_path, vf)

    # 8.1) concat-Demuxer vor den Input injizieren (mit -safe 0)
    itvh.inject_concat_demuxer_for_input(cmd, tmp_file)

    # CFR/VFR: FPS → cfr, Dauer/Frame & Total → vfr
    if fr_mode_effective == "fps":
        target_fps = str(fps_rational or "30")
        if "-r" not in cmd:
            itvh.insert_output_opts(cmd, out_path, ["-r", target_fps])
        if "-vsync" not in cmd:
            itvh.insert_output_opts(cmd, out_path, ["-vsync", "cfr"])
    else:
        if "-vsync" not in cmd:
            itvh.insert_output_opts(cmd, out_path, ["-vsync", "vfr"])

    # 8.3) PNG-Codec → Alpha sicher mitnehmen
    if (str(chosen_codec_key).lower() == "png") and ("-pix_fmt" not in cmd):
        cmd += ["-pix_fmt", "rgba"]

    # 9) Ausführen
    total_frames = len(image_files)
    fps_hint = (1.0 / dur_each) if dur_each > 0 else None

    co.print_debug("ffmpeg", cmd=cmd)

    subprocess.run(cmd)

    pw.run_ffmpeg_with_progress(
        Path(image_files[0]).name,  # rein für die Anzeige (#ifilename)
        cmd,
        _("creating_video_progress"),
        _("creating_video_done"),
        out_path,
        BATCH_MODE=BATCH_MODE,
        total_frames=total_frames,
        fps_hint=fps_hint,
    )

    # 10) Aufräumen
    try:
        Path(tmp_file).unlink(missing_ok=True)
    except Exception:
        pass

    co.print_finished(_("images_to_video_method"))
