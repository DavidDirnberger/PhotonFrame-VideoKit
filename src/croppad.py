#!/usr/bin/env python3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, cast

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
class CropPadArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    resolution: Optional[str] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    output: Optional[str] = None


# -----------------------
# Hilfsfunktionen
# -----------------------


# sehr robuste Fallback-Analyse der CLI-Token
def _fallback_offsets_from_argv() -> tuple[Optional[int], Optional[int]]:
    argv = sys.argv[1:]
    ox: Optional[int] = None
    oy: Optional[int] = None
    pair_raw: Optional[str] = None

    def _to_int(s: str | None) -> Optional[int]:
        if s is None:
            return None
        try:
            return int(s.strip())
        except Exception:
            return None

    i = 0
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None

        # --offset=40:10  |  --offset 40:10
        if tok.startswith("--offset="):
            pair_raw = tok.split("=", 1)[1]
        elif tok in ("--offset", "-O") and nxt and not nxt.startswith("-"):
            pair_raw = nxt
            i += 1

        # --offset-x=40  |  --offset-x 40  (auch alias)
        elif tok.startswith("--offset-x="):
            ox = _to_int(tok.split("=", 1)[1])
        elif (
            tok in ("--offset-x", "--offset_x", "--offx", "--padx")
            and nxt
            and not nxt.startswith("-")
        ):
            ox = _to_int(nxt)
            i += 1

        # --offset-y=10  |  --offset-y 10  (auch alias)
        elif tok.startswith("--offset-y="):
            oy = _to_int(tok.split("=", 1)[1])
        elif (
            tok in ("--offset-y", "--offset_y", "--offy", "--pady")
            and nxt
            and not nxt.startswith("-")
        ):
            oy = _to_int(nxt)
            i += 1

        i += 1

    if pair_raw:
        px, py = _parse_offset_pair(pair_raw)
        # Nur setzen, wenn bisher None
        ox = px if ox is None else ox
        oy = py if oy is None else oy

    return ox, oy


def _parse_custom_resolution(raw: str) -> tuple[int, int] | None:
    if not raw:
        return None
    s = raw.lower().strip()
    for sep in ["x", "X", ",", ".", ":", "/", "-", "#", "+", "*"]:
        s = s.replace(sep, "x")
    parts = [p.strip() for p in s.split("x") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        w = int(parts[0])
        h = int(parts[1])
        if w <= 0 or h <= 0:
            return None
        return (w, h)
    except ValueError:
        return None


def _wh_from_scale(scale_val) -> tuple[int, int] | None:
    if scale_val is None:
        return None
    if isinstance(scale_val, (tuple, list)) and len(scale_val) == 2:
        try:
            w = int(scale_val[0])
            h = int(scale_val[1])
            return (w, h) if w > 0 and h > 0 else None
        except Exception:
            return None
    return _parse_custom_resolution(str(scale_val))


def _resolve_target_resolution(
    res_key_or_custom: Optional[str],
    interactive: bool,
    hide_original_in_menu: bool = False,
) -> tuple[str, int | None, int | None]:
    res_map = getattr(defin, "RESOLUTIONS", {})
    keys = list(res_map.keys())

    # 1) Direkte Übergabe
    if res_key_or_custom:
        given = res_key_or_custom.strip().lower()
        if given in res_map:
            if given == "original":
                return "original", None, None
            if given == "custom":
                if interactive:
                    co.print_info(_("croppad_enter_custom_info"))
                    raw = input(
                        co.return_promt(
                            "   " + _("croppad_prompt_custom_resolution") + ": "
                        )
                    ).strip()
                    wh = _parse_custom_resolution(raw)
                    if not wh:
                        co.print_fail(_("croppad_invalid_resolution"))
                        return "original", None, None
                    return "custom", wh[0], wh[1]
                else:
                    co.print_fail(_("croppad_custom_requires_interactive"))
                    return "original", None, None
            wh = _wh_from_scale(res_map[given].get("scale"))
            if wh:
                return given, wh[0], wh[1]
            return "original", None, None
        else:
            wh = _parse_custom_resolution(res_key_or_custom)
            if wh:
                return "custom", wh[0], wh[1]
            if interactive:
                co.print_fail(_("croppad_invalid_resolution"))
            return "original", None, None

    # 2) Interaktives Menü
    if interactive:
        options_all = keys
        options = (
            [k for k in options_all if k != "original"]
            if hide_original_in_menu
            else options_all[:]
        )
        if not options:
            return "original", None, None

        labels = [
            (
                tr(res_map[k]["name"])
                if isinstance(res_map[k].get("name"), (dict, str))
                else k
            )
            for k in options
        ]
        descs = [
            tr(res_map[k]["description"]) if "description" in res_map[k] else ""
            for k in options
        ]

        choice = ui.ask_user(
            _("croppad_choose_resolution"),
            options=options,
            descriptions=descs,
            display_labels=labels,
            default=4,
        )
        if choice is None:
            return "original", None, None
        if choice == "custom":
            co.print_info(_("croppad_enter_custom_info"))
            raw = input(
                co.return_promt("   " + _("croppad_prompt_custom_resolution") + ": ")
            ).strip()
            wh = _parse_custom_resolution(raw)
            if not wh:
                co.print_fail(_("croppad_invalid_resolution"))
                return "original", None, None
            return "custom", wh[0], wh[1]

        wh = _wh_from_scale(res_map[choice].get("scale"))
        if wh:
            return choice, wh[0], wh[1]
        return "original", None, None

    # 3) Fallback
    return "original", None, None


def _from_args_any(args: Any, *names: str) -> Any:
    """
    Robust: liest Attribute oder Dict-Keys (z.B. aus argparse/click),
    probiert mehrere Namensvarianten.
    """
    # 1) Attribute
    for n in names:
        if hasattr(args, n):
            v = getattr(args, n)
            if v is not None and v != "":
                return v
    # 2) Dict (falls vorhanden)
    try:
        d = dict(args) if not isinstance(args, dict) else args
    except Exception:
        d = None
    if isinstance(d, dict):
        for n in names:
            if n in d and d[n] is not None and d[n] != "":
                return d[n]
    return None


def _parse_offset_pair(raw: str | int | None) -> tuple[Optional[int], Optional[int]]:
    """
    Erlaubt:
      - "40:10", "40,10", "40x10" etc.
      - einzelne ints (nur X)
      - None
    """
    if raw is None:
        return None, None
    if isinstance(raw, int):
        return raw, None
    s = str(raw).strip()
    if s == "":
        return None, None
    # Trennzeichen normalisieren
    for sep in ("x", "X", ":", ",", "/", ";", "|"):
        s = s.replace(sep, ":")
    parts = [p.strip() for p in s.split(":") if p.strip()]
    try:
        if len(parts) == 1:
            return int(parts[0]), None
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None
    return None, None


def _read_offsets_from_args(args: Any) -> tuple[Optional[int], Optional[int]]:
    combo = _from_args_any(args, "offset", "offset_xy", "offsetxy")
    ox_c, oy_c = _parse_offset_pair(combo)

    ox = _from_args_any(args, "offset_x", "offsetx", "offx", "padx")
    oy = _from_args_any(args, "offset_y", "offsety", "offy", "pady")

    def _to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    ox = _to_int(ox) if ox is not None else ox_c
    oy = _to_int(oy) if oy is not None else oy_c

    # HARTER FALLBACK: direkt aus sys.argv lesen, wenn beides weiterhin None ist
    if ox is None and oy is None:
        ax, ay = _fallback_offsets_from_argv()
        ox = ax if ox is None else ox
        oy = ay if oy is None else oy

    return ox, oy


def _resolve_offsets(
    x: Optional[int], y: Optional[int], interactive: bool
) -> tuple[int, int]:
    """
    Holt/prüft Offsets. Interaktiv werden sie abgefragt (Enter=0).
    Positives X/Y: nach rechts/unten. Negative Werte sind erlaubt,
    werden aber auf gültige Grenzen geclamped, sobald wir die Zielgröße kennen.
    """

    def _ask_int(prompt_key: str) -> int:
        raw = input(co.return_promt("   " + _(prompt_key) + " ")).strip()
        if raw == "":
            return 0
        try:
            return int(raw)
        except ValueError:
            co.print_warning(_("croppad_offset_invalid"))
            return 0

    if x is None and interactive:
        x = _ask_int("croppad_prompt_offset_x")
    if y is None and interactive:
        y = _ask_int("croppad_prompt_offset_y")

    try:
        xx = int(x) if x is not None else 0
    except Exception:
        xx = 0
    try:
        yy = int(y) if y is not None else 0
    except Exception:
        yy = 0
    return xx, yy


# -----------------------
# Hauptfunktion
# -----------------------


def crop_pad(args: Any) -> None:
    """
    Crop+Pad mit *achsweiser* Logik:
      • target < source → crop
      • target = source → nichts
      • target > source → pad

    Offsets:
      • Beim Crop verschieben Offsets das Cropfenster.
      • Beim Pad verschieben Offsets die Bildposition auf der Zielleinwand.
    """
    # Interaktiv vs. Batch
    BATCH_MODE, files = _prepare_inputs_typed(args)
    co.print_start(_("croppad"))

    # Zielauflösung lesen
    chosen_key, target_w, target_h = _resolve_target_resolution(
        getattr(args, "resolution", None),
        interactive=(not BATCH_MODE),
        hide_original_in_menu=True,
    )

    # Offsets aus Flags lesen (robust) und ggf. interaktiv erfragen
    raw_ox, raw_oy = _read_offsets_from_args(args)
    offset_x, offset_y = _resolve_offsets(raw_ox, raw_oy, interactive=(not BATCH_MODE))

    co.print_selected_params_table(
        {
            "resolution": f"{target_w}:{target_h}",
            "offset": f"{offset_x}:{offset_y}",
        }
    )

    for i, file in enumerate(files, start=1):
        path = Path(file)
        if not path.exists():
            co.print_warning(_("file_not_found").format(file=path.name))
            continue

        # Cover/Thumbnail konservieren
        preserved_cover: Path | None = None
        try:
            if vt.check_thumbnail(path, silent=True):
                preserved_cover = vt.extract_thumbnail(path)
        except Exception:
            preserved_cover = None

        # Quelle inspizieren
        in_w, in_h, _src_pf = vec.ffprobe_geometry(path)
        try:
            in_w = int(in_w or 0)
            in_h = int(in_h or 0)
        except Exception:
            in_w, in_h = 0, 0

        if chosen_key == "original" or target_w is None or target_h is None:
            co.print_info(_("croppad_skip_original").format(filename=path.name))
            continue
        if in_w <= 0 or in_h <= 0:
            co.print_fail(_("croppad_probe_failed"))
            continue

        # ========= CROPPING-PHASE (achsweise) =========
        crop_w = in_w if target_w >= in_w else target_w
        crop_h = in_h if target_h >= in_h else target_h

        # Crop-Fenster zentriert + Offset, aber nur dort anwenden, wo wirklich gecroppt wird
        crop_x = 0
        if crop_w < in_w:
            base_cx = (in_w - crop_w) // 2
            crop_x = base_cx + (offset_x or 0)
            crop_x = max(0, min(crop_x, in_w - crop_w))
        crop_y = 0
        if crop_h < in_h:
            base_cy = (in_h - crop_h) // 2
            crop_y = base_cy + (offset_y or 0)
            crop_y = max(0, min(crop_y, in_h - crop_h))

        vf_parts: List[str] = []
        did_crop = (crop_w < in_w) or (crop_h < in_h)
        if did_crop:
            vf_parts.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")

        # ========= PADDING-PHASE (achsweise) =========
        pad_needed_w = target_w > crop_w
        pad_needed_h = target_h > crop_h

        pad_x = 0
        pad_y = 0
        if pad_needed_w or pad_needed_h:
            base_px = max((target_w - crop_w) // 2, 0)
            base_py = max((target_h - crop_h) // 2, 0)

            # Offsets immer relativ zur Mittelposition, dann clampen
            pad_x = base_px + (offset_x or 0)
            pad_y = base_py + (offset_y or 0)
            pad_x = max(0, min(pad_x, max(target_w - crop_w, 0)))
            pad_y = max(0, min(pad_y, max(target_h - crop_h, 0)))

            # NEU: transparentes Padding bei Alpha-Quellen
            def _pixfmt_has_alpha(pf: str | None) -> bool:
                if not pf:
                    return False
                p = pf.lower()
                return (
                    p.startswith(
                        ("rgba", "bgra", "argb", "abgr", "gbrap", "yuva", "ya")
                    )
                    or "a" in p  # pragmatischer Fallback
                )

            transparent_pad = _pixfmt_has_alpha(_src_pf)

            pad_color = "black@0" if transparent_pad else "black"

            vf_parts.append(f"pad={target_w}:{target_h}:{pad_x}:{pad_y}:{pad_color}")

        vfilt = ",".join(vf_parts) if vf_parts else None

        # Fortschritts-Texte
        if did_crop and (pad_needed_w or pad_needed_h):
            progress_line = f"{_('croppad_progress_cropping')} + {_('croppad_progress_padding')} #ifilename → {target_w}x{target_h}"
            finished_line = f"{_('croppad_done_croppad')} #ofilename"
        elif did_crop:
            progress_line = (
                f"{_('croppad_progress_cropping')} #ifilename → {target_w}x{target_h}"
            )
            finished_line = f"{_('croppad_done_cropping')} #ofilename"
        elif pad_needed_w or pad_needed_h:
            progress_line = (
                f"{_('croppad_progress_padding')} #ifilename → {target_w}x{target_h}"
            )
            finished_line = f"{_('croppad_done_padding')} #ofilename"
        else:
            co.print_info(
                _("croppad_skip_same_res").format(
                    filename=path.name, w=target_w, h=target_h
                )
            )
            continue

        # Ausgabepfad/Container
        out_path = fs.build_output_path(
            input_path=path,
            output_arg=getattr(args, "output", None),
            default_suffix=f"_croppad_{target_w}x{target_h}",
            idx=i,
            total=len(files),
        )
        container = out_path.suffix.lstrip(".").lower() or "mkv"

        # Codec minimal-invasiv: Quellcodec bevorzugen (kein Cross-Codec)
        src_vcodec = vec.probe_video_codec(path) or "h264"
        desired_key = vec.normalize_codec_key(
            src_vcodec
        ) or vec.suggest_codec_for_container(container)
        try:
            desired_key, enc_hint = vec.resolve_codec_key_with_container_fallback(
                container, desired_key, allow_cross_codec_fallback=False
            )
        except Exception:
            enc_hint = None

        # Preset
        preset_name = getattr(defin, "CROPPAD_PRESET_NAME", "casual")

        # Transcode-Plan (ohne Auto-Scale)
        plan_obj: object | None = None
        try:
            plan_obj = vec.build_transcode_plan(
                input_path=path,
                target_container=container,
                preset_name=preset_name,
                codec_key=desired_key,
                preferred_encoder=enc_hint,
                req_scale=None,
                src_w=None,
                src_h=None,
                src_fps=None,
                user_fps_rational=None,
                preset_max_fps=None,
                force_key_at_start=False,
                preserve_ar=True,
            )
            cmd = list(getattr(plan_obj, "final_cmd_without_output"))
        except Exception:
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
                "0:v:0",
                "-map",
                "0:a?",
                "-map",
                "0:s?",
                "-dn",
                "-c:v",
                vec.encoder_for_codec(desired_key),
                "-c:a",
                "copy",
                "-c:s",
                "copy",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
            ]

        # Unsere Filter anfügen/mergen
        if vfilt:
            if "-vf" in cmd:
                j = cmd.index("-vf")
                existing = cmd[j + 1]
                try:
                    new_vf = vec.vf_join(vfilt, existing)
                except Exception:
                    new_vf = (
                        (vfilt + ("," + existing if existing else ""))
                        if vfilt
                        else existing
                    )
                cmd[j + 1] = new_vf
            else:
                plan_vf = (
                    getattr(plan_obj, "vf_chain", None)
                    if plan_obj is not None
                    else None
                )
                if plan_vf:
                    try:
                        new_vf = vec.vf_join(vfilt, plan_vf)
                    except Exception:
                        new_vf = (
                            (vfilt + ("," + plan_vf if plan_vf else ""))
                            if vfilt
                            else plan_vf
                        )
                    cmd += ["-vf", new_vf]
                else:
                    cmd += ["-vf", vfilt]

        # Wenn Filter aktiv sind, darf -c:v nicht 'copy' sein
        try:
            ci = cmd.index("-c:v")
            if ci + 1 < len(cmd) and cmd[ci + 1].strip().lower() == "copy":
                cmd[ci + 1] = enc_hint or vec.encoder_for_codec(desired_key)
        except ValueError:
            cmd += ["-c:v", (enc_hint or vec.encoder_for_codec(desired_key))]

        # Last-Mile
        try:
            if plan_obj is not None:
                cmd = vec.postprocess_cmd_all_presets(cmd, plan_obj)
        except Exception:
            try:
                vec.dedupe_vf_inplace(cmd)
            except Exception:
                pass

        # Ausführen mit Fortschritt

        cmd = cmd + [str(out_path)]

        cmd = autotune_final_cmd(path, cmd)

        print()

        pw.run_ffmpeg_with_progress(
            input_file=path.name,
            ffmpeg_cmd=cmd,
            progress_line=progress_line,
            finished_line=finished_line,
            output_file=out_path,
            BATCH_MODE=BATCH_MODE,
        )

        # Thumbnail wieder einbetten
        try:
            if out_path.exists() and preserved_cover and preserved_cover.exists():
                vt.set_thumbnail(out_path, value=str(preserved_cover), BATCH_MODE=True)
            else:
                co.print_info(_("no_thumbnail_found"))
        except Exception as e:
            co.print_warning(_("embedding_skipped") + f": {e}")
        try:
            if preserved_cover and preserved_cover.exists():
                preserved_cover.unlink(missing_ok=True)
        except Exception:
            pass

    co.print_finished(_("cropping_method"))
