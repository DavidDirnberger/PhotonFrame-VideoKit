#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, cast

import consoleOutput as co
import fileSystem as fs
import helpers as he
import process_wrappers as pw
import userInteraction as ui
import video_thumbnail as vt
import VideoEncodersCodecs as vc
from ffmpeg_perf import autotune_final_cmd

# local modules
from i18n import _ as _i18n

# Faktor- oder FPS-Parser: akzeptiert
#   - "num/den" (z. B. 30000/1001)  → absolute FPS
#   - "59.94" oder "60"            → absolute FPS
#   - "2x", "1.5x"                 → Faktor
_FACTOR_OR_FPS_RE = re.compile(
    r"^\s*(?:(?P<num>\d+)\s*/\s*(?P<den>\d+)|(?P<float>\d+(?:\.\d+)?)\s*(?P<x>[xX])?)\s*$"
)
# ---------- getypte Helfer gegen list[Unknown] ----------


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


def _pf_has_alpha(pf: Optional[str]) -> bool:
    p = (pf or "").lower()
    return any(x in p for x in ("rgba", "bgra", "argb", "gbrap", "yuva"))


@dataclass
class InterpolateArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    factor: Optional[str] = "2x"
    quality: Optional[str] = "std"
    output: Optional[str] = None


QUALITY_PRESETS = {"std": "Standard", "hq": "High Quality", "max": "Maximum"}


def _fps_to_str(fps_float: float) -> str:
    common = [(23.976, "24000/1001"), (29.97, "30000/1001"), (59.94, "60000/1001")]
    for val, s in common:
        if abs(fps_float - val) < 0.02:
            return s
    if abs(fps_float - round(fps_float)) < 1e-6:
        return str(int(round(fps_float)))
    frac = Fraction(fps_float).limit_denominator(1001)
    return f"{frac.numerator}/{frac.denominator}"


def interpolate(args: Any) -> None:
    # 0) INTERAKTIV vs. BATCH + Eingaben
    BATCH_MODE, files = _prepare_inputs_typed(args)
    co.print_start(_i18n("interpolate_method"))

    # Typisierte Zielvariablen
    is_factor: bool
    value: float
    q_raw: str

    if BATCH_MODE:
        raw = str(getattr(args, "factor", "2x") or "2x")

        ans = ui.ask_factor_or_number(
            prompt=None,
            error_message=None,  # im Batch keine roten Meldungen ausgeben
            user_input=raw,
            suffix="x",
        )
        if ans is None:
            # Fallback-Default im Batch (defensiv): 2x
            is_factor, value = True, 2.0
        else:
            is_factor, value = ans

        q_raw = str(getattr(args, "quality", "std") or "std").strip().lower()
    else:
        q_keys = list(QUALITY_PRESETS)
        q_label = list(QUALITY_PRESETS.values())
        while True:
            ans = ui.ask_factor_or_number(
                prompt=_i18n("enter_factor_or_fps"),
                error_message=_i18n("invalid_framerate_input"),
            )
            if ans is None:
                continue
            is_factor, value = ans
            q_desc = list(_i18n(k + "_quality") for k in q_keys)
            q_sel = ui.ask_user(
                _i18n("choose_quality_preset"),
                q_keys,
                display_labels=q_label,
                descriptions=q_desc,
            )
            if q_sel is None:
                continue
            q_raw = str(q_sel).strip().lower()
            break

    # Quali-Profil bestimmen (std | hq | max)
    QUALITY = (
        "max"
        if q_raw in ("max", "ultra", "cinema")
        else ("hq" if q_raw in ("hq", "high") else "std")
    )

    co.print_value_info(_i18n("quality"), QUALITY_PRESETS[QUALITY])
    desc_key = QUALITY + "_quality"
    co.print_info(_i18n(desc_key))

    # preset_name = _pick_preset()
    preset_name = "ultra"

    total = len(files)
    for i, file in enumerate(files):
        path = Path(file)
        if not path.exists():
            co.print_warning(_i18n("file_not_found").format(file=path.name))
            continue

        # Thumbnail ggf. bewahren
        preserved_cover: Optional[Path] = None
        try:
            if vt.check_thumbnail(path, silent=True):
                preserved_cover = vt.extract_thumbnail(path)
        except Exception:
            preserved_cover = None

        # Original-FPS als Bruch lesen
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            rate_str = probe.stdout.strip() or "0/1"
            orig_frac = Fraction(rate_str)
            original_fps = float(orig_frac)
        except Exception:
            co.print_warning(_i18n("could_not_determine_fps").format(file=path.name))
            continue

        # Ziel-FPS
        if is_factor:
            target_fps = float(orig_frac * Fraction(value).limit_denominator(1001))
        else:
            target_fps = float(value)

        fps_disp_old = he.format_fps(original_fps)
        fps_disp_new = he.format_fps(target_fps)
        file_info = _i18n("fps_change_summary").format(
            original=co.return_promt(f"{fps_disp_old} FPS", color="bright_blue"),
            target=co.return_promt(f"{fps_disp_new} FPS", color="bright_blue"),
        )
        print()
        co.print_file_info(str(path.name), file_info)

        suffix = f"_interpolated_{he.format_fps(target_fps, decimals=0)}fps"
        output = fs.build_output_path(
            input_path=path,
            output_arg=getattr(args, "output", None),
            default_suffix=suffix,
            idx=i,
            total=total,
            target_ext=None,
        )

        # Container/Codec beibehalten
        container = vc.detect_container_from_path(path) or (
            path.suffix.lstrip(".").lower() or "mkv"
        )
        src_codec = vc.probe_video_codec(path) or "h264"
        codec_key = vc.normalize_codec_key(src_codec) or "h264"
        fps_arg = _fps_to_str(target_fps)

        # Plan erzeugen
        try:
            plan = vc.build_transcode_plan(
                input_path=path,
                target_container=container,
                preset_name=preset_name,
                codec_key=codec_key,
                preferred_encoder=None,
                req_scale=None,
                src_w=None,
                src_h=None,
                src_fps=original_fps,
                user_fps_rational=fps_arg,
                preset_max_fps=None,
                force_key_at_start=True,
                preserve_ar=True,
            )
        except Exception as e:
            co.print_error(_i18n("unexpected_error") + f" (plan): {e}")
            continue

        # Basis-Command vom Plan
        cmd: List[str] = list(plan.final_cmd_without_output)

        # Postprocess (Encoder-Quirks etc.) VOR unserem Filterbau laufen lassen
        cmd = vc.postprocess_cmd_all_presets(cmd, plan)

        # ======== HQ/Max – Vorfilter (mildes Denoising für stabilere MVs) ========
        pre_filters: List[str] = []
        try:
            if QUALITY in ("hq", "max") and vc._has_filter("hqdn3d"):
                pre_filters.append(
                    "hqdn3d=1.5:1.5:6:6" if QUALITY == "hq" else "hqdn3d=2:2:9:9"
                )
        except Exception:
            pass

        # ======== Interpolation (hochwertig) ========
        fps_arg = _fps_to_str(target_fps)
        mi = f"minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:fps={fps_arg}"

        # HQ/Max: optional sanfte Nachschärfung
        post_filters: List[str] = []
        try:
            if QUALITY == "max" and vc._has_filter("unsharp"):
                post_filters.append("unsharp=3:3:0.3:3:3:0.3")
        except Exception:
            pass

        # --- Quelle auf Alphakanal prüfen ---
        _tmp_w, _tmp_h, src_pf = vc.ffprobe_geometry(path)
        has_alpha = _pf_has_alpha(src_pf)
        cont = (container or "").lower()
        container_supports_alpha = cont in ("mkv", "matroska", "mov", "avi")

        # Falls der Plan vorher ein -vf gesetzt hat, entfernen – wir benutzen -filter_complex
        if "-vf" in cmd:
            j = cmd.index("-vf")
            # entferne -vf und sein Argument
            del cmd[j : j + 2]

        if has_alpha and container_supports_alpha:
            # Wir splitten RGB & Alpha, interpolieren beide und mergen zurück zu gbrap
            base_chain_parts: List[str] = []
            if pre_filters:
                base_chain_parts.append(",".join(pre_filters))
            base_chain_parts.append("format=gbrp")
            base_chain_parts.append(mi)
            if post_filters:
                base_chain_parts.append(",".join(post_filters))
            base_chain = ",".join(base_chain_parts)

            # Alpha-Zweig: Alpha herausziehen → Graustufen → interpolieren
            alpha_chain = "alphaextract,format=gray," + mi

            # WICHTIG: kein Komma direkt hinter einem Label wie [base] / [a]
            fc = (
                " [0:v]split=2[base][a];"
                f"[base]{base_chain}[bv];"
                f"[a]{alpha_chain}[av];"
                "[bv][av]alphamerge,format=argb[outv]"
            ).strip()

            # Filter-Complex setzen
            cmd += ["-filter_complex", fc]

            # Video-Mapping auf unseren gelabelten Output
            # Ersetze erstes "-map 0:v:0" (falls vorhanden) durch "-map [outv]"
            try:
                k = 0
                replaced = False
                while k < len(cmd) - 1:
                    if cmd[k] == "-map" and str(cmd[k + 1]).startswith("0:v"):
                        cmd[k + 1] = "[outv]"
                        replaced = True
                        break
                    k += 1
                if not replaced:
                    cmd += ["-map", "[outv]"]
            except Exception:
                cmd += ["-map", "[outv]"]

            # CFR-Muxing wie gehabt
            if "-vsync" not in cmd:
                cmd += ["-vsync", "2"]

        else:
            # Standard-Fall (kein Alpha oder Container ohne Alpha) → einfaches -vf
            vf_chain_parts: List[str] = []
            if pre_filters:
                vf_chain_parts.extend(pre_filters)
            vf_chain_parts.append(mi)
            if post_filters:
                vf_chain_parts.extend(post_filters)
            vf_append = ",".join(vf_chain_parts)

            # sicherstellen, dass kein altes -vf mehr drin ist
            if "-vf" in cmd:
                j = cmd.index("-vf")
                del cmd[j : j + 2]
            cmd += ["-vf", vf_append]

            if "-vsync" not in cmd:
                cmd += ["-vsync", "2"]

        # Finalen Command zeigen & ausführen
        # co.print_debug('ffmpeg_cmd', cmd=cmd)
        cmd = autotune_final_cmd(path, cmd + [str(output)])

        pw.run_ffmpeg_with_progress(
            path.name,
            cmd,
            _i18n("interpolation_in_progress").format(
                oFPS=he.format_fps(original_fps), nFPS=he.format_fps(target_fps)
            ),
            _i18n("interpolation_done").format(FPS=he.format_fps(target_fps)),
            output,
            BATCH_MODE=BATCH_MODE,
        )

        # Thumbnail wieder einbetten (falls vorhanden)
        try:
            if output.exists() and preserved_cover and preserved_cover.exists():
                vt.set_thumbnail(output, value=str(preserved_cover), BATCH_MODE=True)
            else:
                co.print_info(_i18n("no_thumbnail_found"))
        except Exception as e:
            co.print_warning(_i18n("embedding_skipped") + f": {e}")
        try:
            if preserved_cover and preserved_cover.exists():
                preserved_cover.unlink(missing_ok=True)
        except Exception:
            pass

    co.print_finished(_i18n("interpolate_method"))
