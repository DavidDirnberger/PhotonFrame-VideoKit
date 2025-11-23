#!/usr/bin/env python3
from __future__ import annotations

import re
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

# --------- kleine, getypte Helfer (Unknown → konkret) ---------


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
class ScaleArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    resolution: Optional[str] = None  # Key in RESOLUTIONS, "custom" oder "W:H"
    output: Optional[str] = None  # Zielpfad/-name (optional)
    preserve_ar: Optional[bool] = None  # True/False; None => interaktiv fragen


# -----------------------
# Scale-Helpers
# -----------------------


def _get_flag_bool(args: Any, *names: str) -> Optional[bool]:
    """
    Liest ein Bool-Flag robust aus verschiedenen Namen (z. B. preserve_ar, ar).
    Akzeptiert True/False, 1/0, yes/no, on/off.
    """

    def _to_bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "on", "y"}:
            return True
        if s in {"0", "false", "no", "off", "n"}:
            return False
        return None

    for n in names:
        if hasattr(args, n):
            b = _to_bool(getattr(args, n))
            if b is not None:
                return b
    return None


def _force_exact_scale_and_square_pixels_inplace(
    cmd: List[str], req_w: int, req_h: int, force_sar1: bool
) -> None:
    """
    Erzwingt finale Dimensionen in der Filterkette:
    - entfernt vorhandene 'scale=', 'setdar=', 'setsar='
    - hängt 'scale=req_w:req_h' ans Ende
    - optional 'setsar=1' ans Ende
    Legt -vf neu an, falls nicht vorhanden.
    """
    # finde -vf
    try:
        i = cmd.index("-vf") + 1
        vf = cmd[i]
    except ValueError:
        # -vf fehlt → neu anlegen
        vf = ""
        cmd.extend(["-vf", ""])

        i = cmd.index("-vf") + 1  # neuer Index
    parts = [p.strip() for p in vf.split(",") if p.strip()] if vf else []

    # alte scale/setdar/setsar entfernen
    parts = [
        p
        for p in parts
        if not (
            p.startswith("scale=") or p.startswith("setdar=") or p.startswith("setsar=")
        )
    ]

    # exakte Skalierung anhängen
    parts.append(f"scale={req_w}:{req_h}")
    if force_sar1:
        parts.append("setsar=1")

    cmd[i] = ",".join(parts)


def _parse_two_ints_any_sep(raw: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extrahiert die ersten zwei Ganzzahlen aus einem String mit beliebigen Trennzeichen.
    Beispiele: '1920x1080', '1920 1080', '1920/1080', '1920-1080', '1920,1080', ...
    """
    nums = re.findall(r"[+-]?\d+", raw or "")
    if len(nums) < 2:
        return None, None
    try:
        return int(nums[0]), int(nums[1])
    except Exception:
        return None, None


def _resolve_resolution_key_and_scale(
    user_value: Optional[str], interactive: bool, hide_original_in_menu: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Ermittelt (res_key, res_scale_str). res_scale_str → "W:H".
    Kein UI im Batch (interactive=False). Akzeptiert beliebige Trennzeichen.
    """
    res_defs = getattr(defin, "RESOLUTIONS", {})
    keys_all = list(res_defs)

    # Case-insensitive Lookup + einfache Aliase (4k, 8k, 2160p, 4320p, …)
    norm_map = {k.lower(): k for k in res_defs.keys()}
    alias_map = {
        "2160p": "4k",
        "uhd": "4k",
        "uhd-1": "4k",
        "4320p": "8k",
        "fuhd": "8k",
    }
    for alias, target_norm in alias_map.items():
        if target_norm in norm_map:
            norm_map.setdefault(alias, norm_map[target_norm])

    # 1) Direkt übergeben?
    if not interactive:
        if not user_value:
            co.print_error(_("passed_no_resolution"))
            return "fail", None
        raw = user_value.strip()
        given = raw.lower()

        # → erst über norm_map auf kanonischen Key mappen
        canonical_key = norm_map.get(given)

        if canonical_key is not None and canonical_key in res_defs:
            key = canonical_key
            if key == "original":
                return "original", None
            if key == "custom":
                if not interactive:
                    co.print_fail(_("scale_custom_requires_interactive"))
                    return "original", None
                while True:
                    candidate = ui.ask_two_ints(
                        _("enter_custom_resolution"),
                        default="1280:720",
                        explanation=_("enter_custom_resolution_explanation"),
                    )
                    ok, msg = vec.validate_custom_scale_leq(
                        None, None, candidate, require_even=True
                    )
                    if ok:
                        return "custom", candidate
                    co.print_fail(msg)
                    return "fail", None
            sc = res_defs[key].get("scale")
            return key, (sc if (isinstance(sc, str) and ":" in sc) else None)

        # freier String → W:H extrahieren
        w, h = _parse_two_ints_any_sep(given)
        if w and h:
            candidate = f"{w}:{h}"
            ok, msg = vec.validate_custom_scale_leq(
                None, None, candidate, require_even=True
            )
            if ok:
                return "custom", candidate
            co.print_fail(msg)
            return "fail", None

        co.print_fail(_("scale_invalid_resolution"))
        return "fail", None

    # 2) Interaktiv
    elif interactive:
        menu_keys = [
            k for k in keys_all if (k != "original" or not hide_original_in_menu)
        ]
        labels = [tr(res_defs[k]["name"]) for k in menu_keys]
        descs = [tr(res_defs[k].get("description", "")) for k in menu_keys]
        choice = ui.ask_user(
            _("choose_resolution"),
            options=menu_keys,
            descriptions=descs,
            display_labels=labels,
            default=4,
        )
        if choice is None:
            return "original", None
        if choice == "custom":
            while True:
                candidate = ui.ask_two_ints(
                    _("enter_custom_resolution"),
                    default="1280:720",
                    explanation=_("enter_custom_resolution_explanation"),
                )
                if candidate is None:
                    return "original", None
                ok, msg = vec.validate_custom_scale_leq(
                    None, None, candidate, require_even=True
                )
                if ok:
                    return "custom", candidate
                co.print_fail(msg)
        sc = res_defs[choice].get("scale")
        return choice, (sc if (isinstance(sc, str) and ":" in sc) else None)

    # 3) Nichts übergeben, kein UI → original
    return "original", None


# -----------------------
# Hauptfunktion
# -----------------------


def scale_video(args: Any) -> None:
    """
    Reines Scaling über die Plan-API:
      - Zielauflösung wird EINMAL vor der File-Schleife ermittelt.
      - Pro Datei: Quelle sondieren, Container/Codec/Encoder festlegen, Plan bauen.
      - Optional: Seitenverhältnis NICHT beibehalten → exakt gewünschte W:H (mit Encoder-Alignment).
      - Hochwertige Presets (Standard: 'cinema').
      - Audio/Subs Mapping via Plan-API.
      - Bei 'original' wird – falls nötig – ein Even-Align Fix angewendet.
    """

    BATCH_MODE, files = _prepare_inputs_typed(args)

    # preserve_ar strikt zu bool normalisieren (liest auch -ar/--ar)
    if not BATCH_MODE:
        resp = ui.ask_yes_no(
            _("scale_preserve_ar_prompt"), default=True, back_option=False
        )
        preserve_ar_bool: bool = True if resp is None else bool(resp)
    else:
        pa = _get_flag_bool(args, "preserve_ar", "ar", "keep_ar")
        preserve_ar_bool = True if pa is None else pa

    # Zielauflösung ermitteln (Key + "W:H") – EINMAL vor der Schleife
    res_key, res_scale = _resolve_resolution_key_and_scale(
        user_value=getattr(args, "resolution", None),
        interactive=(not BATCH_MODE),
        hide_original_in_menu=True,
    )

    if res_key == "fail":
        return
    eff_scale: Optional[str] = res_scale
    eff_w: Optional[int] = None
    eff_h: Optional[int] = None

    if res_scale:
        # parse W:H und auf Encoder-Alignment runden – aber *nicht* AR fixen
        try:
            w0, h0 = _parse_two_ints_any_sep(res_scale)
            if w0 and h0:
                # preferred_encoder brauchen wir für Alignment – kommt später; hier erstmal Platzhalter
                eff_w, eff_h = int(w0), int(h0)
            else:
                eff_w = eff_h = None
        except Exception:
            eff_w = eff_h = None

    for i, f in enumerate(files):
        in_path = Path(f)
        if not in_path.exists():
            co.print_warning(_("file_not_found").format(file=in_path.name))
            continue

        # ggf. vorhandenes Thumbnail konservieren
        preserved_cover: Path | None = None
        try:
            if vt.check_thumbnail(in_path, silent=True):
                preserved_cover = vt.extract_thumbnail(in_path)
        except Exception:
            preserved_cover = None

        # Quelle sondieren
        src_w, src_h, _src_pix = vec.ffprobe_geometry(in_path)
        src_fps = he.probe_src_fps(in_path)
        if not src_w or not src_h:
            co.print_fail(_("scale_probe_failed"))
            continue

        # Outputpfad & Container
        safe_scale_for_name = (res_scale or "").replace(":", "x")  # NTFS-safe
        suffix = f"_scaled_{safe_scale_for_name}" if res_scale else "_scaled"

        # Outputpfad & Container
        out_path = fs.build_output_path(
            input_path=in_path,
            output_arg=getattr(args, "output", None),
            default_suffix=suffix,
            idx=i,
            total=len(files),
        )
        target_container = (
            out_path.suffix.lstrip(".").lower()
            or vec.detect_container_from_path(in_path)
            or "mkv"
        )

        # Ziel-Codec: Quelle bevorzugen, sonst Vorschlag
        src_codec = vec.probe_video_codec(in_path)
        codec_key = vec.normalize_codec_key(src_codec) or "h264"
        if not vec.container_allows_codec(target_container, codec_key):
            codec_key = vec.suggest_codec_for_container(target_container)

        # Preferred Encoder (für Alignment/Plan hilfreich)
        enc_map = vec.select_encoder_map_for_format(
            target_container,
            defin.CONVERT_FORMAT_DESCRIPTIONS,
            ffmpeg_bin="ffmpeg",
            prefer_hw=True,
        )
        preferred_encoder = enc_map.get(codec_key)

        # --- Effektive Zielgrößen pro Datei bestimmen (wie der Plan es tun würde)
        eff_w, eff_h = vec.compute_effective_wh(
            src_w=src_w,
            src_h=src_h,
            req_scale=res_scale,  # vom Nutzer/Resolution-Key
            chosen_encoder=preferred_encoder,
            preserve_ar=preserve_ar_bool,
            allow_upscale=True,
        )
        eff_scale = f"{eff_w}:{eff_h}" if (eff_w and eff_h) else None

        # Nutzerinfo
        if res_scale and eff_scale and res_scale != eff_scale:
            co.print_info(_("scaling_to") + f" => {res_scale}  →  {eff_scale}")
        elif eff_scale:
            co.print_info(_("scaling_to") + f" => {eff_scale}")

        # --- Outputpfad & Container *auf Basis der effektiven Skala dieser Datei*
        safe_scale_for_name = (eff_scale or res_scale or "").replace(":", "x")
        suffix = (
            f"_scaled_{safe_scale_for_name}" if (eff_scale or res_scale) else "_scaled"
        )

        out_path = fs.build_output_path(
            input_path=in_path,
            output_arg=getattr(args, "output", None),
            default_suffix=suffix,
            idx=i,
            total=len(files),
        )
        target_container = (
            out_path.suffix.lstrip(".").lower()
            or vec.detect_container_from_path(in_path)
            or "mkv"
        )

        # --- Fall A: „original“ (oder kein res_scale) → Even-Align Fix, falls nötig
        if res_key == "original" or not res_scale:
            try:
                align = int(
                    getattr(vec, "_required_alignment")(preferred_encoder or "")
                )
            except Exception:
                align = 2
            need_fix = (src_w % align != 0) or (src_h % align != 0)
            if not need_fix:
                co.print_info(_("scale_skip_original").format(filename=in_path.name))
                try:
                    if preserved_cover and preserved_cover.exists():
                        preserved_cover.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            tgt_w = max(align, src_w - (src_w % align))
            tgt_h = max(align, src_h - (src_h % align))
            target_scale = f"{tgt_w}:{tgt_h}"

            co.print_info(
                _("even_align_fix")
                + f"{in_path.name}| {src_w}x{src_h} → {tgt_w}x{tgt_h} (align={align})"
            )

            plan = vec.build_transcode_plan(
                input_path=in_path,
                target_container=target_container,
                preset_name="lossless",
                codec_key=codec_key,
                preferred_encoder=preferred_encoder,
                req_scale=target_scale,
                src_w=src_w,
                src_h=src_h,
                src_fps=src_fps,
                user_fps_rational=None,
                preset_max_fps=defin.CONVERT_PRESET["cinema"].get("max_fps"),
                preserve_ar=True,  # hier egal – wir geben schon das „fixe“ Align-Ziel
            )
            final_cmd = vec.assemble_ffmpeg_cmd(plan, out_path)
            final_cmd = vec.apply_container_codec_quirks(
                final_cmd, target_container, codec_key
            )

            try:
                if hasattr(vec, "dedupe_vf_inplace"):
                    vec.dedupe_vf_inplace(final_cmd)
            except Exception:
                pass

            final_cmd = autotune_final_cmd(in_path, final_cmd)

            pw.run_ffmpeg_with_progress(
                input_file=in_path.name,
                ffmpeg_cmd=final_cmd,
                progress_line=_("scaling_progress").format(
                    width=str(tgt_w), height=str(tgt_h)
                ),
                finished_line=_("scaling_done").format(
                    width=str(tgt_w), height=str(tgt_h)
                ),
                output_file=out_path,
                BATCH_MODE=BATCH_MODE,
            )
            continue

        # --- Fall B: explizite Zielauflösung → Plan bauen
        target_scale = eff_scale or res_scale  # an build_transcode_plan weitergeben

        plan = vec.build_transcode_plan(
            input_path=in_path,
            target_container=target_container,
            preset_name="lossless",
            codec_key=codec_key,
            preferred_encoder=preferred_encoder,
            req_scale=target_scale,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            user_fps_rational=None,
            preset_max_fps=defin.CONVERT_PRESET["lossless"].get("max_fps"),
            preserve_ar=preserve_ar_bool,
        )

        final_cmd = vec.assemble_ffmpeg_cmd(plan, out_path)
        final_cmd = vec.apply_container_codec_quirks(
            final_cmd, target_container, codec_key
        )
        try:
            if hasattr(vec, "dedupe_vf_inplace"):
                vec.dedupe_vf_inplace(final_cmd)
        except Exception:
            pass

        # Exakte Zielmaße erzwingen, falls AR NICHT beibehalten werden soll
        if eff_w and eff_h:
            _force_exact_scale_and_square_pixels_inplace(
                final_cmd, eff_w, eff_h, force_sar1=True
            )

        # --- Progress-Anzeige anhand *finalem* Cmd (robust, auch mit trunc(iw/...))
        out_w, out_h = vec.infer_output_wh_from_cmd(final_cmd, src_w, src_h)
        w_str, h_str = str(out_w), str(out_h)

        final_cmd = autotune_final_cmd(in_path, final_cmd)

        pw.run_ffmpeg_with_progress(
            input_file=in_path.name,
            ffmpeg_cmd=final_cmd,
            progress_line=_("scaling_progress").format(width=w_str, height=h_str),
            finished_line=_("scaling_done").format(width=w_str, height=h_str),
            output_file=out_path,
            BATCH_MODE=BATCH_MODE,
        )

        # Thumbnail wieder einbetten (falls vorhanden)
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

    co.print_finished(_("scaling_method"))
