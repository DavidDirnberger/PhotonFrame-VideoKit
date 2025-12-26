#!/usr/bin/env python3
# ai_enhance.py
from __future__ import annotations

import math
import re
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# import process_wrappers as pw
import ai_backend as ab
import ai_helpers as ah
import ai_ncnn_backend as an
import ai_tta as at
import consoleOutput as co
import definitions as defin
import fileSystem as fs
import helpers as he
import image_helper as ih
import mem_guard as mg
import userInteraction as ui
import VideoEncodersCodecs as vcc

# local
from i18n import _, tr
from loghandler import _resolve_log_path, print_log

# import ai_pipeline as ah
# ah.set_config(force_backend="ncnn", disable_gpu=False)  # NCNN erzwingen, GPU aus

# ─────────────────────────────────────────────────────────────────────────────
# CLI-Args
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AIEnhanceArgs:
    files: List[str] = field(default_factory=list)
    aimodel: Optional[str] = None  # Modell-Key (Real-ESRGAN Name oder 'realcugan-*')
    denoise: Optional[float] = (
        None  # 0.0 … 1.0 (nur falls Modell "strength" unterstützt)
    )
    noise_level: Optional[int] = None
    face_enhance: Optional[bool] = None
    scale: Optional[float] = (
        None  # outscale (float) oder diskreter Faktor (int-kompatibel)
    )
    tta: Optional[bool] = None
    blend: Optional[bool] = None
    blend_opacity: Optional[float] = None
    priority: Optional[str] = None
    output: Optional[str] = None  # Zieldatei (nur sinnvoll bei EINER Eingabedatei)
    force_overwrite: Optional[bool] = False
    chunk: Optional[int] = None  # Frames/Chunk; default via defin oder dynamisch


def _ffmpeg_path_and_version() -> str:
    exe = shutil.which("ffmpeg") or "ffmpeg"
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        first = (out.stdout or "").splitlines()[:1]
        return f"{exe} | {first[0] if first else '(no version)'}"
    except Exception:
        return f"{exe} | (version unknown)"


# ─────────────────────────────────────────────────────────────────────────────
# Interne Helfer
# ─────────────────────────────────────────────────────────────────────────────


def _log_dir_for_concat() -> Path:
    try:
        return Path(_resolve_log_path()).resolve().parent
    except Exception:
        return Path("logs")


def _purge_old_concat_logs() -> None:
    try:
        log_dir = _log_dir_for_concat()
        if not log_dir.exists():
            return
        for p in log_dir.glob("*.concat.log"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _get_model_meta() -> Dict[str, Dict[str, Any]]:
    """
    Unterstützt defin.MODEL_META als Dict **oder** als Callable, das ein Dict zurückgibt.
    """
    try:
        obj = getattr(defin, "MODEL_META")
        return obj() if callable(obj) else obj  # type: ignore[misc]
    except Exception:
        return {}


_FRAME_RE = re.compile(r"frame_(\d+)")


def _frame_index_key(p: Path) -> Tuple[int, list]:
    m = re.search(r"frame_(\d+)", p.stem)
    # natürlicher Zusatzsortierschlüssel (Case-insensitive, Ziffern numerisch)
    s = p.name
    parts = re.split(r"(\d+)", s)
    nat_key = [int(x) if x.isdigit() else x.lower() for x in parts]
    return (int(m.group(1)) if m else 10**9, nat_key)


def _collect_png_files(up_dir: Path) -> List[Path]:
    items = list(up_dir.glob("*.png"))
    files = [p for p in items if p.exists() and p.is_file()]
    files.sort(key=_frame_index_key)
    return files


def _frames_are_contiguous(frames: List[Path]) -> bool:
    if not frames:
        return False
    for i, p in enumerate(frames, start=1):
        if p.name != f"frame_{i:06d}.png":
            return False
    return True


def _write_ffconcat(frames: List[Path], fps_float: float, tmp_root: Path) -> Path:
    """
    Erzeugt ein ffconcat-File mit exakter Dauer je Frame nach fps_float.
    """
    from tempfile import NamedTemporaryFile

    lists_dir = tmp_root / "ffconcat_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix="sr_", suffix=".ffconcat", dir=str(lists_dir), delete=False
    ) as f:
        list_path = Path(f.name)

    if not frames:
        raise RuntimeError(_("ai_no_valid_png_frames"))

    dur = 1.0 / max(1e-6, float(fps_float))
    with list_path.open("w", encoding="utf-8") as fh:
        fh.write("ffconcat version 1.0\n")
        for p in frames[:-1]:
            s = p.resolve().as_posix().replace("\\", "\\\\").replace("'", r"\'")
            fh.write(f"file '{s}'\n")
            fh.write(f"duration {dur:.6f}\n")
        sl = frames[-1].resolve().as_posix().replace("\\", "\\\\").replace("'", r"\'")
        fh.write(f"file '{sl}'\n")
    return list_path


def _desired_container_for(input_path: Path, output_arg: Optional[str]) -> str:
    """
    Wähle das gewünschte Zielformatsuffix **ohne** Hardcoding.
    1) aus --output (falls vorhanden),
    2) sonst aus Eingabecontainer,
    3) Fallback: 'mkv' (neutral, robust).
    """
    if output_arg:
        oc = vcc.detect_container_from_path(Path(output_arg))
        if oc:
            return oc
        suf = Path(output_arg).suffix.lstrip(".")
        if suf:
            return suf
    ic = vcc.detect_container_from_path(input_path)
    if ic:
        return ic
    suf = input_path.suffix.lstrip(".")
    return suf if suf else "mkv"


# ─────────────────────────────────────────────────────────────────────────────
# Encoding von PNG-Frames → Video-Only (mem_guard-gesteuert, format-agnostisch)
# ─────────────────────────────────────────────────────────────────────────────


def _encode_from_frames(
    frames: List[Path],
    audio_src: Path,
    out_path: Path,
    fps_label_str: str,
    fps_float: float,
    tmp_root: Path,
) -> bool:
    """
    Encodiert die PNG-Frames zu einem **Video-Only** Stream (CFR).
    - Container/Codec werden **aus out_path** abgeleitet (keine Hardcodes).
    - Alle Subprozesse laufen über mem_guard, inkl. Cancel/cleanup.
    """
    exe = shutil.which("ffmpeg") or "ffmpeg"
    logs_dir = tmp_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "ffmpeg_concat_encode.log"

    if not frames:
        co.print_warning(_("ai_no_upscaled_abort_encode"))
        return False

    # FPS ggf. aus frames_meta.json präziser wählen (label bleibt String)
    frames_dir = frames[0].parent
    fps_use_label = ab.fps_label_from_meta(frames_dir, fps_label_str)

    use_image2 = _frames_are_contiguous(frames)
    ffconcat: Optional[Path] = None

    # ───────── Debug: grundlegende Info ─────────
    try:
        print_log(
            f"[AI-ENCODE] up_dir={frames_dir} "
            f"n_frames={len(frames)} contiguous={use_image2} "
            f"fps_label={fps_use_label} fps_float={fps_float:.6f}"
        )
    except Exception:
        pass

    if not use_image2:
        try:
            ffconcat = _write_ffconcat(frames, float(fps_float), tmp_root)
            print_log(f"[AI-ENCODE] using ffconcat input: {ffconcat}")
        except Exception as e:
            co.print_warning(_("ai_ffconcat_write_failed").format(err=str(e)))
            return False

    # Container/Codec über Zielpfad & Quelle bestimmen – **ohne** Hardcodes
    container = (
        vcc.detect_container_from_path(out_path)
        or vcc.detect_container_from_path(audio_src)
        or out_path.suffix.lstrip(".")
        or audio_src.suffix.lstrip(".")
        or "mkv"
    )

    src_vcodec = (vcc.probe_video_codec(audio_src) or "").lower()
    codec_key = vcc.pick_crf_codec_for_container(container, src_vcodec)

    # Farbdaten/Geometrie der Quelle (für VF/Signalisierung)
    src_w, src_h, src_pf = vcc.ffprobe_geometry(audio_src)
    meta = vcc.probe_color_metadata(audio_src)

    try:
        print_log(
            f"[AI-ENCODE] container={container} "
            f"src_vcodec={src_vcodec} codec_key={codec_key} "
            f"src_geom={src_w}x{src_h} src_pix_fmt={src_pf}"
        )
        print_log(f"[AI-ENCODE] color_meta={meta}")
    except Exception:
        pass

    # Generische VF-Kette
    vf_chain, vf_extra = vcc.build_visual_chain_generic(
        meta=meta,
        src_w=src_w,
        src_h=src_h,
        container=container,
        codec_key=codec_key,
        src_pix_fmt=src_pf,
        scale_already_planned=False,
        prefer_zscale=True,
    )

    # Eingangs-Args (Sequenz vs. Concat) – **nur Videoeingang**
    if use_image2:
        pattern = frames[0].with_name("frame_%06d.png")
        in_args = [
            "-analyzeduration",
            "200M",
            "-probesize",
            "200M",
            "-framerate",
            str(fps_use_label),
            "-f",
            "image2",
            "-start_number",
            "1",
            "-i",
            str(pattern),
        ]
    else:
        in_args = [
            "-analyzeduration",
            "200M",
            "-probesize",
            "200M",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(ffconcat),
        ]

    # Basis-Videoausgabe (ohne Container-Spezialflags, die fügt vcc bei Bedarf ein)
    base_common = [
        "-map",
        "0:v:0",
        "-vsync",
        "cfr",
        "-r",
        str(fps_use_label),
        "-pix_fmt",
        vcc.playback_pix_fmt_for(container) or "yuv420p",
    ]

    base_cmd: List[str] = [exe, "-hide_banner", "-y", *in_args, *vf_extra, *base_common]
    try:
        print_log("[AI-ENCODE] base_cmd(no-out)=" + " ".join(map(str, base_cmd)))
    except Exception:
        pass

    # Encoder + Encoder-Flags auswählen (noch ohne Output)
    cmd_no_out = vcc.try_encode_with_fallbacks(
        base_cmd,
        codec_key=codec_key,
        container=container,
        preset_name="lossless",
        vf_chain=vf_chain,
        ffmpeg_bin=exe,
        preferred_encoder=None,
    )
    try:
        print_log(
            "[AI-ENCODE] after try_encode_with_fallbacks="
            + " ".join(map(str, cmd_no_out))
        )
    except Exception:
        pass

    # Farbraum-/Color-Tag-Signalisierung auf dem Command OHNE Output
    cmd_no_out = vcc.apply_color_signaling(
        cmd_no_out,
        container=container,
        codec_key=codec_key,
        meta=meta,
        src_pix_fmt=src_pf,
    )
    try:
        print_log(
            "[AI-ENCODE] after apply_color_signaling=" + " ".join(map(str, cmd_no_out))
        )
    except Exception:
        pass

    # Jetzt das tatsächliche Ausgabefile anhängen
    out_str = str(out_path)
    final_cmd: List[str] = cmd_no_out + [out_str]

    # WICHTIG:
    # Pixfmt-/format-Entscheidungen NUR auf dem finalen Command mit
    # echter Output-Datei treffen, damit _insert_before_output()
    # nicht zwischen Flag und dessen Wert landet (z.B. -x264-params).
    vcc.ensure_terminal_pix_fmt(
        final_cmd,
        container=container,
        codec_key=codec_key,
        input_path=audio_src,
    )
    try:
        print_log(
            "[AI-ENCODE] after ensure_terminal_pix_fmt=" + " ".join(map(str, final_cmd))
        )
    except Exception:
        pass

    # Reihenfolge vor dem Output bereinigen & movflags säubern
    vcc.ensure_pre_output_order(final_cmd)
    vcc.sanitize_movflags_inplace(final_cmd)

    # ───────── Hardening: sicherstellen, dass out_path WIRKLICH letzter Token ist ─────────
    if final_cmd[-1] != out_str:
        try:
            idx = final_cmd.index(out_str)
            trailing = final_cmd[idx + 1 :]
            print_log(
                f"[AI-ENCODE] output not last – moving to end; "
                f"trailing_after_output={trailing}"
            )
            # Alles hinter out_path vorziehen, out_path ganz ans Ende
            final_cmd = final_cmd[:idx] + final_cmd[idx + 1 :] + [out_str]
        except ValueError:
            print_log(
                "[AI-ENCODE] WARN: out_path not found in final_cmd – cannot enforce output-last"
            )

    # IDR-Insert (setzt seine Flags selbst vor dem Output)
    vcc.inject_idr_at_t0(final_cmd)

    try:
        print_log("[AI-ENCODE] final_cmd=" + " ".join(map(str, final_cmd)))
    except Exception:
        pass

    # Ausführen – **mem_guard-gesteuert**, Log in Datei
    ok = False
    try:
        proc = mg.popen(final_cmd, text=True, log_to=log_file)
        while True:
            try:
                _t = proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if mg.is_cancelled():
                    mg.terminate_process(proc, name="ffmpeg-encode", timeout=2.0)
                    break
        ok = (
            (proc.returncode or 0) == 0
            and out_path.exists()
            and out_path.stat().st_size > 0
        )
    except Exception as e:
        try:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n[EXC] {e}\n")
        except Exception:
            pass
        ok = False

    if not ok:
        try:
            tail = (
                log_file.read_text(encoding="utf-8") if log_file.exists() else ""
            ).splitlines()[-60:]
            if tail:
                co.print_warning(
                    _("ai_ffmpeg_concat_diag_tail") + "\n" + "\n".join(tail)
                )
        except Exception:
            pass

    return ok


def _concat_segments_with_mg(
    segments_file: Path, out_path: Path, src_name: str
) -> bool:
    """
    Führt mehrere Segmentdateien per concat-demuxer zusammen (Streamcopy).
    **mem_guard** verwaltet den Subprozess; keine Format-Hardcodes.
    """
    exe = shutil.which("ffmpeg") or "ffmpeg"
    cmdc = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(segments_file),
        "-c",
        "copy",
        str(out_path),
    ]
    log_dir = _log_dir_for_concat()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{out_path.name}.concat.log"
    try:
        co.print_info(_("ai_concat_progress"))
    except Exception:
        pass
    try:
        proc = mg.popen(cmdc, text=True, log_to=log_file)
        while True:
            try:
                _t = proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if mg.is_cancelled():
                    mg.terminate_process(proc, name="ffmpeg-concat", timeout=2.0)
                    break
        ok = (
            (proc.returncode or 0) == 0
            and out_path.exists()
            and out_path.stat().st_size > 0
        )
        if ok:
            co.print_info(_("ai_concat_done").format(video=out_path.name))
        else:
            try:
                tail = (
                    log_file.read_text(encoding="utf-8") if log_file.exists() else ""
                ).splitlines()[-60:]
                if tail:
                    co.print_warning("\n".join(tail))
            except Exception:
                pass
        return ok
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def ai_enhance(args: AIEnhanceArgs | Any) -> None:
    """
    Minimal-invasiv überarbeitet:
      - FPS/Framecount/FPS-Format **zentral** via helpers (he.*) & ai_helpers (ah.*)
      - **mem_guard** steuert lange Subprozesse (encode/concat), inkl. Cancel
      - Keine hartcodierten Container/Formate – Ziel wird aus Pfaden/Quelle abgeleitet
    """
    co.print_start(_("ai_enhance_method"))
    mg.install_global_cancel_handlers()  # ESC/Strg+C → globaler Cancel
    print_log(f"FFmpeg: {_ffmpeg_path_and_version()}")
    _purge_old_concat_logs()

    project_root = (
        getattr(defin, "PROJECT_ROOT", None)
        or Path(
            os.environ.get("VM_BASE", Path(__file__).resolve().parents[1])
        ).resolve()
    )
    esrgan_dir = (project_root / "real-esrgan").resolve()
    venv_python = Path(sys.executable)

    esr_script = esrgan_dir / "inference_realesrgan.py"
    if not esr_script.exists() and not an.which_ncnn() and not an.which_realcugan():
        co.print_warning(_("ai_backends_missing").format(path=str(esrgan_dir)))

    TMP_ROOT: Optional[Path] = None

    try:
        try:
            TMP_ROOT = fs.select_tmp_root(project_root)
        except RuntimeError as e:
            co.print_error(str(e))
            return

        co.print_info(_("ai_tmp_dir_using").format(dir=str(TMP_ROOT)))

        # ───────── Platz ─────────
        ok_space, free_gb = fs.has_free_space(TMP_ROOT)
        if not ok_space:
            co.print_error(_("ai_not_enough_space").format(need=5.0, free=free_gb))
            return

        # Inputs (zentral über helpers)
        BATCH_MODE, files = he.prepare_inputs(args)
        try:
            args.files = list(files)
        except Exception:
            pass
        if not files:
            co.print_error(_("no_video_files_found"))
            return

        while True:
            # ───────────────────────── Modellwahl ─────────────────────────
            opt, descs = _model_choices_with_status(esrgan_dir)
            if BATCH_MODE:
                ai_model = getattr(args, "aimodel", None)
                if not ai_model:
                    try:
                        def_idx = opt.index("realesr-general-x4v3")
                    except ValueError:
                        def_idx = 0
                    ai_model = opt[def_idx]
                    co.print_info(_("ai_model_default_used").format(model=ai_model))
                else:
                    if ai_model not in _get_model_meta():
                        co.print_error(_("ai_model_not_found").format(model=ai_model))
                        return
            else:
                sel = ui.ask_user(
                    _("ai_choose_model"),
                    opt,
                    descriptions=descs,
                    default=0,
                    display_labels=opt,
                    back_button=False,
                )
                if sel is None:
                    continue
                ai_model = sel

            # Meta + Caps + Backends
            meta_all = _get_model_meta()
            meta: Dict[str, Any] = dict(meta_all.get(ai_model, {}))
            caps: Dict[str, Any] = dict(meta.get("caps") or {})  # ← NIE None

            # verfügbare Backends (zur Laufzeit abschätzen)
            try:
                available_backends = ah.available_backends(ai_model)  # type: ignore[attr-defined]
                if not isinstance(available_backends, list):
                    available_backends = []
            except Exception:
                b = str(meta.get("backend", "")).lower()
                available_backends = [b] if b else ["pytorch", "ncnn"]
            pytorch_possible = "pytorch" in available_backends
            ncnn_possible = ("ncnn" in available_backends) or ai_model.startswith(
                "realcugan"
            )

            caps_runtime = ah.probe_system_caps(
                venv_python=venv_python, esr_script=esr_script
            )
            entry_backend = ah.pick_backend_for_model_runtime(ai_model, caps_runtime)
            print_log(
                f"[ENTRY] chosen start-backend for '{ai_model}': {entry_backend or 'auto'}"
            )

            # ============================ TTA ===============================
            try:
                py_has_tta, ncnn_has_tta, emu_tta = at.tta_capabilities(
                    venv_python, esr_script, an.which_ncnn()
                )
            except Exception:
                py_has_tta, ncnn_has_tta, emu_tta = (False, False, False)
            print_log(
                f"[CAPS] model={ai_model}; backends={available_backends}; py_tta={py_has_tta}; ncnn_tta={ncnn_has_tta}; emu_tta={emu_tta}"
            )

            # ───────────────────────── Face-Enhance ─────────────────────────
            fe_cfg = caps.get("face_enhance") or {"native": False}
            face_enhance_allowed = False
            if bool(fe_cfg.get("native", False)) and pytorch_possible:
                face_enhance_allowed = True
            elif pytorch_possible and bool(fe_cfg.get("via")):
                face_enhance_allowed = True

            if face_enhance_allowed:
                if BATCH_MODE:
                    fe_raw = getattr(args, "face_enhance", None)
                    face_enhance = he.str2bool(fe_raw)
                else:
                    face_enhance = ui.ask_yes_no(
                        _("ai_face_enhance_q"), default=False, back_option=True
                    )
                    if face_enhance is None:
                        continue
            else:
                face_enhance = False

            # ───────────────────────── Scale/Outscale ─────────────────────────
            outscale: float
            scale_section = caps.get("scale") or {}
            scale_cfg_py = scale_section.get("pytorch")
            scale_cfg_nc = scale_section.get("ncnn")

            use_py_scale = bool(pytorch_possible and isinstance(scale_cfg_py, dict))
            use_nc_scale = bool(
                (not use_py_scale) and ncnn_possible and isinstance(scale_cfg_nc, dict)
            )

            allowed_scales: List[int] = []
            arbitrary_outscale = False

            msd_raw = ab.infer_model_scale(ai_model)
            model_scale_default: int = int(msd_raw) if (msd_raw is not None) else 2

            a_scale = getattr(args, "scale", None)

            if use_py_scale and isinstance(scale_cfg_py, dict):
                mode = str(scale_cfg_py.get("mode", "")).lower()
                if mode == "arbitrary_outscale":
                    arbitrary_outscale = True
                    model_scale_default = int(
                        scale_cfg_py.get("model_scale", model_scale_default)
                    )
                    if (
                        BATCH_MODE
                        and a_scale is not None
                        and (a_scale < 1 or a_scale > model_scale_default)
                    ):
                        co.print_error(
                            tr(
                                {
                                    "de": f"Der Skalierungswert '{a_scale}' ist außerhalb des gültigen Bereiches [1,{model_scale_default}]!",
                                    "en": f"The scaling factor '{a_scale}' is outside the valid range [1,{model_scale_default}]!",
                                }
                            )
                        )
                        return
                elif mode == "discrete":
                    vals = scale_cfg_py.get("values") or []
                    allowed_scales = (
                        [int(x) for x in vals] if isinstance(vals, list) else []
                    )
                    if (
                        BATCH_MODE
                        and a_scale is not None
                        and (a_scale not in allowed_scales)
                    ):
                        co.print_error(
                            tr(
                                {
                                    "de": f"Der Skalierungswert '{a_scale}' ist nicht eines der gültigen diskreten Werte {allowed_scales}!",
                                    "en": f"The scaling factor '{a_scale}' is not one of the discreet values {allowed_scales}!",
                                }
                            )
                        )
                        return
                elif mode == "fixed":
                    fixed_scale = int(
                        scale_cfg_py.get("model_scale", model_scale_default)
                    )
                    if (
                        BATCH_MODE
                        and a_scale is not None
                        and (int(a_scale) != fixed_scale)
                    ):
                        co.print_error(
                            tr(
                                {
                                    "de": f"Der Skalierungswert '{a_scale}' wird nicht unterstützt es wird nur {fixed_scale}x unterstützt!",
                                    "en": f"The scaling factor '{a_scale}' is not supported, only {fixed_scale}x is possible!",
                                }
                            )
                        )
                        return
            elif use_nc_scale and isinstance(scale_cfg_nc, dict):
                mode = str(scale_cfg_nc.get("mode", "")).lower()
                if mode == "discrete":
                    vals = scale_cfg_nc.get("values") or []
                    allowed_scales = [
                        int(x) for x in (vals if isinstance(vals, list) else [])
                    ]
                    if (
                        BATCH_MODE
                        and a_scale is not None
                        and (a_scale not in allowed_scales)
                    ):
                        co.print_error(
                            tr(
                                {
                                    "de": f"Der Skalierungswert '{a_scale}' ist nicht eines der gültigen diskreten Werte {allowed_scales}!",
                                    "en": f"The scaling factor '{a_scale}' is not one of the discreet values {allowed_scales}!",
                                }
                            )
                        )
                        return

            if (not arbitrary_outscale) and (not allowed_scales):
                allowed_scales = [model_scale_default]

            if arbitrary_outscale:
                if BATCH_MODE:
                    outscale = (
                        float(a_scale)
                        if a_scale is not None
                        else float(model_scale_default)
                    )
                else:
                    try:
                        outscale = float(
                            ui.ask_float(
                                _("ai_outscale_prompt"),
                                default=float(model_scale_default),
                                min_value=1.0,
                                max_value=model_scale_default,
                            )
                        )
                    except Exception:
                        outscale = float(model_scale_default)
            elif allowed_scales:
                if not BATCH_MODE:
                    if len(allowed_scales) == 1:
                        outscale = allowed_scales[0]
                    else:
                        labels = [f"{s}×" for s in allowed_scales]
                        try:
                            default_idx = (
                                labels.index("2×")
                                if "2×" in labels
                                else len(labels) - 1
                            )
                        except ValueError:
                            default_idx = len(labels) - 1
                        sel = ui.ask_user(
                            tr(
                                {
                                    "de": "Wähle Upscale-Stufe",
                                    "en": "Choose upscale factor",
                                }
                            ),
                            labels,
                            default=default_idx,
                            display_labels=labels,
                        )
                        if sel is None:
                            continue
                        try:
                            outscale = float(
                                allowed_scales[labels.index(sel)]
                                if (sel in labels)
                                else allowed_scales[default_idx]
                            )
                        except Exception:
                            outscale = float(allowed_scales[default_idx])
                else:
                    outscale = (
                        float(a_scale)
                        if a_scale is not None
                        else float(2 if 2 in allowed_scales else allowed_scales[-1])
                    )
            else:
                outscale = float(model_scale_default)

            # ───────────────────────── Denoise (Caps-gesteuert) ─────────────────────────
            denoise: Optional[float] = None
            noise_level: Optional[int] = None
            dn_cfg = caps.get("denoise") or {"type": "none"}
            dn_type = str(dn_cfg.get("type", "none")).lower()
            arg_dn = getattr(args, "denoise", None)

            if dn_type == "strength":
                if BATCH_MODE:
                    arg_noise = getattr(args, "noise_level", None)
                    if arg_noise is not None:
                        co.print_warning(
                            tr(
                                {
                                    "de": f"Übergebener noise_level Parameter '{arg_noise}' is für das Model {ai_model} nicht definiert und wird ignoriert!",
                                    "en": f"The passed noise_level parameter '{arg_noise}' is not defined for model {ai_model} and will be ignored!",
                                }
                            )
                        )
                    denoise = float(arg_dn) if arg_dn is not None else 0.4

                    # Kontinuierliche Range-Validierung (0..1 oder Caps-min/max)
                    try:
                        lo = float(dn_cfg.get("min", 0.0))
                        hi = float(dn_cfg.get("max", 1.0))
                    except Exception:
                        lo, hi = 0.0, 1.0

                    # Falls jemand Werte in Caps hinterlegt hat, interpretiere sie als Range-Grenzen (nicht diskret)
                    vals = dn_cfg.get("values")
                    if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                        try:
                            vmin = min(float(x) for x in vals)
                            vmax = max(float(x) for x in vals)
                            lo, hi = vmin, vmax
                        except Exception:
                            pass

                    if denoise is None or not (lo <= float(denoise) <= hi):
                        co.print_error(
                            tr(
                                {
                                    "de": f"Ungültiger Denoise-Strength {denoise:g} für Modell '{ai_model}'. Erlaubter Bereich: {lo:g}..{hi:g}.",
                                    "en": f"Invalid denoise strength {denoise:g} for model '{ai_model}'. Allowed range: {lo:g}..{hi:g}.",
                                }
                            )
                        )
                        try:
                            shutil.rmtree(TMP_ROOT, ignore_errors=True)
                        except Exception:
                            pass
                        co.print_finished(_("ai_enhance_method"))
                        return
                else:
                    pct = (
                        ui.read_percent(_("ai_denoise_percent_prompt"), default=50)
                        or 50
                    )
                    try:
                        denoise = max(0.0, min(1.0, float(pct) / 100.0))
                    except Exception:
                        denoise = 0.4

            elif dn_type == "levels":
                # raw_levels = dn_cfg.get("values")
                raw_scale_levels = caps.get("noise_levels_by_scale") or {}
                raw_levels = raw_scale_levels.get(str(int(outscale)))
                levels: List[int] = [
                    int(x) for x in (raw_levels if isinstance(raw_levels, list) else [])
                ]
                if not levels:
                    raw = caps.get("noise_levels")
                    levels = [int(x) for x in (raw if isinstance(raw, list) else [])]
                if levels:
                    labels = [str(x) for x in levels]
                    if BATCH_MODE:
                        if arg_dn is not None:
                            co.print_warning(
                                tr(
                                    {
                                        "de": f"Übergebener denoise Parameter '{arg_dn}' is für das Model {ai_model} nicht definiert und wird ignoriert!",
                                        "en": f"The passed denoise parameter '{arg_dn}' is not defined for model {ai_model} and will be ignored!",
                                    }
                                )
                            )
                        noise_level = getattr(args, "noise_level", levels[0]) or (
                            0 if 0 in levels else levels[0]
                        )
                        # ✅ Batch-Validierung: nur erlaubte Level zulässig
                        try:
                            nl = int(noise_level) if noise_level is not None else None
                        except Exception:
                            nl = None
                        if nl is None or nl not in set(levels):
                            co.print_error(
                                tr(
                                    {
                                        "de": f"Ungültiger Noise-Level {noise_level} für Modell '{ai_model}'. Erlaubt: {levels}",
                                        "en": f"Invalid noise level {noise_level} for model '{ai_model}'. Allowed: {levels}",
                                    }
                                )
                            )
                            try:
                                shutil.rmtree(TMP_ROOT, ignore_errors=True)
                            except Exception:
                                pass
                            co.print_finished(_("ai_enhance_method"))
                            return
                    else:
                        if len(levels) == 1:
                            sel = 0
                        else:
                            try:
                                default_idx = (
                                    labels.index("-1") if "-1" in labels else 0
                                )
                            except ValueError:
                                default_idx = 0
                            sel = ui.ask_user(
                                tr(
                                    {
                                        "de": "Wähle Noise-Level",
                                        "en": "Choose noise level",
                                    }
                                ),
                                labels,
                                default=default_idx,
                                display_labels=labels,
                                explanation=tr(
                                    {
                                        "de": "Noise-Level: -1=aus, 0=leicht, 1–3 stärker (modellabhängig).",
                                        "en": "Noise level: -1=off, 0=light, 1–3 stronger (model-dependent).",
                                    }
                                ),
                            )
                            if sel is None:
                                continue
                        noise_level = int(sel)
            else:
                denoise = None
                noise_level = None

            # ───────────────────────── TTA ─────────────────────────
            tta_caps_nc = bool(
                isinstance(scale_cfg_nc, dict) and bool(scale_cfg_nc.get("tta"))
            )
            tta_possible = (ncnn_possible and tta_caps_nc and ncnn_has_tta) or (
                pytorch_possible and (py_has_tta or emu_tta)
            )

            if BATCH_MODE:
                a_tta = getattr(args, "tta", None)
                tta = bool(a_tta) and tta_possible
            else:
                if tta_possible:
                    tta = ui.ask_yes_no(
                        _("ai_tta_question"),
                        default=False,
                        explanation=tr(
                            {
                                "de": "TTA (Test-Time Augmentation) kann die Qualität leicht verbessern, ist aber deutlich langsamer.",
                                "en": "TTA (test-time augmentation) can slightly improve quality but is much slower.",
                            }
                        ),
                    )
                    if tta is None:
                        continue
                else:
                    tta = False

            # ───────────────────────── Blending ─────────────────────────
            if BATCH_MODE:
                do_blend = getattr(args, "blend", False)
                blend_opacity = 0.85
            else:
                do_blend = ui.ask_yes_no(
                    _("ai_blend_question"),
                    default=False,
                    explanation=tr(
                        {
                            "de": "Ein Blend-Modus definiert, wie Ebenenpixel mathematisch mit darunterliegenden kombiniert werden.",
                            "en": "A blend mode defines how a layer’s pixels combine mathematically with those beneath.",
                        }
                    ),
                )
                if do_blend is None:
                    continue
                blend_opacity = (
                    float(
                        ui.ask_float(
                            _("ai_blend_opacity_prompt"),
                            default=0.85,
                            min_value=0.0,
                            max_value=1.0,
                        )
                    )
                    if do_blend
                    else 1.0
                )

            # ───────────────────────── Worker-Override (nur interaktiv) ─────────────────────────
            user_worker_profile_choice = (
                "auto",
                "max",
                "medium",
                "minimal",
                "no_parallelisation",
            )
            user_worker_profile = "auto"
            if BATCH_MODE:
                user_worker_profile = getattr(args, "priority", "auto") or "auto"
                if user_worker_profile is None:
                    user_worker_profile = "auto"
                    co.print_info(_("ai_worker_default_choose"))
                else:
                    if (
                        ab.normalize_worker_profile_name(user_worker_profile)
                        not in user_worker_profile_choice
                    ):
                        co.print_error(_("ai_worker_setting_not_found"))
            else:
                wlabels = [
                    tr({"de": "Auto (empfohlen)", "en": "Auto (recommended)"}),
                    tr(
                        {
                            "de": "Maximale Auslastung/Priorität",
                            "en": "Maximum throughput / priority",
                        }
                    ),
                    tr({"de": "Medium", "en": "Medium"}),
                    tr({"de": "Minimal (1 Worker)", "en": "Minimal (1 worker)"}),
                    tr(
                        {
                            "de": "Serial - Keine Parallelisierung",
                            "en": "Serial - No parallelisation",
                        }
                    ),
                ]
                sel = ui.ask_user(
                    tr({"de": "Worker-Modus wählen", "en": "Choose worker mode"}),
                    user_worker_profile_choice,
                    default=0,
                    display_labels=wlabels,
                    explanation=tr(
                        {
                            "de": "Legt die Parallelität der per-Frame-Rechnung fest.",
                            "en": "Controls per-frame parallelism during upscaling.",
                        }
                    ),
                )
                if sel is None:
                    continue
                user_worker_profile = sel

            # Nach der Auswahl:
            norm_profile = ab.normalize_worker_profile_name(
                user_worker_profile or "auto"
            )
            # Alias vereinheitlichen
            if norm_profile is None or norm_profile in ("no_parallelisation",):
                norm_profile = "serial"
            user_worker_profile = norm_profile

            # Torch-Umgebung nur, wenn PyTorch irgendeine Rolle spielt
            try:
                if pytorch_possible:
                    he.ensure_torch_packages(venv_python)
            except RuntimeError as err:
                co.print_error(str(err))
                return

            break

        # ───────── Preflight/Chunking ─────────
        # Gesamt-Frame-Schätzung **zentral** via helpers
        total_frames_all = he.calculate_total_frames(files)
        w, h = ab.detect_resolution_via_helper(Path(files[0]), venv_python)
        chunk_default = int(getattr(defin, "AI_CHUNK_DEFAULT", 500) or 500)
        user_chunk = getattr(args, "chunk", None)
        # CHUNK dynamisch – abhängig von freiem Speicher & Auflösung
        tta_disk_heavy = bool(tta and emu_tta)
        CHUNK = (
            int(user_chunk)
            if user_chunk
            else ab.compute_dynamic_chunk(
                free_gb,
                w,
                h,
                int(round(outscale)),
                max(1, total_frames_all),
                chunk_default,
                tta=tta_disk_heavy,
            )
        )
        chunks_total = int(math.ceil(max(1, total_frames_all) / max(1, CHUNK)))
        final_out = None

        # Anzeige (nur relevante Keys)
        show_params: Dict[str, Any] = {
            "files": files,
            "aimodel": ai_model,
            "face_enhance": _("yes") if face_enhance else _("no"),
            "outscale": outscale,
            "tta": (
                _("yes")
                if tta
                else tr({"de": "nein / nicht verfügbar", "en": "no / not available"})
            ),
            "blend": (f"{blend_opacity:.2f}" if do_blend else _("no")),
            "workers_mode": {
                "auto": "Auto",
                "max": "Max",
                "medium": "Medium",
                "minimal": "Minimal",
                "no_parallelisation": "Serial",
                "serial": "Serial",
            }[user_worker_profile],
        }
        if dn_type == "strength":
            show_params["denoise"] = denoise
        if dn_type == "levels":
            show_params["noise_level"] = noise_level

        labels = {
            "aimodel": {"de": "KI-Modell", "en": "AI model"},
            "denoise": {"de": "Denoise (0–1)", "en": "Denoise (0–1)"},
            "face_enhance": {"de": "Gesichts-Verbesserung", "en": "Face enhancement"},
            "outscale": {"de": "Outscale", "en": "Outscale"},
            "noise_level": {"de": "Noise-Level", "en": "Noise level"},
            "tta": {"de": "TTA", "en": "TTA"},
            "blend": {"de": "Blending", "en": "Blending"},
            "workers_mode": {"de": "Worker-Modus", "en": "Worker mode"},
        }
        relevant = {"general": set(k for k in show_params.keys() if k != "files")}
        co.print_selected_params_table(
            show_params, labels=labels, relevant_groups=relevant, show_files=True
        )

        # Preflight (Best-Effort)
        try:
            preflight_stub_dir = TMP_ROOT / "__preflight__"
            preflight_stub_dir.mkdir(parents=True, exist_ok=True)
            pre = ah.compute_preflight_info(
                venv_python=venv_python,
                model=ai_model,
                raw_dir=preflight_stub_dir,
                total_frames=min(CHUNK, max(1, total_frames_all)),
                face_enhance=bool(face_enhance),
                user_profile=user_worker_profile,  # ← NEU
                tta=bool(tta),
                backend=entry_backend or "pytorch",
            )
            sel, lab, groups = ah.preflight_table_payload(pre)
            sel["chunks_total"] = chunks_total
            lab["chunks_total"] = {"de": "Chunks gesamt", "en": "Chunks total"}
            groups.setdefault("plan", set()).add("chunks_total")
            co.print_selected_params_table(
                sel,
                labels=lab,
                relevant_groups=groups,
                show_files=False,
                title=_("process_parameters"),
            )
        except Exception:
            pass

        print_log(
            f"[PARAMS] model={ai_model}; outscale={outscale}; dn_type={dn_type}; denoise={denoise}; noise_level={noise_level}; face_enhance={face_enhance}; tta={tta}; backends={available_backends}"
        )

        print("")

        try:
            for i, file in enumerate(files):
                if mg.is_cancelled():
                    break
                final_out = None
                path = Path(file)
                if not path.exists():
                    co.print_fail(_("file_not_found").format(file=str(path)))
                    continue

                # Zielcontainer **vorab** bestimmen (für Segment-Suffix)
                desired_container = _desired_container_for(
                    path, getattr(args, "output", None)
                )
                seg_suffix = f".{desired_container}"

                # FPS/Framecount **zentral über helpers**
                fps_f = he.probe_src_fps(path) or 25.0
                fps_label = he.format_fps(
                    fps_f, decimals=3, strip_trailing_zeros=True, fallback="25"
                )
                total_frames = he.detect_frame_count(
                    path, venv_python, fallback_fps=str(fps_f)
                )
                if total_frames <= 0:
                    co.print_fail(_("ai_framecount_failed"))
                    continue

                ok_space, free_gb = fs.has_free_space(TMP_ROOT)
                if not ok_space:
                    co.print_error(
                        _("ai_not_enough_space").format(need=5.0, free=free_gb)
                    )
                    continue

                segments: List[Path] = []
                idx = 0
                start = 0
                while start < total_frames:
                    if mg.is_cancelled():
                        break
                    ok, free_gb_loop = fs.has_free_space(TMP_ROOT)
                    if not ok:
                        co.print_error(
                            _("ai_not_enough_space").format(need=5.0, free=free_gb_loop)
                        )
                        break

                    end_frame = min(start + CHUNK - 1, total_frames - 1)
                    print_log(
                        f"[CHUNK] idx={idx} start={start} end={end_frame} total={total_frames} "
                        f"chunk={CHUNK} free_gb={free_gb_loop:.2f} tta={tta} "
                        f"raw_dir={TMP_ROOT / f'raw_{idx:03d}'} up_dir={TMP_ROOT / f'up_{idx:03d}'}"
                    )

                    raw_dir = TMP_ROOT / f"raw_{idx:03d}"
                    up_dir = TMP_ROOT / f"up_{idx:03d}"
                    seg_out = path.parent / f"{path.stem}_seg{idx:03d}{seg_suffix}"

                    shutil.rmtree(raw_dir, ignore_errors=True)
                    shutil.rmtree(up_dir, ignore_errors=True)
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    up_dir.mkdir(parents=True, exist_ok=True)

                    # 1) Frames extrahieren (ai_helpers kapselt die Subprozesse)
                    if not ih.extract_frames_to_bmp(
                        video_path=path,
                        raw_dir=raw_dir,
                        start_frame=start,
                        end_frame=end_frame,
                        idx=idx,
                        input_name=path.name,
                        BATCH_MODE=BATCH_MODE,
                        total_chunks=chunks_total,
                    ):
                        idx += 1
                        start = end_frame + 1
                        continue
                    if mg.is_cancelled():
                        break

                    # 2) Upscaling
                    ok_esr = ah.run_ai_for_chunk(
                        venv_python=venv_python,
                        esr_script=esr_script,
                        esrgan_root=esrgan_dir,
                        model=ai_model,
                        raw_dir=raw_dir,
                        up_dir=up_dir,
                        face_enhance=bool(face_enhance),
                        denoise=(
                            float(denoise)
                            if (dn_type == "strength" and denoise is not None)
                            else None
                        ),
                        chunk_idx=idx,
                        chunks_total=chunks_total,
                        outscale=float(outscale),
                        tta=bool(tta),
                        realcugan_noise=(
                            int(noise_level)
                            if (dn_type == "levels" and noise_level is not None)
                            else None
                        ),
                        user_worker_profile=user_worker_profile,
                        entry_backend=entry_backend,
                    )
                    if mg.is_cancelled():
                        break
                    if not ok_esr:
                        co.print_error(_("ai_chunk_no_upscaled").format(idx=idx))
                        shutil.rmtree(raw_dir, ignore_errors=True)
                        idx += 1
                        start = end_frame + 1
                        continue

                    # 2b) PNGs sammeln
                    seq = _collect_png_files(up_dir)
                    if not seq:
                        co.print_error(_("ai_chunk_no_upscaled").format(idx=idx))
                        shutil.rmtree(raw_dir, ignore_errors=True)
                        idx += 1
                        start = end_frame + 1
                        continue

                    print_log(
                        _("ai_sequence_stats").format(
                            n=len(seq), first=seq[0].name, last=seq[-1].name
                        )
                    )

                    # 3) Encode via CFR (fps_label unverändert übernehmen)
                    try:
                        if seg_out.exists():
                            seg_out.unlink()
                    except Exception:
                        pass

                    ok_encode = _encode_from_frames(
                        seq, path, seg_out, str(fps_label), float(fps_f), TMP_ROOT
                    )

                    if (
                        not ok_encode
                        or not seg_out.exists()
                        or seg_out.stat().st_size == 0
                    ):
                        co.print_error(_("ai_chunk_encode_failed").format(idx=idx))
                        co.print_warning(
                            tr(
                                {
                                    "de": f"Segment {idx:03d} fehlgeschlagen – Rohframes bleiben erhalten: {up_dir.name}",
                                    "en": f"Segment {idx:03d} failed — keeping raw frames: {up_dir.name}",
                                }
                            )
                        )
                        shutil.rmtree(raw_dir, ignore_errors=True)  # nur RAW löschen
                    else:
                        shutil.rmtree(raw_dir, ignore_errors=True)
                        shutil.rmtree(up_dir, ignore_errors=True)
                        segments.append(seg_out)

                    idx += 1
                    start = end_frame + 1

                if mg.is_cancelled():
                    co.print_warning(_("ai_cancel_cleanup"))
                    for seg in segments:
                        try:
                            seg.unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
                    break

                # 4) Zusammenfügen der Segmente (Video-only) + anschließendes Remux der Original-Streams
                if segments and not mg.is_cancelled():
                    concat_file = TMP_ROOT / "concat.txt"
                    concat_file.write_text(
                        "".join(f"file '{s.resolve()}'\n" for s in segments),
                        encoding="utf-8",
                    )

                    # final_out **format-agnostisch** aufbauen (behält/folgt gewünschtem Container)
                    final_out = fs.build_output_path(
                        input_path=path,
                        output_arg=getattr(args, "output", None),
                        default_suffix="_upscaled",
                        idx=i,
                        total=len(files),
                    )

                    # Video-only Ziel (wird später remuxt)
                    concat_out = final_out.with_name(
                        final_out.stem + ".__video_only__" + final_out.suffix
                    )
                    try:
                        if concat_out.exists():
                            concat_out.unlink()
                    except Exception:
                        pass

                    if len(segments) == 1:
                        # single segment → einfach umbenennen = video-only
                        try:
                            if concat_out.exists():
                                try:
                                    concat_out.unlink()
                                except Exception:
                                    pass
                            shutil.move(str(segments[0]), str(concat_out))
                        except Exception:
                            # Fallback: Streamcopy via ffmpeg – **mem_guard.run**
                            cmdc = [
                                "ffmpeg",
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                "-y",
                                "-i",
                                str(segments[0]),
                                "-c",
                                "copy",
                                str(concat_out),
                            ]
                            mg.run(
                                cmdc,
                                check=False,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                    else:
                        # mem_guard-gesteuerte Concat (statt pw.run_ffmpeg_with_progress)
                        _t = _concat_segments_with_mg(
                            concat_file, concat_out, path.name
                        )

                    # ── NEU: Remux der Original-Streams (Audio, Subs, Attachments, Chapters)
                    try:
                        src_info = ab.probe_source_media_info(path)
                    except Exception:
                        src_info = {
                            "container": vcc.detect_container_from_path(final_out)
                            or final_out.suffix.lstrip(".")
                        }

                    remux_ok = ab.remux_with_source_streams(
                        video_only=concat_out,
                        source_media=path,
                        out_path=final_out,
                        container=(
                            src_info.get("container")
                            or vcc.detect_container_from_path(final_out)
                        ),
                        tmp_root=TMP_ROOT,
                    )

                    if not remux_ok:
                        co.print_warning(
                            tr(
                                {
                                    "de": "Remux der Original-Streams fehlgeschlagen – liefere Video ohne Zusatzspuren aus.",
                                    "en": "Remux of original streams failed — delivering video-only output.",
                                }
                            )
                        )
                        # Fallback: das video-only als Ergebnis verwenden
                        try:
                            if final_out.exists():
                                final_out.unlink()
                            shutil.move(str(concat_out), str(final_out))
                        except Exception:
                            pass
                    else:
                        # Remux ok → video-only aufräumen
                        try:
                            concat_out.unlink()
                        except Exception:
                            pass

                    # Optional: Blending **nach** dem Remux (damit Audio/Subs sicher vorhanden sind)
                    if do_blend and not mg.is_cancelled():
                        blended = final_out.with_stem(final_out.stem + "_blended")
                        ok_blend = ab.blend_with_original(
                            final_out, path, blend_opacity, blended
                        )
                        if ok_blend:
                            try:
                                final_out.unlink(missing_ok=True)  # type: ignore[arg-type]
                            except Exception:
                                pass
                            final_out = blended
                            co.print_info(
                                _("ai_blend_created").format(name=final_out.name)
                            )
                        else:
                            co.print_warning(_("ai_blend_failed_fallback"))

                    # Segment-Dateien aufräumen
                    for seg in segments:
                        try:
                            seg.unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception as e:
                            co.print_warning(
                                _("ai_segment_delete_fail").format(
                                    seg=str(seg), err=str(e)
                                )
                            )

        finally:
            try:
                shutil.rmtree(TMP_ROOT, ignore_errors=True)
                co.print_info(_("ai_tmp_cleanup_ok").format(dir=str(TMP_ROOT)))
            except Exception:
                co.print_warning(_("ai_tmp_cleanup_warn").format(dir=str(TMP_ROOT)))

        if final_out is not None and final_out.exists():
            co.print_info(_("ai_concat_done").format(video=str(final_out.name)))
    except KeyboardInterrupt:
        # Freundlicher Abbruch ohne Traceback
        try:
            mg.CANCEL.set()
        except Exception:
            pass
        try:
            mg.kill_all()
        except Exception:
            pass
        # co.print_info("Abgebrochen (ESC/Strg+C).")
        # Best-effort Cleanup, falls noch nicht erfolgt
        if TMP_ROOT:
            try:
                shutil.rmtree(TMP_ROOT, ignore_errors=True)
                co.print_info(_("ai_tmp_cleanup_ok").format(dir=str(TMP_ROOT)))
            except Exception:
                co.print_warning(_("ai_tmp_cleanup_warn").format(dir=str(TMP_ROOT)))
        return
    finally:
        co.print_finished(_("ai_enhance_method"))


# ─────────────────────────────────────────────────────────────────────────────
# Modelle/Backends-Auswahl
# ─────────────────────────────────────────────────────────────────────────────


def _model_choices_with_status(
    esrgan_root: Path,
    *,
    venv_python: Optional[Path] = None,
    esr_script: Optional[Path] = None,
    prefer_vulkan: Optional[bool] = None,
) -> Tuple[List[str], List[str]]:
    """
    Liefert (options, descs) anhand der **laufzeit**-Fähigkeiten:
      - filtert Modelle ohne verfügbares Backend (z. B. CUGAN ohne Binary)
      - wählt einen sinnvollen Start-Backend (Torch(CUDA/MPS/CPU) vs. NCNN)
      - erzeugt eine Statuszeile pro Modell („… — [PyTorch • CUDA]“)
    """
    caps = ah.probe_system_caps(venv_python=venv_python, esr_script=esr_script)
    avail = ah.compute_available_models_for_caps(caps)

    options: List[str] = []
    descs: List[str] = []
    meta_all = _get_model_meta()

    for key in avail:
        # Modell kennt keine Backends → raus
        if key not in meta_all:
            continue
        backend = ah.pick_backend_for_model_runtime(
            key, caps, prefer_vulkan=prefer_vulkan
        )
        if backend is None:
            # zur Sicherheit nochmal filtern
            continue
        desc = ah.format_model_status_line(key, backend, caps)
        options.append(key)
        descs.append(desc)

    # Sanitätscheck – lieber eine klare Meldung als leere Liste
    if not options:
        co.print_warning(
            tr(
                {
                    "de": "Keine lauffähigen Modelle gefunden. Prüfe bitte RealESRGAN/RealCUGAN-Installation und Pfade.",
                    "en": "No runnable models found. Please check your RealESRGAN/RealCUGAN installation and paths.",
                }
            )
        )

    return options, descs
