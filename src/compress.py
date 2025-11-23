#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

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
from i18n import _


# --- getypte Default-Fabrik für List[str] ---
def _list_str_factory() -> List[str]:
    return []


@dataclass
class CompressArgs:
    files: List[str] = field(default_factory=_list_str_factory)  # <- statt list
    quality: Optional[int] = 40
    output: Optional[str] = None


# ----------------------- kleine, lokale Helfer -----------------------


def _ffprobe_json(
    path: Path, select: str = "format", entries: str = ""
) -> Dict[str, Any]:
    """
    select: 'format' oder 'stream'
    entries: ffprobe -show_entries Wert(e)
    """
    cmd: List[str] = ["ffprobe", "-v", "error"]
    if select == "format":
        if not entries:
            entries = "format=duration,size"
        cmd += ["-show_entries", entries, "-of", "json", str(path)]
    else:
        if not entries:
            entries = (
                "stream=index,codec_type,codec_name,channels,bit_rate,avg_frame_rate"
            )
        cmd += ["-show_entries", entries, "-of", "json", str(path)]
    try:
        out = subprocess.check_output(cmd, text=True)
        return cast(Dict[str, Any], json.loads(out))
    except Exception:
        return {}


def _probe_audio_info(path: Path) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """=> (codec_name:str|None, channels:int|None, bitrate_bps:int|None)"""
    j: Dict[str, Any] = _ffprobe_json(
        path,
        select="stream",
        entries="stream=index,codec_type,codec_name,channels,bit_rate",
    )
    streams: List[Dict[str, Any]] = cast(List[Dict[str, Any]], j.get("streams") or [])
    for st in streams:
        if (cast(str, st.get("codec_type") or "")).lower() == "audio":
            br_val: Optional[str] = cast(Optional[str], st.get("bit_rate"))
            try:
                br: Optional[int] = int(br_val) if br_val is not None else None
            except Exception:
                br = None
            ch_val: Optional[int] | Optional[str] = cast(
                Optional[str], st.get("channels")
            )
            try:
                ch: Optional[int] = int(ch_val) if ch_val is not None else None
            except Exception:
                ch = None
            return (cast(Optional[str], st.get("codec_name")), ch, br)
    return (None, None, None)


def _probe_video_codec(path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return out or None
    except Exception:
        return None


def _probe_duration_sec(path: Path) -> Optional[float]:
    j: Dict[str, Any] = _ffprobe_json(path, select="format", entries="format=duration")
    try:
        fmt: Dict[str, Any] = cast(Dict[str, Any], j.get("format") or {})
        d = float(fmt.get("duration") or 0.0)
        return d if d > 0 else None
    except Exception:
        return None


def _probe_fps(path: Path) -> Optional[float]:
    j: Dict[str, Any] = _ffprobe_json(
        path, select="stream", entries="stream=index,codec_type,avg_frame_rate"
    )
    streams: List[Dict[str, Any]] = cast(List[Dict[str, Any]], j.get("streams") or [])
    for st in streams:
        if (cast(str, st.get("codec_type") or "")).lower() == "video":
            afr: Optional[str] = cast(Optional[str], (st.get("avg_frame_rate") or ""))
            if afr and afr != "0/0":
                try:
                    if "/" in afr:
                        a_str, b_str = afr.split("/")
                        a_f = float(a_str)
                        b_f = float(b_str)
                        return a_f / b_f if b_f != 0 else None
                    return float(afr)
                except Exception:
                    pass
    return None


def _as_str(x: object, default: str = "libx264") -> str:
    if x is None:
        return default
    if isinstance(x, (list, tuple, set)):
        try:
            first = next(iter(cast(Sequence[object], x)))
            x = first
        except StopIteration:
            return default
    if isinstance(x, dict):
        x = (
            x.get("encoder")
            or x.get("codec")
            or x.get("key")
            or x.get("name")
            or default
        )  # type: ignore[assignment]
    if not isinstance(x, str):
        x = str(x)
    return x


def _encoder_for_codec(codec_name: str) -> str:
    cn = _as_str((codec_name or "").lower())
    try:
        enc = defin.CODEC_FALLBACK_POLICY.get(cn)  # type: ignore[attr-defined]
        if enc:
            return _as_str(enc)
    except Exception:
        pass
    mapping: Dict[str, str] = {
        "h264": "libx264",
        "avc1": "libx264",
        "h265": "libx265",
        "hev1": "libx265",
        "hevc": "libx265",
        "vp9": "libvpx-vp9",
        "vp8": "libvpx",
        "av1": "libaom-av1",
        "mpeg4": "mpeg4",
        "mpeg2video": "mpeg2video",
        "mpeg1video": "mpeg1video",
        "mjpeg": "mjpeg",
        "prores": "prores_ks",
        "dnxhd": "dnxhd",
        "ffv1": "ffv1",
        "huffyuv": "huffyuv",
        "utvideo": "utvideo",
        "rawvideo": "rawvideo",
        "magicyuv": "magicyuv",
        "jpeg2000": "jpeg2000",
        "png": "png",
        "qtrle": "qtrle",
    }
    return mapping.get(cn, "libx264")


# ---- Audio-Policy gemäß „Kompressions-%“-Schwellen ----
def _choose_audio_flags_for_policy(
    src_audio_codec: Optional[str], src_audio_bps: Optional[int], quality_percent: int
) -> List[str]:
    """
    Schwellen:
    - ≥ 25 %: copy wenn (AAC/Opus/MP3) und <= 192k, sonst 192k
    - 5–24 %: copy wenn (AAC/Opus/MP3) und <= 128k, sonst 128k
    - <  5 %: copy wenn (AAC/Opus/MP3) und <= 96k,  sonst 96k
    """
    lc = (src_audio_codec or "").lower()
    is_copyable = lc in {"aac", "mp3", "opus"}
    br_k = 0 if (src_audio_bps is None) else int(round(src_audio_bps / 1000.0))

    if quality_percent >= 25:
        if is_copyable and br_k <= 192:
            return ["-c:a", "copy"]
        return ["-c:a", "aac", "-b:a", "192k"]
    elif 5 <= quality_percent <= 24:
        if is_copyable and br_k <= 128:
            return ["-c:a", "copy"]
        return ["-c:a", "aac", "-b:a", "128k"]
    else:  # < 5
        if is_copyable and br_k <= 96:
            return ["-c:a", "copy"]
        return ["-c:a", "aac", "-b:a", "96k"]


def _strip_audio_flags(tokens: List[str]) -> None:
    names = {"-c:a", "-acodec", "-b:a", "-ab", "-ac", "-ar"}
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in names and i + 1 < len(tokens):
            i += 2
            continue
        out.append(t)
        i += 1
    tokens[:] = out


def _ffmpeg_quality_flags(codec_key: str, crf_value: float) -> List[str]:
    key = (codec_key or "").lower()
    if key in {"mpeg4", "mpeg2video", "mpeg1video", "mjpeg"}:
        q = max(2, min(31, int(round(2 + (crf_value - 18) * (6.0 / 12.0)))))
        return ["-qscale:v", str(q)]
    return ["-crf", str(int(round(crf_value)))]


def _strip_flag_with_value(tokens: List[str], flag: str) -> None:
    if flag in tokens:
        idx = tokens.index(flag)
        del tokens[idx : idx + 2]


# ---------- Input-Typing-Helfer (eliminiert list[Unknown]) ----------


def _to_str_list(v: object) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(p) for p in cast(Sequence[object], v) if p is not None]
    return [str(v)]


def _prepare_inputs_typed(args: Any) -> Tuple[bool, List[str]]:
    bm_raw, fl_any = he.prepare_inputs(args)
    return bool(bm_raw), _to_str_list(fl_any)


# ----------------------------- Hauptfunktion -----------------------------


def compress(args: Any) -> None:
    # 0) INTERAKTIV vs. BATCH + Eingaben
    BATCH_MODE, files = _prepare_inputs_typed(args)
    co.print_start(_("compressing_method"))

    # 1) Qualität / CRF
    if BATCH_MODE:
        quality_percent_any: Any = getattr(args, "quality", None)
        quality_percent: int = (
            40 if quality_percent_any is None else int(quality_percent_any)
        )
        if not (0 <= quality_percent <= 100):
            co.print_error(_("invalid_quality_value"))
            return
        crf_value: float = he.map_quality_to_crf(quality_percent)
    else:
        while True:
            quality_percent = ui.read_percent(_("enter_quality_percent") + ": ")
            crf_value = he.map_quality_to_crf(quality_percent)
            sample_path = Path(files[0]).resolve()
            src_codec_ffprobe = _probe_video_codec(sample_path) or "h264"
            formats = [Path(f).suffix.lstrip(".").lower() or "unknown" for f in files]
            vcodecs = [_probe_video_codec(Path(f)) or "unknown" for f in files]
            co.print_selected_params_table(
                {
                    "files": files,
                    "Format": formats,
                    "Codec": vcodecs,
                    "quality": f"{quality_percent}%",
                    "crf": crf_value,
                }
            )
            ok = ui.preview_and_confirm_compress(
                sample_path, crf_value, codec=src_codec_ffprobe
            )
            if ok is None:
                co.print_fail(_("aborted_by_user"))
                return
            if ok is True:
                break

    print()
    total = len(files)

    for i, file in enumerate(files, start=1):
        path = Path(file).resolve()
        if not path.exists():
            co.print_warning(_("file_not_found").format(file=path.name))
            continue

        # Thumbnail ggf. bewahren
        preserved_cover: Optional[Path] = None
        try:
            if vt.check_thumbnail(path, silent=True):
                preserved_cover = vt.extract_thumbnail(path)
        except Exception:
            preserved_cover = None

        # Ausgabe-Dateiname
        suffix = "_compressed_" + f"Q{quality_percent}"
        output = fs.build_output_path(
            input_path=path,
            output_arg=getattr(args, "output", None),
            default_suffix=suffix,
            idx=i,
            total=total,
            target_ext=None,
        )

        # Quelle (Codec/Container)
        src_codec_ffprobe = _probe_video_codec(path) or "h264"
        src_codec_key = vec.normalize_codec_key(src_codec_ffprobe) or src_codec_ffprobe
        target_container = (vec.detect_container_from_path(path) or "mp4").lower()

        # Codec nicht ändern (außer vorhandener Fallback-Logik)
        desired_codec_key_orig = src_codec_key
        desired_codec_key = vec.pick_crf_codec_for_container(
            target_container, desired_codec_key_orig
        )
        if desired_codec_key != desired_codec_key_orig:
            co.print_info(
                _("compress_codec_change").format(
                    ocodec=desired_codec_key_orig, ncodec=desired_codec_key
                )
            )

        # ffprobe: Dauer & FPS (für Kurzclip-Tuning)
        dur = _probe_duration_sec(path) or 0.0
        fps = _probe_fps(path) or 25.0
        very_short = dur > 0 and dur <= 15.0

        # Audio-Policy (gemäß Schwellen)
        a_codec, a_ch, a_bps = _probe_audio_info(path)
        audio_flags = _choose_audio_flags_for_policy(
            a_codec, a_bps, int(quality_percent)
        )

        # Preset patchen (Qualität & Speed)
        _preset_name = "casual"
        _store: Dict[str, Any] = cast(
            Dict[str, Any], defin.CONVERT_PRESET.get(_preset_name, {})
        )

        _preset_max_fps_noop = 9999

        try:
            patched: Dict[str, Any] = dict(_store)
            patched["quality"] = int(crf_value)
            patched["speed"] = "slow"
            if target_container in ("mp4", "m4v"):
                patched["faststart"] = True

            _PRESETS: MutableMapping[str, Any] = cast(
                MutableMapping[str, Any], defin.CONVERT_PRESET
            )
            _PRESETS[_preset_name] = patched

            # Plan (Encoderwahl & sinnvolle -vf/-pix_fmt nur wenn nötig)
            plan = vec.build_transcode_plan(
                input_path=path,
                target_container=target_container,
                preset_name=_preset_name,
                codec_key=desired_codec_key,
                preferred_encoder=None,
                req_scale=None,
                src_w=None,
                src_h=None,
                src_fps=None,
                user_fps_rational=None,
                preset_max_fps=_preset_max_fps_noop,
                force_key_at_start=True,
            )

            # Basiskommando
            cmd: List[str] = list(plan.final_cmd_without_output)
            cmd = vec.postprocess_cmd_all_presets(cmd, plan)

            # 1) Audio-Flags unserer Policy durchsetzen:
            _strip_audio_flags(cmd)
            cmd += audio_flags

            # 2) Kurzclip-GOP nur für sehr kurze Clips:
            if very_short:
                target_g = int(min(max(int(round(fps * 2)), 12), 120))
                if "-g" in cmd:
                    gi = cmd.index("-g")
                    cmd[gi + 1] = str(target_g)
                else:
                    cmd += ["-g", str(target_g)]
                _strip_flag_with_value(cmd, "-keyint_min")

            # === Smart keep: in Temp encoden, Größen vergleichen ===
            with tempfile.TemporaryDirectory() as td:
                tmp_out = Path(td) / (output.name)
                probe_cmd = cmd + [str(tmp_out)]
                autotune_final_cmd(path, probe_cmd)
                try:
                    pw.run_ffmpeg_with_progress(
                        path.name,
                        probe_cmd,
                        _("analysing_video"),
                        _("analysing_video_done"),
                    )
                except subprocess.CalledProcessError:
                    co.print_fail(_("ffmpeg_failed"))
                    continue

                try:
                    in_sz = path.stat().st_size
                    out_sz = tmp_out.stat().st_size
                except Exception:
                    in_sz, out_sz = None, None

                if in_sz and out_sz and out_sz >= int(in_sz * 0.98):
                    shutil.copy2(str(path), str(output))
                    co.print_info(_("smart_keep_used"))
                else:
                    shutil.move(str(tmp_out), str(output))
                    co.print_info(
                        _("compressing_file_finished").format(file=str(output.name))
                    )

        except Exception:
            # Robustes Fallback
            enc_out = _as_str(_encoder_for_codec(desired_codec_key))
            qflags = _ffmpeg_quality_flags(desired_codec_key, crf_value)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-c:v",
                enc_out,
                *qflags,
                "-preset",
                "slow",
                *audio_flags,
                str(output),
            ]
            autotune_final_cmd(path, cmd)
            pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("compressing_file_progress"),
                _("compressing_file_done"),
                output,
                BATCH_MODE=BATCH_MODE,
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

        # Größeninfo
        try:
            original_size = path.stat().st_size / (1024 * 1024)
            compressed_size = output.stat().st_size / (1024 * 1024)
            file_info = (
                _("original size")
                + " = "
                + co.return_promt(f"{original_size:.2f} MB", color="bright_blue")
                + " → "
                + _("compressed size")
                + " = "
                + co.return_promt(f"{compressed_size:.2f} MB", color="bright_blue")
            )
            co.print_info(file_info, color="reset")
        except Exception:
            pass

    co.print_finished(_("compressing_method"))
