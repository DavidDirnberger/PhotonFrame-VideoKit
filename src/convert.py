#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import consoleOutput as co
import definitions as defin
import fileSystem as fs
import helpers as he
import i18n as _i18n
import info_helpers as ih
import process_wrappers as pw
import userInteraction as ui
import video_pixfmt as vp
import video_thumbnail as vt
import VideoEncodersCodecs as vec

# local modules
from ffmpeg_perf import autotune_final_cmd

_RES_CUSTOM_RE = re.compile(r"^\s*custom\s*(?:=|:|\s)\s*(\d+)\s*[:x]\s*(\d+)\s*$", re.I)
_RES_PLAIN_RE = re.compile(r"^\s*(\d+)\s*[:x]\s*(\d+)\s*$")
_FPS_RE = re.compile(r"^\s*(?:([0-9]+(?:\.[0-9]+)?)|([0-9]+)\s*/\s*([0-9]+))\s*$")

# ---- Typing helpers ----------------------------------------------------------


class _Translator(Protocol):
    def __call__(self, value: object | None = ...) -> str: ...


_: _Translator = cast(_Translator, _i18n._)
tr: _Translator = cast(_Translator, _i18n.tr)


def _list_str_factory() -> List[str]:
    """Explizit getypte Factory für List[str], damit Pylance kein list[Unknown] annimmt."""
    return []


@dataclass
class ConvertArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    preset: Optional[str] = "casual"
    format: Optional[str] = None
    resolution: Optional[str] = "original"
    resolution_scale: Optional[str] = None
    framerate: Optional[str] = "original"
    codec: Optional[str] = None
    output: Optional[str] = None


# --- kleine, aber wichtige Typ-Helfer ----------------------------------------


def _to_str_list(v: object) -> List[str]:
    """
    Normalisiert beliebige File-Argumente strikt zu List[str].
    Vermeidet Unknown in Comprehensions.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(p) for p in cast(Sequence[object], v) if p is not None]
    return [str(v)]


def _prepare_inputs_typed(args: Any) -> Tuple[bool, List[str]]:
    """
    Typisierter Wrapper um he.prepare_inputs:
    - BATCH_MODE als bool
    - files garantiert als List[str]
    """
    bm_raw, fl_any = he.prepare_inputs(args)
    files_list: List[str] = _to_str_list(fl_any)
    return bool(bm_raw), files_list


def _merge_with_convert_defaults(args: Any) -> ConvertArgs:
    cfg = ConvertArgs()

    # files sauber zu List[str] normalisieren (ohne Unknown in Comprehension)
    if hasattr(args, "files"):
        cfg.files = _to_str_list(getattr(args, "files"))

    # restliche (stringartige) Felder setzen – leere Listen ignorieren
    def _set_str_opt(name: str) -> None:
        if hasattr(args, name):
            val = getattr(args, name)
            if val is None:
                return
            if isinstance(val, list) and val == []:
                return
            setattr(cfg, name, str(val))

    for n in ("preset", "format", "resolution", "framerate", "codec", "output"):
        _set_str_opt(n)

    cfg.resolution, cfg.resolution_scale = _parse_resolution_arg(cfg.resolution)

    return cfg


def _parse_resolution_arg(val: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Interpretiert '--resolution custom=WxH', 'custom:WxH', 'custom WxH' oder nur 'WxH'
    → ('custom', 'W:H'). Andernfalls → (val or 'original', None).
    """
    if not val:
        return ("original", None)
    s = str(val)
    m = _RES_CUSTOM_RE.match(s)
    if m:
        w, h = m.group(1), m.group(2)
        return ("custom", f"{w}:{h}")
    m2 = _RES_PLAIN_RE.match(s)
    if m2:
        w, h = m2.group(1), m2.group(2)
        return ("custom", f"{w}:{h}")
    return (s, None)


def _parse_framerate_arg(val: Optional[str]) -> Optional[str]:
    """
    Akzeptiert:
      - ganze/float Zahl: '20', '23.976'
      - rational: '24000/1001', '30000/1001'
    Rückgabe: normalisierte Zeichenkette für ffmpeg ('num/den' oder Zahl) oder None bei Ungültigkeit.
    """
    if not val:
        return None
    s = str(val).strip()
    if s.lower() == "original":
        return None
    m = _FPS_RE.match(s)
    if not m:
        return None
    # Fall 1: Dezimalzahl
    if m.group(1):
        try:
            f = float(m.group(1))
            if 1.0 <= f <= 240.0:
                # Schlank formatieren (ohne überflüssige .0)
                txt = f"{f}".rstrip("0").rstrip(".")
                return txt if txt else None
        except Exception:
            return None
        return None
    # Fall 2: Rational num/den
    try:
        num = int(m.group(2))
        den = int(m.group(3))
        if den <= 0:
            return None
        valf = num / den
        if 1.0 <= valf <= 240.0:
            return f"{num}/{den}"
    except Exception:
        return None
    return None


# --- Lossless Cache -----------------------------------------------------------
_LOSSLESS_CACHE: Dict[str, Dict[str, Dict[str, List[str]]]] = {}


class _PlanAdapter:
    """Adapter, damit {final_cmd_without_output: [...]} erwartet werden kann."""

    def __init__(self, tokens: Sequence[Any]):
        self.final_cmd_without_output: List[str] = [str(t) for t in tokens]


T = TypeVar("T")


def _first_of_iterable(seq: Iterable[T]) -> Optional[T]:
    """
    Gibt das erste Element der Iteration zurück oder None, wenn leer.
    (Kein `for x in seq` -> verhindert 'Type of x is unknown')
    """
    it = iter(seq)
    try:
        return next(it)
    except StopIteration:
        return None


def _normalize_encoder_candidate(val: Any) -> Any:
    """
    Normalisiert beliebige Encoder-Kandidaten (Listen/Tuples/Dicts/Objekte).
    """
    v: Any = val
    while isinstance(v, (list, tuple)):
        if not v:
            return None
        nxt: Optional[Any] = _first_of_iterable(cast(Iterable[Any], v))
        if nxt is None:
            return None
        v = nxt

    if isinstance(v, dict):
        picked: Optional[Any] = None
        for _k, _vv in cast(Mapping[Any, Any], v).items():
            norm = _normalize_encoder_candidate(_vv)
            if norm is not None:
                picked = norm
                break
        if picked is None:
            return None
        v = picked

    if hasattr(v, "final_cmd_without_output"):
        return v

    if isinstance(v, (list, tuple)):
        return _PlanAdapter(cast(Sequence[Any], v))

    return v


def _lossless_cache_key(files: List[str], preset: str) -> str:
    first = str(Path(files[0]).resolve()) if files else ""
    try:
        mtime = int(Path(files[0]).stat().st_mtime)
    except Exception:
        mtime = 0
    return f"{preset}|{first}|{len(files)}|{mtime}"


def _compute_and_cache_lossless_for_all_containers(
    files: List[str],
    format_keys: List[str],
    preset: str = "lossless",
) -> Dict[str, Dict[str, List[str]]]:
    key = _lossless_cache_key(files, preset)
    cached = _LOSSLESS_CACHE.get(key)
    if cached is not None:
        return cached

    if not files:
        _LOSSLESS_CACHE[key] = {}
        _LOSSLESS_CACHE["_last"] = {}
        return {}

    container_to_codecmap: Dict[str, OrderedDict[str, Any]] = OrderedDict()
    for fk in format_keys:
        try:
            emap = cast(
                Dict[str, OrderedDict[str, Any]],
                vec.prepare_encoder_maps(
                    files,
                    fk,
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                ),
            )
        except Exception:
            emap = {}

        for cont, cmap in (emap or {}).items():
            if cont not in container_to_codecmap:
                container_to_codecmap[cont] = OrderedDict(cmap)
            else:
                for ck, v in cmap.items():
                    container_to_codecmap[cont].setdefault(ck, v)

    if not container_to_codecmap:
        try:
            emap_keep = cast(
                Dict[str, OrderedDict[str, Any]],
                vec.prepare_encoder_maps(
                    files,
                    "keep",
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                ),
            )
            if emap_keep:
                container_to_codecmap.update(emap_keep)
        except Exception as e:
            co.print_error(
                _("prepare_encoder_maps_fallback_keep_failed").format(error=str(e))
            )

    if not container_to_codecmap:
        _LOSSLESS_CACHE[key] = {}
        _LOSSLESS_CACHE["_last"] = {}
        return {}

    squashed: Dict[str, OrderedDict[str, Any]] = OrderedDict()
    for cont, cmap in container_to_codecmap.items():
        out: OrderedDict[str, Any] = OrderedDict()
        for ck, raw in cmap.items():
            cand = _normalize_encoder_candidate(raw)
            if cand is None:
                continue
            if isinstance(cand, (list, tuple)):
                cand = _PlanAdapter(cand)
            out[ck] = cand
        squashed[cont] = out

    try:
        matrix: Dict[str, Dict[str, List[str]]] = vec.find_lossless_combinations(
            input_path=Path(files[0]),
            container_to_codecmap=squashed,
            req_scale=None,
            user_fps_rational=None,
            preset_max_fps=None,
        )
    except Exception:
        matrix = {}

    ALIASES = {"m4v": "mp4"}
    for alias, target in ALIASES.items():
        if alias in format_keys and alias not in matrix and target in matrix:
            matrix[alias] = matrix[target]

    _LOSSLESS_CACHE[key] = matrix
    _LOSSLESS_CACHE["_last"] = matrix
    return matrix


# --- Tabellen-Param-Building --------------------------------------------------
def _build_params_for_table_interactive(
    *,
    files: List[str],
    preset_choice: Optional[str],
    format_choice: Optional[str],
    resolution_key: Optional[str],
    resolution_scale: Optional[str],
    codec_choice_by_container: Mapping[str, str],
    presets: Mapping[str, Mapping[str, Any]],
    res_defs: Mapping[str, Mapping[str, Any]],
    user_fps_rational: Optional[str] = None,
    target_pix_fmt: Optional[str] = None,
    profile_hint: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _target_container() -> Optional[str]:
        if format_choice != "keep":
            return (format_choice or "").lower() if format_choice else None
        return vec.detect_container_from_path(Path(files[0])) if files else None

    def _resolution_display(res_key: Optional[str], scale: Optional[str]) -> Any:
        if res_key == "custom":
            return {
                "de": f"Benutzerdefiniert ({scale or '—'})",
                "en": f"Custom ({scale or '—'})",
            }
        if res_key in ("original", None):
            return {"de": "Original", "en": "Original"}
        name = (res_defs.get(res_key or "", {}) or {}).get(
            "name", res_key or "original"
        )
        return name

    def _fr_label(preset_name: Optional[str], user_cap: Optional[str]) -> Any:
        if user_cap:
            return {
                "de": f"Begrenzen auf {user_cap} fps",
                "en": f"Cap at {user_cap} fps",
            }
        max_fps = (presets.get(preset_name or "", {}) or {}).get("max_fps")
        if max_fps:
            return f"≤{max_fps}"
        return {"de": "Original", "en": "Original"}

    def _summarize_codec_choice_for_table(
        fmt_choice: Optional[str],
        files_list: List[str],
        by_container: Mapping[str, str],
    ) -> str:
        def _detect(p: str) -> Optional[str]:
            return vec.detect_container_from_path(Path(p))

        if fmt_choice != "keep":
            cont = (fmt_choice or "").lower()
            ck = by_container.get(cont, "copy")
            return ck
        seen: List[str] = []
        parts: List[str] = []
        for f in files_list:
            c = _detect(f)
            if c and c not in seen:
                seen.append(c)
                cc = cast(str, c)
                parts.append(f"{cc}:{by_container.get(cc, 'copy')}")
        return ", ".join(parts) if parts else "copy"

    def _humanize_audio(container: Optional[str]) -> str:
        c = (container or "").lower()
        if c in ("mp4", "mov", "m4v"):
            return "aac 128k stereo"
        if c == "webm":
            return "opus 112k stereo"
        return "copy"

    tgt_container = _target_container()

    preset_key = preset_choice or ""
    format_key = format_choice or ""

    preset_val = (presets.get(preset_key, {}) or {}).get("name", preset_choice)
    format_val = (defin.CONVERT_FORMAT_DESCRIPTIONS.get(format_key, {}) or {}).get(
        "name", format_choice
    )

    if resolution_key == "custom":
        scale_val = resolution_scale
    elif resolution_key in ("original", None):
        scale_val = None
    else:
        scale_val = (res_defs.get(resolution_key or "", {}) or {}).get("scale")

    resolution_val = _resolution_display(resolution_key, scale_val)
    video_codec_val = _summarize_codec_choice_for_table(
        format_choice, files, codec_choice_by_container
    )
    framerate_val = _fr_label(preset_choice, user_fps_rational)
    audio_codec_val = _humanize_audio(tgt_container) if tgt_container else None
    faststart_on = bool(
        (presets.get(preset_key, {}) or {}).get("faststart") and tgt_container == "mp4"
    )

    params: Dict[str, Any] = {
        "files": [str(Path(f)) for f in files],
        "preset": preset_val,
        "format": format_val,
        "video_codec": video_codec_val,
        "resolution": resolution_val,
        "framerate": framerate_val,
    }
    if scale_val:
        params["scale"] = scale_val
    if audio_codec_val:
        params["audio_codec"] = audio_codec_val
    if faststart_on:
        params["faststart"] = {"de": "an", "en": "on"}
    if target_pix_fmt:
        params["target_pix_fmt"] = target_pix_fmt
    if profile_hint:
        params["profile"] = profile_hint

    labels: Dict[str, Any] = {
        "preset": {"de": "Voreinstellung", "en": "Preset"},
        "format": {"de": "Format", "en": "Format"},
        "video_codec": {"de": "Video-Codec", "en": "Video codec"},
        "resolution": {"de": "Zielauflösung", "en": "Target resolution"},
        "scale": {"de": "Skalierung", "en": "Scale"},
        "framerate": {"de": "Bildrate", "en": "Framerate"},
        "audio_codec": {"de": "Audio-Codec", "en": "Audio codec"},
        "faststart": {"de": "Faststart (MP4)", "en": "Faststart (MP4)"},
        "target_pix_fmt": {"de": "Pixelformat", "en": "Pixel format"},
        "profile": {"de": "Profil", "en": "Profile"},
    }
    return params, labels


def _plan_to_cmd(
    plan: Any,
    *,
    input_path: Path,
    output: Path,
    target_container: str,
    preset_name: str,
) -> List[str]:
    """
    Erzeugt eine komplette ffmpeg-CLI aus Plan-Objekt oder Token-Fragment.
    """
    if isinstance(plan, (list, tuple)):
        tokens: List[str] = [str(t) for t in cast(Sequence[Any], plan)]
        if tokens and "ffmpeg" in tokens[0]:
            return tokens + [str(output)]

        base = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-stats_period",
            "0.5",
            "-i",
            str(input_path),
        ]
        try:
            base += vec.build_stream_mapping_args(target_container)
        except Exception:
            base += ["-map", "0:v:0", "-map", "0:a?"]
        try:
            base += vec.build_audio_args(target_container, preset_name, input_path)
        except Exception:
            base += ["-c:a", "copy"]

        return base + tokens + [str(output)]

    return vec.assemble_ffmpeg_cmd(plan, output)


# --- Lossless-Badges ----------------------------------------------------------
def _badge(color: str) -> str:
    return {"green": "🟢", "blue": "🔵"}.get(color, "")


def _lossless_mark_label(base_label: str, *, copy_ok: bool, strict_ok: bool) -> str:
    badges: List[str] = []
    if copy_ok:
        badges.append(_badge("green"))
    if strict_ok:
        badges.append(_badge("blue"))
    suffix = " ".join(b for b in badges if b)
    label = f"{base_label} {suffix}" if suffix else base_label
    color = "khaki"
    return co.return_promt(label, color=color)


def convert(args: Any) -> None:
    # 0) Inputs (mit voll typisiertem Wrapper) – bewusst ohne Tuple-Unpacking
    _inp: Tuple[bool, List[str]] = _prepare_inputs_typed(args)
    BATCH_MODE: bool = _inp[0]
    files: List[str] = _inp[1]

    co.print_start(_("converting_method"))

    # Einheitliche Typen für beide Pfade
    preset_choice: Optional[str] = None
    format_choice: Optional[str] = None
    resolution: Optional[str] = None
    framerate: str = "original"
    resolution_scale: Optional[str] = None
    user_fps_rational: Optional[str] = None

    encoder_maps_by_container: Dict[str, OrderedDict[str, Any]] = OrderedDict()
    codec_choice_by_container: Dict[str, str] = {}

    # 1) Parameter
    if BATCH_MODE:
        cfg = _merge_with_convert_defaults(args)
        preset_choice = cfg.preset or "casual"
        format_choice = cfg.format or "keep"
        resolution = cfg.resolution or "original"
        framerate = cfg.framerate or "original"  # spätere Anzeige/Suffix
        codec_key_def = vec.normalize_codec_key(cfg.codec) or "h264"
        fps_norm = _parse_framerate_arg(framerate)
        user_fps_rational = fps_norm

        # container/codec aus deiner CLI
        warnings, errors, suggestions = vec.plan_warnings_and_errors_for_choice(
            input_path=Path(files[0]),
            target_container=format_choice,  # z.B. "mp4", "mkv", ...
            codec_key=codec_key_def,  # z.B. "h264", "prores", ...
            convert_format_descriptions=defin.CONVERT_FORMAT_DESCRIPTIONS,
            encoder_name=None,  # optional, sonst aus Codec abgeleitet
        )

        if errors:
            for e in errors:
                co.print_error(e)
            return

        for w in warnings:
            co.print_warning(w)  # oder deine i18n-Ausgabe

        # Optional: Alternativen anzeigen (Container, Codec)
        if suggestions:
            pretty = ", ".join([f"{c}+{k}" for (c, k) in suggestions[:6]])
            co.print_info(_("alpha_alternative_suggestions").format(list=pretty))

        if framerate != "original" and user_fps_rational is None:
            co.print_warning(_("invalid_framerate_arg").format(value=str(framerate)))
            framerate = "original"

        resolution_scale = cfg.resolution_scale

        if resolution == "custom" and resolution_scale:
            try:
                min_src_w, min_src_h = vec.probe_min_source_geometry(cfg.files or files)
                ok, msg = vec.validate_custom_scale_leq(
                    min_src_w, min_src_h, resolution_scale, require_even=True
                )
                if not ok:
                    co.print_warning(msg)
                    resolution = "original"
                    resolution_scale = None
            except Exception as e:
                co.print_warning(
                    _("custom_resolution_validation_failed").format(error=str(e))
                )
                resolution = "original"

        encoder_maps_by_container = cast(
            Dict[str, OrderedDict[str, Any]],
            vec.prepare_encoder_maps(
                files,
                format_choice,
                defin.CONVERT_FORMAT_DESCRIPTIONS,
                prefer_hw=True,
                ffmpeg_bin="ffmpeg",
            ),
        )
        for c, enc_map in encoder_maps_by_container.items():
            if codec_key_def == "copy":
                codec_choice_by_container[c] = "copy"
            elif codec_key_def in enc_map or vec.container_allows_codec(
                c, codec_key_def
            ):
                # User-Wunsch respektieren, auch wenn prepare_encoder_maps ihn nicht gelistet hat
                codec_choice_by_container[c] = codec_key_def
            elif "h264" in enc_map:
                codec_choice_by_container[c] = "h264"
            else:
                codec_choice_by_container[c] = next(iter(enc_map.keys()))

        # resolution_scale = None

        params, labels = _build_params_for_table_interactive(
            files=files,
            preset_choice=preset_choice,
            format_choice=format_choice,
            resolution_key=resolution,
            resolution_scale=resolution_scale,
            codec_choice_by_container=codec_choice_by_container,
            presets=defin.CONVERT_PRESET,
            res_defs=defin.RESOLUTIONS,
            user_fps_rational=user_fps_rational,
        )
    else:
        while True:
            # Preset
            preset_keys = list(defin.CONVERT_PRESET)
            preset_descriptions = [
                str(tr(defin.CONVERT_PRESET[k].get("description") or ""))
                for k in preset_keys
            ]
            preset_labels = [
                str(defin.CONVERT_PRESET[k].get("name", k)) for k in preset_keys
            ]
            preset_choice = ui.ask_user(
                _("choose_quality_preset"),
                preset_keys,
                preset_descriptions,
                3,
                preset_labels,
                back_button=False,
            )
            if preset_choice is None:
                continue

            # Infoblöcke
            info = ih.general_info_for_preset(preset_choice)
            print()
            co.print_headline(str(tr(info["headline"])), "orchid")
            for block in info["blocks"]:
                co.print_line(
                    str(tr(block["title"])), color="bright_orange", style="bold"
                )
                names = [
                    str(b.get("name", ""))
                    for b in block.get("bullets", [])
                    if isinstance(b, dict)
                ]
                description = [
                    (
                        str(tr(b.get("description")))
                        if b.get("description") is not None
                        else ""
                    )
                    for b in block.get("bullets", [])
                    if isinstance(b, dict)
                ]
                co.print_bullet_list(names, description)
            print()

            # Format-Auswahl (inkl. Lossless-Badges)
            format_keys = list(defin.CONVERT_FORMAT_DESCRIPTIONS)
            format_descriptions = [
                str(tr(defin.CONVERT_FORMAT_DESCRIPTIONS[k].get("description") or ""))
                for k in format_keys
            ]

            lossless_matrix: Dict[str, Dict[str, List[str]]] = {}

            if preset_choice == "lossless":
                print()
                co.print_text(str(tr(ih.LOSSLESS_DISCLAIMER)))
                print()
                co.print_line("    " + _badge("green") + " " + _("bitstream_copy"))
                co.print_line("    " + _badge("blue") + " " + _("strict_copy"))
                print()
                co.print_line("    " + _("analysing_lossless"), color="gold")

                lossless_all = _compute_and_cache_lossless_for_all_containers(
                    files, format_keys, preset_choice
                )

                format_labels: List[str] = []
                for fk in format_keys:
                    base_label = str(
                        tr(defin.CONVERT_FORMAT_DESCRIPTIONS[fk].get("name", fk))
                    )
                    if fk == "keep":
                        has_copy = any(
                            bool((lossless_all.get(c, {}).get("copy")))
                            for c in (lossless_all or {})
                        )
                        has_strict = any(
                            bool((lossless_all.get(c, {}).get("strict")))
                            for c in (lossless_all or {})
                        )
                    else:
                        mtx = lossless_all.get(fk, {})
                        has_copy = bool(mtx.get("copy"))
                        has_strict = bool(mtx.get("strict"))
                    format_labels.append(
                        _lossless_mark_label(
                            base_label, copy_ok=has_copy, strict_ok=has_strict
                        )
                    )
                co.delete_last_n_lines(2)
            else:
                format_labels = [
                    str(tr(defin.CONVERT_FORMAT_DESCRIPTIONS[fk].get("name", fk)))
                    for fk in format_keys
                ]

            format_choice = ui.ask_user(
                _("choose_format"), format_keys, format_descriptions, 0, format_labels
            )
            if format_choice is None:
                continue

            # Encoder-Map
            encoder_maps_by_container = cast(
                Dict[str, OrderedDict[str, Any]],
                vec.prepare_encoder_maps(
                    files,
                    format_choice,
                    defin.CONVERT_FORMAT_DESCRIPTIONS,
                    prefer_hw=True,
                    ffmpeg_bin="ffmpeg",
                ),
            )

            # Lossless-Matrix (nur für Labeling/Infos)
            if preset_choice == "lossless":
                _cache_key = _lossless_cache_key(files, preset_choice)
                lossless_matrix = (
                    _LOSSLESS_CACHE.get(_cache_key)
                    or _LOSSLESS_CACHE.get("_last")
                    or {}
                )

            # Codec je Container
            codec_choice_by_container = {}
            ans: Optional[str] = None
            for c in encoder_maps_by_container.keys():
                encoders_codec_keys: List[str] = list(
                    encoder_maps_by_container[c].keys()
                )
                codec_descriptions = [
                    str(tr(defin.VIDEO_CODECS[e].get("description") or ""))
                    for e in encoders_codec_keys
                ]
                if preset_choice == "lossless":
                    codec_labels: List[str] = []
                    mtx = lossless_matrix.get(c, {"copy": [], "strict": []})
                    copy_set = set(mtx.get("copy", []))
                    strict_set = set(mtx.get("strict", []))
                    for e in encoders_codec_keys:
                        base = str(defin.VIDEO_CODECS[e].get("name", e))
                        copy_ok = e in copy_set
                        lbl = _lossless_mark_label(
                            base,
                            copy_ok=copy_ok,
                            strict_ok=False if copy_ok else e in strict_set,
                        )
                        codec_labels.append(lbl)
                else:
                    codec_labels = [
                        str(defin.VIDEO_CODECS[e].get("name", e))
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
                codec_choice_by_container[c] = ans

            if ans is None:
                continue

            # Auflösung
            if preset_choice == "lossless":
                resolution = "original"
                resolution_scale = None
            else:
                min_src_w, min_src_h = vec.probe_min_source_geometry(files)
                resolution_keys = list(defin.RESOLUTIONS)
                filtered_keys = vec.filter_resolution_keys_for_source(
                    cast(Mapping[str, Mapping[str, str]], defin.RESOLUTIONS),
                    resolution_keys,
                    min_src_w,
                    min_src_h,
                )
                resolution_descriptions = [
                    str(tr(defin.RESOLUTIONS[k].get("description") or ""))
                    for k in filtered_keys
                ]
                resolution_labels = [
                    str(defin.RESOLUTIONS[k].get("name", k)) for k in filtered_keys
                ]
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
                        candidate = ui.ask_two_ints(
                            _("enter_custom_resolution")
                        )  # "W:H"
                        ok, msg = vec.validate_custom_scale_leq(
                            min_src_w, min_src_h, candidate, require_even=True
                        )
                        if ok:
                            resolution_scale = candidate
                            break
                        co.print_fail(msg)

            # FPS-Cap
            if preset_choice == "lossless":
                framerate = "original"
                user_fps_rational = None
            else:
                user_fps_rational = None
                min_fps, max_fps = he.get_framerate_range(files)
                max_fps_disp = (
                    f"{max_fps:.2f}" if isinstance(max_fps, (int, float)) else "?"
                )
                preset_max_fps = defin.CONVERT_PRESET[preset_choice].get("max_fps")
                if preset_max_fps is None:
                    ans_cap = ui.ask_yes_no(
                        _("want_framerate_cap").format(max_fps=max_fps_disp)
                    )
                    if ans_cap is None:
                        continue
                    if ans_cap:
                        default_val = (
                            "30"
                            if isinstance(min_fps, (int, float)) and min_fps >= 30
                            else (
                                str(min_fps)
                                if isinstance(min_fps, (int, float))
                                else "30"
                            )
                        )
                        max_val = (
                            float(min_fps)
                            if isinstance(min_fps, (int, float))
                            else 30.0
                        )
                        val = ui.ask_float(
                            _("enter_framerate_cap"),
                            default=default_val,
                            min_value=1,
                            max_value=max_val,
                            return_type="rational",
                        )
                        user_fps_rational = str(val) if val is not None else None

                framerate = (
                    "original" if user_fps_rational is None else user_fps_rational
                )

            params, labels = _build_params_for_table_interactive(
                files=files,
                preset_choice=preset_choice,
                format_choice=format_choice,
                resolution_key=resolution,
                resolution_scale=resolution_scale,
                codec_choice_by_container=codec_choice_by_container,
                presets=defin.CONVERT_PRESET,
                res_defs=defin.RESOLUTIONS,
                user_fps_rational=user_fps_rational,
            )
            break

    # 2) Status
    co.print_selected_params_table(params, labels=labels)

    # 3) Konvertierung
    for i, file in enumerate(files):
        input_path = Path(file).resolve()
        if not input_path.exists():
            co.print_warning(_("file_not_found").format(file=input_path.name))
            continue

        # Cover der Quelle sichern
        preserved_cover: Optional[Path] = None
        try:
            if vt.check_thumbnail(input_path, silent=True):
                preserved_cover = vt.extract_thumbnail(input_path)
        except Exception:
            preserved_cover = None

        target_container = (
            (format_choice or "keep").lower()
            if format_choice != "keep"
            else (vec.detect_container_from_path(input_path) or "mkv")
        )

        encoder_map: OrderedDict[str, Any] = encoder_maps_by_container.get(
            target_container, OrderedDict()
        )

        # Quelle
        src_fps = he.probe_src_fps(input_path)
        src_w, src_h, src_fmt = vec.ffprobe_geometry(input_path)

        # Container/Codec prüfen
        chosen_codec_key = codec_choice_by_container.get(target_container, "copy")
        if chosen_codec_key != "copy" and not vec.container_allows_codec(
            target_container, chosen_codec_key
        ):
            sugg = vec.suggest_codec_for_container(target_container)
            co.print_warning(
                _("unsuitable_codec").format(
                    codec=chosen_codec_key, container=target_container, sugg=sugg
                )
            )
            chosen_codec_key = sugg

        preferred: Optional[Any] = (
            encoder_map.get(chosen_codec_key) if chosen_codec_key != "copy" else None
        )

        # Ziel-Skalierung
        if resolution == "custom" and resolution_scale:
            req_scale: Optional[str] = resolution_scale
        elif resolution != "original":
            # FIX: cast(str, resolution) statt cast[str](resolution)
            req_scale = str(defin.RESOLUTIONS[cast(str, resolution)]["scale"])
        else:
            preset_scale = defin.CONVERT_PRESET.get(cast(str, preset_choice), {}).get(
                "scale"
            )
            req_scale = preset_scale if isinstance(preset_scale, str) else None

        # Plan bauen
        plan = vec.build_transcode_plan(
            input_path=input_path,
            target_container=target_container,
            preset_name=cast(str, preset_choice),
            codec_key=chosen_codec_key,
            preferred_encoder=preferred,
            req_scale=req_scale,
            src_w=src_w,
            src_h=src_h,
            src_fps=src_fps,
            user_fps_rational=user_fps_rational,
            preset_max_fps=defin.CONVERT_PRESET[cast(str, preset_choice)].get(
                "max_fps"
            ),
        )

        eff_pix_fmt = vp.infer_target_pix_fmt_from_plan(plan)
        co.print_info(_("target_pix_fmt") + f" => {eff_pix_fmt or '—'}")

        in_ext = input_path.suffix.lstrip(".").lower()
        out_ext = target_container

        is_plan_obj = hasattr(plan, "codec_key")
        real_codec_key = (
            getattr(plan, "codec_key", chosen_codec_key) or chosen_codec_key
        )
        plan_safe_scale = getattr(plan, "safe_scale", None) if is_plan_obj else None

        suffix_parts: List[str] = []
        if real_codec_key != "copy":
            suffix_parts.append(real_codec_key)
        if plan_safe_scale:
            suffix_parts.append(str(plan_safe_scale))
        if framerate != "original":
            suffix_parts.append(f"{framerate}fps")

        suffix = "_converted" + (
            ("_" + "_".join(map(str, suffix_parts))) if suffix_parts else ""
        )
        suffix = suffix.replace(":", "X").replace("/", "_")

        output = fs.build_output_path(
            input_path=input_path,
            output_arg=getattr(args, "output", None),
            default_suffix=suffix,
            idx=i,
            total=len(files),
            target_ext=None if in_ext == out_ext else out_ext,
        )

        final_cmd = _plan_to_cmd(
            plan,
            input_path=input_path,
            output=output,
            target_container=target_container,
            preset_name=cast(str, preset_choice),
        )
        final_cmd = vec.apply_container_codec_quirks(
            final_cmd, target_container, chosen_codec_key
        )

        final_cmd = autotune_final_cmd(input_path, final_cmd)

        # co.print_debug("ffmeg_cmd",final_cmd=final_cmd)

        # subprocess.run(final_cmd)

        pw.run_ffmpeg_with_progress(
            input_path.name,
            final_cmd,
            _("converting_file_progress"),
            _("converting_file_done"),
            output,
            BATCH_MODE=BATCH_MODE,
        )

        # Cover zurückschreiben
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

    co.print_finished(_("converting_method"))
