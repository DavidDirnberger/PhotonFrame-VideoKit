# cli_parser.py
from __future__ import annotations

import argparse
import dataclasses
import re
from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)

try:
    from argcomplete.completers import (  # type: ignore
        DirectoriesCompleter,
        FilesCompleter,
    )
except Exception:
    FilesCompleter = None  # type: ignore[assignment]
    DirectoriesCompleter = None  # type: ignore[assignment]
import definitions as defin
from i18n import _

# ========= Negative-Value Normalizer (global) =========

# Flags, die negative "Zeit/Prozent" als NÄCHSTES Token akzeptieren sollen (trim)
_NEG_TIME_FLAGS = {"--start", "-s", "--duration", "-d", "--end", "-e"}

# Flags, die negative INTS als NÄCHSTES Token akzeptieren sollen
_NEG_INT_FLAGS = {"--offset-x", "-x", "--offset-y", "-y"}

# Flags, die negative PAARE (X:Y / XxY / X,Y / X|Y / X;Y / X/Y) als NÄCHSTES Token akzeptieren sollen
_NEG_PAIR_FLAGS = {
    "--offset",
    "-O",
    "-off",
}  # -off bleibt als Legacy-Alias im Normalizer erhalten

_SEPS = (":", "x", "X", ",", "/", ";", "|")

_TIME_RE = re.compile(
    r"""
    ^\s*-
    (?:
        (?:(\d+):(\d+):(\d+(?:[.,]\d+)?)) |    # -HH:MM:SS(.ms)
        (?:(\d+):(\d+(?:[.,]\d+)?))       |    # -MM:SS(.ms)
        (\d+(?:[.,]\d+)?)                      # -SS(.ms)
    )
    (?:\s*[%pP])?$                              # optional %/p/P am Ende
""",
    re.X,
)

_INT_RE = re.compile(r"^\s*-\d+\s*$")

_PAIR_RE = re.compile(r"^\s*-?\d+\s*(?:[:xX,/\|;])\s*-?\d+\s*$")


def _looks_like_neg_time_or_percent(s: str) -> bool:
    return bool(_TIME_RE.match(s or ""))


def _looks_like_neg_int(s: str) -> bool:
    return bool(_INT_RE.match(s or ""))


def _looks_like_pair(s: str) -> bool:
    return bool(_PAIR_RE.match(s or ""))


def normalize_cli_argv(argv: List[str]) -> List[str]:
    """
    Vereinheitlichte Vorverarbeitung für ALLE Subcommands:
      - '--start -20:00'   -> '--start=-20:00'
      - '--offset -400:20' -> '--offset=-400:20'
      - '--offset-x -300'  -> '--offset-x=-300'
      - Legacy '-off' wird auf '--offset' gemappt (Kompatibilität).
    """
    if not argv:
        return argv
    out = [argv[0]]
    i = 1
    L = len(argv)
    while i < L:
        tok = argv[i]

        # Legacy-Kurzalias '-off' auf Long-Form mappen (bevor wir mergen)
        if tok == "-off":
            tok = "--offset"

        # Merge-Regeln: nur wenn es einen "nächsten" Token gibt
        if (i + 1) < L:
            nxt = argv[i + 1]

            # Zeit/Prozent (trim)
            if (
                tok in _NEG_TIME_FLAGS
                and not nxt.startswith("--")
                and _looks_like_neg_time_or_percent(nxt)
            ):
                out.append(f"{tok}={nxt}")
                i += 2
                continue

            # negative ints (offset-x / offset-y)
            if (
                tok in _NEG_INT_FLAGS
                and not nxt.startswith("--")
                and _looks_like_neg_int(nxt)
            ):
                out.append(f"{tok}={nxt}")
                i += 2
                continue

            # Paare (offset)
            if (
                tok in _NEG_PAIR_FLAGS
                and not nxt.startswith("--")
                and _looks_like_pair(nxt)
            ):
                out.append(f"{tok}={nxt}")
                i += 2
                continue

        # kein Merge → normal übernehmen
        out.append(tok)
        i += 1

    return out


# ========= Typ-Helfer =========

T = TypeVar("T")  # Dataclass-Typ


def _is_iterable_but_not_str(val: Any) -> bool:
    return isinstance(val, Iterable) and not isinstance(val, (str, bytes, Path))


# ========= Normalizer =========


def _normalize_files(val: Any) -> List[str]:
    """
    Normalisiert CLI-Argumente für Dateilisten zu List[str].
    Akzeptiert: None, str, Path, beliebige Iterables.
    """
    if val is None:
        return []
    if isinstance(val, (str, Path)):
        return [str(val)]
    if _is_iterable_but_not_str(val):
        it: Iterable[Any] = cast(Iterable[Any], val)
        return [str(x) for x in it]
    return [str(val)]


def _normalize_list(val: Any) -> Optional[List[str]]:
    """
    Normalisiert Werte, die optional eine Liste sind.
    None bleibt None, ansonsten Liste aus Strings.
    """
    if val is None:
        return None
    if isinstance(val, list):
        lst: List[object] = cast(List[object], val)
        return [str(item) for item in lst]
    if isinstance(val, (str, Path)):
        return [str(val)]
    if _is_iterable_but_not_str(val):
        it_obj: Iterable[object] = cast(Iterable[object], val)
        return [str(item) for item in it_obj]
    return [str(val)]


# ========= Namespace → Dataclass =========


def namespace_to_dataclass(
    dc_type: Type[T] | T,
    ns: argparse.Namespace,
) -> T:
    """
    Überführt einen argparse.Namespace in eine Dataclass-Instanz.

    - dc_type kann eine Dataclass-KLASSE oder bereits eine Instanz sein.
      In beiden Fällen wird auf Basis der Dataclass-Felder ein neues Objekt erzeugt.
    - Unbekannte Namespace-Attribute werden trotzdem als Attribute am Objekt angeheftet.
    - 'files' und 'tag' werden per Normalizer aufbereitet.
    """
    # 1) Klasse der Dataclass ermitteln (Pylance-freundlich)
    if isinstance(dc_type, type):
        dc_cls: Type[T] = cast(Type[T], dc_type)
    else:
        dc_cls = cast(Type[T], dc_type.__class__)

    if not dataclasses.is_dataclass(dc_cls):
        raise TypeError(f"{dc_type} ist keine Dataclass-Klasse/Instanz")

    # 2) Namespace in ein typisiertes Dict überführen
    ns_dict: Dict[str, Any] = vars(ns) if hasattr(ns, "__dict__") else {}

    # 3) Feldliste der Dataclass
    dc_fields: Dict[str, Field[Any]] = {f.name: f for f in dataclasses.fields(dc_cls)}

    # 4) Werte übernehmen (mit Spezialbehandlung für 'files' und 'tag')
    kwargs: Dict[str, Any] = {}
    for name, f in dc_fields.items():
        if name not in ns_dict:
            continue
        val: Any = ns_dict[name]
        if name == "files":
            kwargs[name] = _normalize_files(val)
        elif name == "tag":
            kwargs[name] = _normalize_list(val)
        else:
            kwargs[name] = val

    # 5) Instanz bauen
    obj: T = cast(T, dc_cls(**kwargs))

    # 6) Unbekannte Namespace-Attribute trotzdem anheften
    for name, val in ns_dict.items():
        if name in dc_fields:
            continue
        setattr(obj, name, val)

    # 7) praktische Quelle vormerken
    try:
        if getattr(obj, "files", None):
            setattr(obj, "_files_source", "cli")
        else:
            setattr(obj, "_files_source", None)
    except Exception:
        pass

    return obj


# ========= Defaults aus Dataclass + Args =========


def dict_with_defaults_from_args(cls: Type[T], args: Any) -> Dict[str, Any]:
    """
    Gibt ein dict mit allen Feldern der Dataclass zurück:
    - Wert vom CLI-Objekt (args), falls gesetzt/nicht None,
    - sonst Defaultwert aus der Dataclass.
    """
    if not is_dataclass(cls):
        raise ValueError("cls muss eine Dataclass-Klasse sein!")

    def get_default(field: Field[Any]) -> Any:
        if field.default is not MISSING:
            return field.default
        if field.default_factory is not MISSING:  # type: ignore[attr-defined]
            # field.default_factory ist ein Callable ohne Argumente
            return field.default_factory()  # type: ignore[misc,call-arg]
        return None

    defaults: Dict[str, Any] = {f.name: get_default(f) for f in fields(cls)}
    keys: Iterable[str] = defaults.keys()

    # Werte aus args (Namespace oder Dataclass-Instanz)
    if hasattr(args, "__dataclass_fields__"):
        args_dict: Dict[str, Any] = cast(Dict[str, Any], args.__dict__)
    else:
        args_dict = cast(Dict[str, Any], vars(args))

    result: Dict[str, Any] = {}
    for k in keys:
        val: Any = args_dict.get(k, None)
        result[k] = val if val is not None else defaults[k]
    return result


# ================ cli-helper =====================


def _parse_target_res_cli(v: str) -> str:
    v = v.strip()
    allowed = {"no-scale", "smallest", "average", "largest", "fixed"}
    if v in allowed:
        return v

    # erlaubt: fixed:WxH / fixed:W×H / fixed:W:H
    if v.lower().startswith("fixed:"):
        rest = v.split(":", 1)[1]
        if re.match(r"^\s*\d+\s*(?:x|×|:)\s*\d+\s*$", rest, flags=re.I):
            rest = re.sub(r"\s*", "", rest.replace("×", "x")).replace(":", "x")
            return f"fixed:{rest}"
        raise argparse.ArgumentTypeError(
            "Ungültige feste Größe. Beispiel: fixed:3000x3000"
        )

    # erlaubt: nur WxH / W×H / W:H → implizit fixed
    if re.match(r"^\s*\d+\s*(?:x|×|:)\s*\d+\s*$", v, flags=re.I):
        v = re.sub(r"\s*", "", v.replace("×", "x")).replace(":", "x")
        return f"fixed:{v}"

    raise argparse.ArgumentTypeError(
        "Erlaubt: no-scale|smallest|average|largest|fixed|fixed:WxH oder schlicht WxH (z.B. 3000x3000)"
    )


# ========= Parser-Aufbau =========


def create_parser() -> argparse.ArgumentParser:
    convert_preset_keys = list(getattr(defin, "CONVERT_PRESET", {}).keys())
    parser = argparse.ArgumentParser(
        description=_("cli_description"),
        usage="video <command> [<args>]",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--help", "-h", action="store_true", help=_("help_show_help"))

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False

    # --- convert ---
    convert_parser = subparsers.add_parser(
        "convert", description=_("help_convert"), help=_("help_convert")
    )
    convert_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    convert_parser.add_argument(
        "--format",
        "-f",
        choices=list(getattr(defin, "FORMATS", [])),
        default=None,
        help=_("help_format"),
    )
    convert_parser.add_argument(
        "--preset",
        "-p",
        choices=list(getattr(defin, "CONVERT_PRESET", [])),
        default=None,
        help=_("help_preset"),
    )
    convert_parser.add_argument("--resolution", "-r", help=_("help_resolution"))
    convert_parser.add_argument("--framerate", "-fr", help=_("help_framerate"))
    convert_parser.add_argument(
        "--codec",
        "-c",
        choices=list(getattr(defin, "VIDEO_CODECS", [])) + ["copy"],
        default=None,
        help=_("help_codec"),
    )
    convert_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- compress ---
    compress_parser = subparsers.add_parser(
        "compress", description=_("help_compress"), help=_("help_compress")
    )
    compress_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    compress_parser.add_argument("--quality", "-q", help=_("help_quality_percent"))
    compress_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- trim ---
    trim_parser = subparsers.add_parser(
        "trim", description=_("help_trim"), help=_("help_trim")
    )
    trim_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    trim_parser.add_argument("--start", "-s", help=_("help_trim_start"))
    trim_parser.add_argument("--duration", "-d", help=_("help_trim_duration"))
    trim_parser.add_argument("--end", "-e", help=_("help_trim_end"))
    trim_parser.add_argument(
        "--precise", "-p", action="store_true", help=_("help_trim_precise")
    )
    trim_parser.add_argument("--quality", "-q", help=_("help_quality_preset"))
    trim_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- scale ---
    scale_parser = subparsers.add_parser(
        "scale", description=_("help_scale"), help=_("help_scale")
    )
    scale_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    scale_parser.add_argument("--resolution", "-r", help=_("help_resolution"))
    scale_parser.add_argument(
        "--preserve-ar",
        "-ar",
        dest="preserve_ar",
        nargs="?",
        const=True,
        default=None,
        help=_("help_preserve_ar"),
    )
    scale_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- crop-pad ---
    croppad_parser = subparsers.add_parser(
        "croppad", description=_("help_croppad"), help=_("help_croppad")
    )
    croppad_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    croppad_parser.add_argument("--resolution", "-r", help=_("help_resolution"))
    croppad_parser.add_argument("--offset", "-O", help=_("help_offset"))
    croppad_parser.add_argument("--offset-x", "-x", type=int, help=_("help_offset_x"))
    croppad_parser.add_argument("--offset-y", "-y", type=int, help=_("help_offset_y"))
    croppad_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- interpolate ---
    interp_parser = subparsers.add_parser(
        "interpolate", description=_("help_interpolate"), help=_("help_interpolate")
    )
    interp_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    interp_parser.add_argument("--factor", "-f", help=_("help_factor_or_fps"))
    interp_parser.add_argument("--quality", "-q", help=_("help_interpolate_quality"))
    interp_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- images-to-video ---
    PRESETS_IMAGE2VIDEO = [p for p in convert_preset_keys if p != "lossless"]
    imgtovideo_parser = subparsers.add_parser(
        "img2vid", description=_("help_img2vid"), help=_("help_img2vid")
    )
    imgtovideo_parser.add_argument(
        "files", nargs="*", help=_("help_input_files_images")
    )
    imgtovideo_parser.add_argument(
        "--format",
        "-fm",
        choices=list(getattr(defin, "FORMATS", [])),
        help=_("help_format"),
    )
    imgtovideo_parser.add_argument(
        "--codec",
        "-c",
        choices=list(getattr(defin, "VIDEO_CODECS", [])),
        help=_("help_codec"),
    )
    imgtovideo_parser.add_argument(
        "--preset", "-p", choices=PRESETS_IMAGE2VIDEO, help=_("help_preset")
    )
    imgtovideo_parser.add_argument("--resolution", "-r", help=_("help_resolution"))
    imgtovideo_parser.add_argument(
        "--framerate", "-fr", help=_("help_framerate_or_duration")
    )
    imgtovideo_parser.add_argument("--duration", "-d", help=_("help_total_duration"))
    imgtovideo_parser.add_argument(
        "--scale", "-s", action="store_true", help=_("help_img_scale")
    )
    imgtovideo_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- merge ---
    merge_parser = subparsers.add_parser(
        "merge", description=_("help_merge"), help=_("help_merge")
    )
    merge_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    merge_parser.add_argument(
        "--target-res",
        "-tr",
        type=_parse_target_res_cli,
        default="no-scale",
        help=_("help_merge_target_res"),
    )
    merge_parser.add_argument("--offset", "-O", help=_("help_offset"))
    merge_parser.add_argument("--audio-offset", "-ao", help=_("help_audio_offset"))
    merge_parser.add_argument(
        "--subtitle-offset", "-so", help=_("help_subtitle_offset")
    )
    merge_parser.add_argument("--pause", "-pa", help=_("help_pause_between"))
    merge_parser.add_argument(
        "--burn-subtitle", "-bs", action="store_true", help=_("help_burn_subtitle")
    )
    merge_parser.add_argument("--subtitle-name", "-sn", help=_("help_subtitle_name"))
    merge_parser.add_argument("--audio-name", "-an", help=_("help_audio_name"))
    merge_parser.add_argument(
        "--extend", "-e", action="store_true", help=_("help_extend_canvas")
    )
    merge_parser.add_argument(
        "--format",
        "-f",
        choices=list(getattr(defin, "FORMATS", [])),
        help=_("help_format"),
    )
    merge_parser.add_argument(
        "--codec",
        "-c",
        choices=list(getattr(defin, "VIDEO_CODECS", [])),
        help=_("help_codec"),
    )
    merge_parser.add_argument(
        "--preset",
        "-pr",
        choices=list(getattr(defin, "CONVERT_PRESET", [])),
        help=_("help_preset"),
    )
    merge_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- extract ---
    extract_parser = subparsers.add_parser(
        "extract", description=_("help_extract"), help=_("help_extract")
    )
    extract_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    extract_parser.add_argument(
        "--frame", "-fr", nargs="?", const="", help=_("help_extract_frame")
    )
    extract_parser.add_argument(
        "--audio", "-a", action="store_true", help=_("help_extract_audio")
    )
    extract_parser.add_argument(
        "--video", "-v", action="store_true", help=_("help_extract_video")
    )
    extract_parser.add_argument(
        "--subtitle", "-s", nargs="?", const="", help=_("help_extract_subtitle")
    )
    extract_parser.add_argument(
        "--format",
        "-fm",
        help=_("help_extract_format"),
    )
    extract_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- enhance ---
    ENHANCE_PRESETS = [
        p for p in list(getattr(defin, "ENHANCE_PRESETS", [])) if p != "custom"
    ]
    enhance_parser = subparsers.add_parser(
        "enhance",
        description=_("help_enhance"),
        help=_("help_enhance"),
    )
    enhance_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    enhance_parser.add_argument(
        "--preset", "-p", choices=ENHANCE_PRESETS, help=_("help_enhance_preset")
    )
    enhance_parser.add_argument(
        "--stabilize", "-st", action="store_true", help=_("help_enhance_stabilize")
    )
    enhance_parser.add_argument(
        "--stab-method",
        "-stm",
        choices=["vidstab", "deshake"],
        help=_("help_enhance_stab_method"),
    )
    enhance_parser.add_argument(
        "--stab-smooth", "-sts", type=int, help=_("help_enhance_stab_smooth")
    )
    enhance_parser.add_argument(
        "--stab-rx", "-stx", type=int, help=_("help_enhance_stab_rx")
    )
    enhance_parser.add_argument(
        "--stab-ry", "-sty", type=int, help=_("help_enhance_stab_ry")
    )
    enhance_parser.add_argument(
        "--denoise", "-d", action="store_true", help=_("help_enhance_denoise")
    )
    enhance_parser.add_argument(
        "--denoise-method",
        "-dm",
        choices=["hqdn3d", "nlmeans"],
        help=_("help_enhance_denoise_method"),
    )
    enhance_parser.add_argument(
        "--denoise-intensity", "-di", type=int, help=_("help_enhance_denoise_intensity")
    )
    enhance_parser.add_argument(
        "--warmth", "-w", type=int, help=_("help_enhance_warmth")
    )
    enhance_parser.add_argument("--tint", "-t", type=int, help=_("help_enhance_tint"))
    enhance_parser.add_argument(
        "--brightness", "-b", type=int, help=_("help_enhance_brightness")
    )
    enhance_parser.add_argument(
        "--contrast", "-c", type=int, help=_("help_enhance_contrast")
    )
    enhance_parser.add_argument(
        "--saturation", "-s", type=int, help=_("help_enhance_saturation")
    )
    enhance_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- ai-enhance ---

    aienhance_parser = subparsers.add_parser(
        "aienhance", description=_("help_aienhance"), help=_("help_aienhance")
    )
    aienhance_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    aienhance_parser.add_argument(
        "--aimodel",
        "-am",
        choices=list(getattr(defin, "MODEL_META", [])),
        help=_("help_ai_model"),
    )
    aienhance_parser.add_argument(
        "--denoise", "-d", type=float, help=_("help_ai_denoise")
    )
    aienhance_parser.add_argument(
        "--noise-level", "-nl", type=int, help=_("help_ai_noise_level")
    )
    aienhance_parser.add_argument(
        "--face-enhance", "-fe", action="store_true", help=_("help_ai_face_enhance")
    )
    aienhance_parser.add_argument("--scale", "-s", type=float, help=_("help_ai_scale"))
    aienhance_parser.add_argument(
        "--tta", "-t", action="store_true", help=_("help_ai_tta")
    )
    aienhance_parser.add_argument(
        "--blend", "-b", action="store_true", help=_("help_ai_blend")
    )
    aienhance_parser.add_argument(
        "--blend-opacity", "-bo", type=float, help=_("help_ai_blend_opacity")
    )
    aienhance_parser.add_argument(
        "--priority",
        "-p",
        choices=["auto", "max", "medium", "minimal", "no_parallelisation"],
        help=_("help_ai_priority"),
    )
    aienhance_parser.add_argument(
        "--force-overwrite",
        "-fo",
        action="store_true",
        help=_("help_ai_force_overwrite"),
    )
    aienhance_parser.add_argument("--chunk", "-ch", type=int, help=_("help_ai_chunk"))
    aienhance_parser.add_argument(
        "--output", "-o", type=str, help=_("help_output_path")
    )

    # --- gif ---
    gif_parser = subparsers.add_parser(
        "gif", description=_("help_gif"), help=_("help_gif")
    )
    gif_parser.add_argument("files", nargs="*", help=_("help_input_files"))
    gif_parser.add_argument("--text-top", "-tt", help=_("help_gif_text_top"))
    gif_parser.add_argument("--text-bottom", "-tb", help=_("help_gif_text_bottom"))
    gif_parser.add_argument(
        "--font-size", "-fs", type=str, help=_("help_gif_font_size")
    )
    gif_parser.add_argument(
        "--fs-top", "-ft", type=str, help=_("help_gif_font_size_top")
    )
    gif_parser.add_argument(
        "--fs-bottom", "-fb", type=str, help=_("help_gif_font_size_bottom")
    )
    gif_parser.add_argument(
        "--no-auto-open",
        "-na",
        dest="no_auto_open",
        action="store_true",
        help=_("help_gif_no_auto_open"),
    )
    gif_parser.add_argument(
        "--keep-quality", "-q", action="store_true", help=_("help_gif_keep_quality")
    )
    gif_parser.add_argument("--output", "-o", help=_("help_output_path"))

    # --- metadata ---
    meta_parser = subparsers.add_parser(
        "metadata", description=_("help_metadata"), help=_("help_metadata")
    )
    meta_parser.add_argument("files", nargs="*", help=_("help_input_files"))

    # Listen
    meta_parser.add_argument(
        "--list-tags", action="store_true", help=_("help_meta_list_tags")
    )
    meta_parser.add_argument(
        "--list-tags-json", action="store_true", help=_("help_meta_list_tags_json")
    )
    meta_parser.add_argument(
        "--list-tagnames", action="store_true", help=_("help_meta_list_tagnames")
    )
    meta_parser.add_argument("--all", action="store_true", help=_("help_meta_all"))

    meta_parser.add_argument(
        "--delete-thumbnail", action="store_true", help=_("help_meta_delete_thumbnail")
    )
    meta_parser.add_argument("--set-thumbnail", help=_("help_meta_set_thumbnail"))
    meta_parser.add_argument(
        "--show-thumbnail", action="store_true", help=_("help_meta_show_thumbnail")
    )

    # Generisches --tag: KEY (lesen)  |  KEY=VALUE (setzen). Mehrfach nutzbar.
    meta_parser.add_argument(
        "--tag",
        action="append",
        help=_("help_meta_tag_generic"),
    )

    # Für alle EDITABLE_KEYS dedizierte Flags erzeugen:
    for tag in list(getattr(defin, "EDITABLE_META_KEYS", [])):
        # --title              -> lesen (wenn ohne Wert)  | setzen (wenn Wert angegeben)
        meta_parser.add_argument(
            f"--{tag}",
            nargs="?",
            const="__READ__",
            help=_("help_meta_read_or_set").format(tag=tag),
        )
        # --list-tag-title     -> lesen
        meta_parser.add_argument(
            f"--list-tag-{tag}",
            action="store_true",
            help=_("help_meta_list_tag").format(tag=tag),
        )
        # --set-tag-title v    -> setzen (Alias)
        meta_parser.add_argument(
            f"--set-tag-{tag}", help=_("help_meta_set_tag").format(tag=tag)
        )
        # --delete-tag-title   -> löschen
        meta_parser.add_argument(
            f"--delete-tag-{tag}",
            action="store_true",
            help=_("help_meta_delete_tag").format(tag=tag),
        )

    for tag in list(getattr(defin, "PROTECTED_META_KEYS", [])):
        meta_parser.add_argument(
            f"--list-tag-{tag}",
            action="store_true",
            help=_("help_meta_list_tag").format(tag=tag),
        )

    for tag in list(getattr(defin, "VIRTUAL_META_INFO", [])):
        meta_parser.add_argument(
            f"--list-tag-{tag}",
            action="store_true",
            help=_("help_meta_list_tag").format(tag=tag),
        )

    _attach_completers(parser)
    return parser


def _attach_completers(parser: argparse.ArgumentParser) -> None:
    """
    Hängt Dateipfad-/Verzeichnis-Completer an gängige Positionals/Optionen.
    Läuft no-op, wenn argcomplete nicht installiert ist.
    """
    if FilesCompleter is None:
        return

    # Subparser mappen
    subparsers_action = next(
        (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None
    )  # type: ignore[attr-defined]
    if not subparsers_action:
        return

    # Helper: FilesCompleter für beliebige Argumente per Name setzen
    def _fc(p: argparse.ArgumentParser, *arg_names: str, dirs: bool = False) -> None:
        # wenn keine Completer importiert werden konnten → raus
        if FilesCompleter is None and DirectoriesCompleter is None:
            return

        comp = None
        if dirs and DirectoriesCompleter is not None:
            try:
                comp = DirectoriesCompleter()
            except Exception:
                comp = None

        if comp is None:
            if FilesCompleter is None:
                return
            try:
                comp = FilesCompleter()
            except Exception:
                return

        for a in getattr(p, "_actions", []):  # type: ignore[attr-defined]
            # Optionen
            for n in getattr(a, "option_strings", []):
                if n in arg_names:
                    try:
                        a.completer = comp  # type: ignore[attr-defined]
                    except Exception:
                        pass
            # Positionals
            if (
                not getattr(a, "option_strings", None)
                and getattr(a, "dest", None) in arg_names
            ):
                try:
                    a.completer = comp  # type: ignore[attr-defined]
                except Exception:
                    pass

    # Für jeden Subcommand-Parser spezifische Completer setzen
    for name, sp in subparsers_action.choices.items():  # type: ignore[attr-defined]
        # Standard: 'files' Positionals → Datei-Completion
        _fc(sp, "files")

        # Häufige Optionen
        _fc(sp, "--input", "-i")  # Dateien
        _fc(sp, "--in", "--source", "--file")
        _fc(sp, "--mask", "--script")  # falls Pfad
        _fc(sp, "--output", "-o", dirs=True)  # Ziel meist Verzeichnis/Datei

        # Spezialfälle je Subcommand
        if name in {"extract", "images-to-video"}:
            _fc(sp, "--format")  # bleibt Wort-Completion (Choices handled by argparse)

        if name in {"record"}:
            _fc(sp, "--output", "-o", dirs=True)
