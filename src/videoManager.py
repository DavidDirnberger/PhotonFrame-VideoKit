#!/usr/bin/env python3
from __future__ import annotations

# PYTHON_ARGCOMPLETE_OK
import os
import sys
from pathlib import Path
from typing import (
    Any,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    runtime_checkable,
)

try:
    import argcomplete as _argcomplete  # type: ignore[import-not-found]
except Exception:
    _argcomplete = None  # type: ignore[assignment]

# local modules
import compress as comp
import consoleOutput as co
import convert as cv
import croppad as cp
import definitions as defin
import enhance as eh
import extract as ex
import gif as gf
import imagesToVideo as itv
import interpolate as inter
import mem_guard as mg
import merge as me
import metadata as md
import scale as sc
import trim as tr
from cli_parser import create_parser, namespace_to_dataclass, normalize_cli_argv
from configManager import ConfigManager
from i18n import _, set_lang
from loghandler import start_log
from mem_guard import enable_escape_cancel, install_global_cancel_handlers
from process_wrappers import FFmpegFailed

try:
    import aienhance as ae

    _AI_IMPORT_ERROR: Exception | None = None
except Exception as e:  # AI stack optional
    ae = None  # type: ignore[assignment]
    _AI_IMPORT_ERROR = e


@runtime_checkable
class CommandFunc(Protocol):
    """Callable protocol for command functions that accept a single dataclass-like args object."""

    def __call__(self, args: Any) -> None: ...


# (command, description, icon, function, ArgsClass)
OptionEntry = Tuple[str, str, str, CommandFunc, Type[Any]]
OptionSpec = Tuple[str, str, str, CommandFunc, Type[Any]]  # desc-key variant

_OPTION_SPECS: List[OptionSpec] = [
    ("convert", "desc_convert", "🔁", cv.convert, cv.ConvertArgs),
    ("compress", "desc_compress", "🗜️", comp.compress, comp.CompressArgs),
    ("trim", "desc_trim", "✂️", tr.trim, tr.TrimArgs),
    ("scale", "desc_scale", "📐", sc.scale_video, sc.ScaleArgs),
    ("croppad", "desc_crop_pad", "🔲", cp.crop_pad, cp.CropPadArgs),
    ("interpolate", "desc_interpolate", "🌀", inter.interpolate, inter.InterpolateArgs),
    (
        "img2vid",
        "desc_images_to_video",
        "🖼️➜🎞️",
        itv.images_to_video,
        itv.ImagesToVideoArgs,
    ),
    ("merge", "desc_merge", "🔗", me.merge, me.MergeArgs),
    ("extract", "desc_extract", "🧲", ex.extract, ex.ExtractArgs),
    ("enhance", "desc_enhance", "✨", eh.enhance, eh.EnhanceArgs),
    ("gif", "desc_gif", "🖼️", gf.gif, gf.GifArgs),
    ("metadata", "desc_metadata", "🏷️", md.metadata, md.MetadataArgs),
]

if ae is not None:
    _OPTION_SPECS.insert(
        10, ("aienhance", "desc_ai_enhance", "🪄", ae.ai_enhance, ae.AIEnhanceArgs)
    )


def _options_for(ai_enabled: bool) -> List[OptionEntry]:
    """Return menu/dispatch options respecting AI availability and config."""
    opts: List[OptionEntry] = []
    for cmd, desc_key, icon, func, args in _OPTION_SPECS:
        if cmd == "aienhance" and (not ai_enabled or ae is None):
            continue
        opts.append((cmd, _(desc_key), icon, func, args))
    return opts


def interactive_menu(opts: List[OptionEntry]) -> None:
    """Render interactive menu and dispatch the selected command."""
    co.print_package_headline()

    items: List[Tuple[str, str]] = [
        (symb, cmd) for (cmd, _desc, symb, _func, _args) in opts
    ]
    descriptions: List[str] = [desc for (_cmd, desc, _symb, _func, _args) in opts]

    co.print_list(
        items,
        descriptions=descriptions,
        seperator="-",
        comment_col=32,
    )
    co.print_exit()

    choice = input(co.return_promt(_("select_action_prompt") + " ")).strip()
    if choice == "0":
        sys.exit(0)

    try:
        _t1, _t2, _t3, func, dargs = opts[int(choice) - 1]
    except (ValueError, IndexError):
        co.print_fail(_("invalid_selection"))
        sys.exit(1)

    # Instantiate default args for interactive mode and run
    spec_args = dargs()
    func(spec_args)


def bootstrap(project_root: Path | None = None) -> ConfigManager:
    if project_root is None:
        # 1. Versuch: vom Launcher gesetztes VM_BASE verwenden
        vm_base = os.environ.get("VM_BASE")
        if vm_base:
            project_root = Path(vm_base).resolve()
        else:
            # 2. Fallback: von src/videoManager.py ein Verzeichnis hoch
            project_root = Path(__file__).resolve().parents[1]

    cm = ConfigManager(project_root=project_root)
    cm.load()

    lang = cm.get_str("general", "language") or "en"
    lang = lang.strip().lower().replace("_", "-").split("-")[0]
    set_lang(lang)
    return cm


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point. When *argv* is None, sys.argv[1:] is used."""
    # Ensure UTF-8 stdout on Windows consoles
    if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    if os.environ.get("_ARGCOMPLETE"):
        parser = create_parser()
        try:
            if _argcomplete is not None:
                _argcomplete.autocomplete(parser, always_complete_options=True)
        finally:
            sys.exit(0)

    cm = bootstrap()
    ai_enabled_cfg = cm.get_bool("ai", "enabled", fallback=True)
    ai_available = bool(ai_enabled_cfg)
    options = _options_for(ai_available)

    install_global_cancel_handlers()
    enable_escape_cancel()
    mg.guard_path(defin.TMP_DIR, recursive=True)

    parser = create_parser()

    try:
        if _argcomplete is not None:
            _argcomplete.autocomplete(parser, always_complete_options=True)
    except Exception:
        pass

    argv_local: List[str] = list(sys.argv[1:] if argv is None else argv)

    # Lightweight help passthrough for top-level and subcommands
    if argv_local:
        if argv_local[0] in {"--help", "-h"}:
            co.show_info()
            return
        if argv_local[0] in {"--version", "-v"}:
            if defin.__version__ is None:
                _vm_version = "unknown"
            else:
                _vm_version = defin.__version__
            co.print_info(f"PhotonFrame - VideoKit { _vm_version }")
            return
        elif len(argv_local) > 1 and argv_local[1] in {"--help", "-h"}:
            co.show_info(argv_local[0])

    # ZENTRALE, generische Vorverarbeitung (einmal für alle Subcommands)
    argv_local = normalize_cli_argv(["video"] + argv_local)[1:]

    args, unknown = parser.parse_known_args(args=argv_local)

    if unknown:
        co.print_error(f"Unknown argument(s): {' '.join(unknown)}")
        sys.exit(1)

    try:
        if not getattr(args, "command", None):
            start_log()
            interactive_menu(options)
            return

        if args.command == "aienhance":
            if _AI_IMPORT_ERROR is not None:
                co.print_error(_("ai_feature_not_installed"))
                sys.exit(1)
            if not ai_available:
                co.print_error(_("ai_feature_disabled"))
                sys.exit(1)

        # Dispatch to the selected command
        for cmd, _desc, _symb, func, refarg in options:
            if args.command == cmd:
                spec_args = namespace_to_dataclass(refarg, args)
                func(spec_args)
                break
        else:
            co.show_info()

    except FFmpegFailed:
        sys.exit(1)
    except KeyboardInterrupt:
        # Sauberer Abbruch ohne Traceback
        try:
            mg.CANCEL.set()
        except Exception:
            pass
        try:
            mg.kill_all()
        except Exception:
            pass
        co.print_info(_("canceled_info"))
        sys.exit(130)  # üblicher Exit-Code für SIGINT


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            mg.CANCEL.set()
        except Exception:
            pass
        try:
            mg.kill_all()
        except Exception:
            pass
        co.print_info(_("canceled_info"))
        sys.exit(130)
