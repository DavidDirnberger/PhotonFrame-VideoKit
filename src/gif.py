#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, cast

import consoleOutput as co
import definitions as defin
import fileSystem as fs
import helpers as he
import process_wrappers as pw
import userInteraction as ui
from i18n import _, tr

BASE_FONT_WIDTH = 640


def _list_str_factory() -> List[str]:
    """Explizit getypte Factory für List[str], damit Pylance kein list[Unknown] annimmt."""
    return []


@dataclass
class GifArgs:
    files: List[str] = field(default_factory=_list_str_factory)
    text_top: Optional[str] = None
    text_bottom: Optional[str] = None
    font_size: Optional[str] = None
    fs_top: Optional[str] = None
    fs_bottom: Optional[str] = None
    no_auto_open: Optional[bool] = None
    keep_quality: Optional[bool] = None
    output: Optional[str] = None


def _find_impact_fontfile() -> tuple[Optional[str], bool]:
    """Versucht Impact systemweit oder in unseren Installationspfaden zu finden."""
    if shutil.which("fc-match"):
        try:
            r = subprocess.run(
                ["fc-match", "-f", "%{file}\n", "Impact"],
                capture_output=True,
                text=True,
            )
            cand = (r.stdout or "").strip()
            if r.returncode == 0 and cand and Path(cand).exists():
                return cand, True
        except Exception:
            pass

    for candidate in _impact_font_candidates():
        if candidate.exists():
            return str(candidate), True

    return None, False


def _impact_font_candidates() -> List[Path]:
    """Bekannte Orte für impact.ttf (Installer, Env-Overrides, System)."""
    home = Path.home()
    candidates: List[Path] = []

    env_font = os.environ.get("VIDEOMANAGER_IMPACT_FONT")
    if env_font:
        candidates.append(Path(env_font).expanduser())

    env_dir = os.environ.get("VIDEOMANAGER_FONT_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / "impact.ttf")

    vm_base = os.environ.get("VM_BASE")
    if vm_base:
        candidates.append(Path(vm_base) / "assets" / "impact.ttf")

    xdg_data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    candidates.append(xdg_data / "videoManager" / "fonts" / "impact.ttf")
    candidates.append(
        home / ".local" / "share" / "videoManager" / "fonts" / "impact.ttf"
    )
    candidates.append(home / ".local" / "share" / "fonts" / "impact.ttf")

    if sys.platform == "darwin":
        candidates.append(
            home
            / "Library"
            / "Application Support"
            / "videoManager"
            / "fonts"
            / "impact.ttf"
        )
        candidates.append(home / "Library" / "Fonts" / "Impact.ttf")

    if os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        candidates.append(local_app / "videoManager" / "fonts" / "impact.ttf")
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        candidates.append(windir / "Fonts" / "impact.ttf")

    candidates.append(Path(__file__).with_name("impact.ttf"))

    uniq: List[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        resolved = cand.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            uniq.append(resolved)
    return uniq


def _is_int_literal(val: str) -> bool:
    if not val:
        return False
    s = val.strip()
    if s.startswith(("+", "-")):
        s = s[1:]
    return s.isdigit()


def _normalize_cli_font_arg(flag: str, raw: Optional[str]) -> Optional[str]:
    """
    Validiert CLI-Werte (font_size/fs_top/fs_bottom).
    Erlaubt: ganze Zahlen >=1 oder Schlüssel aus defin.MEME_FONTSIZE.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if _is_int_literal(s):
        try:
            intval = int(s, 10)
        except ValueError:
            intval = None
        if intval is not None and intval >= 1:
            return str(intval)
    key = s.lower()
    if key in defin.MEME_FONTSIZE:
        return key
    allowed = ", ".join(defin.MEME_FONTSIZE.keys())
    msg = tr(
        {
            "de": f"Ungültiger Schriftgrößenwert '{s}' für {flag}. Erlaubt sind ganze Zahlen (>=8) oder: {allowed}",
            "en": f"Invalid font size '{s}' for {flag}. Allowed values are integers (>=8) or: {allowed}",
        }
    )
    co.print_error(msg)
    raise ValueError(f"invalid font size for {flag}: {s}")


def _resolve_fontsize(value: Optional[str], default_label: str = "medium") -> int:
    if value is None or str(value).strip() == "":
        label = default_label
    else:
        s = str(value).strip().lower()
        try:
            return max(8, int(float(s)))
        except ValueError:
            label = s

    try:
        return int(defin.MEME_FONTSIZE[label]["size"])
    except Exception:
        return 25


def _build_pre_chain(
    file: str | Path,
    max_width: int = 640,
    fps_limit: int = 30,
) -> tuple[Optional[str], int, int]:
    r"""
    Liefert (pre_chain, source_width, source_height). pre_chain ist z.B.
    "fps=30,scale=min(iw\,640):-2:flags=lanczos" oder None, wenn nichts zu tun
    ist.
    """
    p = Path(file)
    if not p.exists():
        return None, 0, 0

    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                str(p),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data: dict[str, Any] = json.loads(probe.stdout or "{}")
        streams: List[dict[str, Any]] = cast(
            List[dict[str, Any]], data.get("streams") or []
        )
        st: dict[str, Any] = streams[0] if streams else {}
        width = int(st.get("width") or 0)
        height = int(st.get("height") or 0)
        r_str: str = str(st.get("r_frame_rate") or "0/0")
        r_parts = r_str.split("/")
        fps = (
            (float(r_parts[0]) / float(r_parts[1]))
            if (len(r_parts) == 2 and r_parts[1] != "0")
            else 0.0
        )
    except Exception:
        return None, 0, 0

    parts: List[str] = []
    if fps > float(fps_limit):
        parts.append(f"fps={int(fps_limit)}")

    if width > max_width:
        parts.append(f"scale=min(iw\\,{int(max_width)}):-2:flags=lanczos")

    return (",".join(parts) if parts else None), width, height


@dataclass
class _GifJob:
    path: Path
    pre_chain: Optional[str]
    src_width: int
    src_height: int
    target_resolution: Optional[tuple[int, int]] = None


def _estimate_target_resolution(
    src_w: int, src_h: int, *, keep_quality: bool, max_width: int
) -> Optional[tuple[int, int]]:
    """Berechne die erwartete Ausgabeauflösung gem. Scaling-Regeln."""
    if src_w <= 0 or src_h <= 0:
        return None
    if keep_quality or src_w <= max_width:
        return (src_w, src_h)
    target_w = min(src_w, max_width)
    target_h_float = (src_h * target_w) / float(src_w)
    # ffmpeg rundet bei -2 auf gerade Zahlen
    target_h = int(round(target_h_float / 2.0) * 2) or 2
    return (target_w, target_h)


def _format_resolution_entry(
    path: Path, src_w: int, src_h: int, target: Optional[tuple[int, int]]
) -> str:
    """String für die Parameter-Tabelle (inkl. Quelle->Ziel)."""
    if src_w <= 0 or src_h <= 0 or target is None:
        return f"{path.name}: " + tr({"de": "unbekannt", "en": "unknown"})
    tgt_w, tgt_h = target
    if tgt_w == src_w and tgt_h == src_h:
        return f"{src_w}x{src_h}"
    return f"{src_w}x{src_h} -> {tgt_w}x{tgt_h}"


def _scale_font(px: int, factor: float) -> int:
    if factor <= 1.0:
        return px
    return max(8, int(round(px * factor)))


def _clean_cli_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def gif(args: Any) -> None:
    files_any = fs.prepare_files(args, defin.VIDEO_EXTENSIONS)
    files: List[str] = [str(x) for x in (files_any or [])]
    if not files:
        return

    BATCH_MODE: bool = getattr(args, "_files_source", None) == "cli"

    fontfile, have_file = _find_impact_fontfile()
    use_font_expr = f"fontfile='{fontfile}'" if have_file else "font=Impact"
    if not have_file:
        co.print_warning(
            tr(
                {
                    "de": (
                        "Impact.ttf nicht gefunden. Fuehre install.sh aus, akzeptiere die Microsoft-Core-Fonts-EULA "
                        "oder setze VIDEOMANAGER_IMPACT_FONT auf den lokalen impact.ttf Pfad."
                    ),
                    "en": (
                        "Impact.ttf not found. Run install.sh, accept the Microsoft Core Fonts EULA, "
                        "or set VIDEOMANAGER_IMPACT_FONT to your local impact.ttf path."
                    ),
                }
            )
        )

    text_top: Optional[str] = cast(Optional[str], getattr(args, "text_top", None))
    text_bottom: Optional[str] = cast(Optional[str], getattr(args, "text_bottom", None))

    raw_font_size = _clean_cli_value(
        cast(Optional[str], getattr(args, "font_size", None))
    )
    raw_fs_top = _clean_cli_value(cast(Optional[str], getattr(args, "fs_top", None)))
    raw_fs_bottom = _clean_cli_value(
        cast(Optional[str], getattr(args, "fs_bottom", None))
    )

    try:
        cli_font_size = _normalize_cli_font_arg("--font-size", raw_font_size)
        cli_fs_top = _normalize_cli_font_arg("--fs-top", raw_fs_top)
        cli_fs_bottom = _normalize_cli_font_arg("--fs-bottom", raw_fs_bottom)
    except ValueError:
        return

    fs_global: str = (
        cli_font_size if cli_font_size is not None else (raw_font_size or "medium")
    )
    fs_top: Optional[str] = cli_fs_top if cli_fs_top is not None else raw_fs_top
    fs_bottom: Optional[str] = (
        cli_fs_bottom if cli_fs_bottom is not None else raw_fs_bottom
    )
    keep_quality_flag: Optional[bool] = cast(
        Optional[bool], getattr(args, "keep_quality", None)
    )

    if not BATCH_MODE:
        if keep_quality_flag is None:
            keep_quality_flag = not ui.ask_yes_no(
                _("convert_to_gif_resolution"), default=True, back_option=False
            )

        if text_top is None and text_bottom is None:
            co.print_info(_("gif_press_enter_to_skip_text"))
            text_top = input(
                co.return_promt("  " + _("gif_text_prompt_top") + ": ")
            ).strip()
            text_bottom = input(
                co.return_promt("  " + _("gif_text_prompt_bottom") + ": ")
            ).strip()

        meme_keys = list(defin.MEME_FONTSIZE)
        shown = [tr(defin.MEME_FONTSIZE[k]["name"]) for k in meme_keys]

        if text_top and fs_top is None:
            idx = ui.ask_user(
                _("gif_choose_fs_top"), meme_keys, display_labels=shown, default=2
            )
            fs_top = idx if idx is not None else "medium"

        if text_bottom and fs_bottom is None:
            idx = ui.ask_user(
                _("gif_choose_fs_bottom"), meme_keys, display_labels=shown, default=2
            )
            fs_bottom = idx if idx is not None else "medium"
    else:
        if keep_quality_flag is None:
            keep_quality_flag = False

    keep_quality = bool(keep_quality_flag)
    no_auto_open_flag = bool(getattr(args, "no_auto_open", False))

    # Auflösungen pro Datei vorab ermitteln (für Tabelle und Font-Scaling)
    jobs: List[_GifJob] = []
    res_entries: List[str] = []
    for file in files:
        path = Path(file)
        pre_chain, src_width, src_height = _build_pre_chain(
            path, max_width=480, fps_limit=20
        )
        target_res = _estimate_target_resolution(
            src_width, src_height, keep_quality=keep_quality, max_width=480
        )
        jobs.append(
            _GifJob(
                path=path,
                pre_chain=pre_chain,
                src_width=src_width,
                src_height=src_height,
                target_resolution=target_res,
            )
        )
        res_entries.append(
            _format_resolution_entry(path, src_width, src_height, target_res)
        )

    # Parametertabelle (nur gesetzte Felder)
    def _add_param(key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        params_table[key] = value

    font_size_was_set = (cli_font_size is not None) or (raw_font_size is not None)
    params_table: dict[str, Any] = {"files": files}
    _add_param("text_top", text_top)
    _add_param("text_bottom", text_bottom)
    if font_size_was_set:
        _add_param("font_size", fs_global)
    _add_param("fs_top", fs_top if text_top else None)
    _add_param("fs_bottom", fs_bottom if text_bottom else None)
    params_table["keep_quality"] = _("yes") if keep_quality else _("no")
    params_table["no_auto_open"] = _("yes") if no_auto_open_flag else _("no")
    if res_entries:
        params_table["result_resolution"] = res_entries

    labels = {
        "text_top": {"de": "Text oben", "en": "Text top"},
        "text_bottom": {"de": "Text unten", "en": "Text bottom"},
        "font_size": {"de": "Schriftgröße", "en": "Font size"},
        "fs_top": {"de": "Schriftgröße oben", "en": "Font size top"},
        "fs_bottom": {"de": "Schriftgröße unten", "en": "Font size bottom"},
        "keep_quality": {
            "de": "Originalauflösung behalten",
            "en": "Keep original resolution",
        },
        "no_auto_open": {"de": "Nicht automatisch öffnen", "en": "Do not auto-open"},
        "result_resolution": {
            "de": "resultierende Auflösung",
            "en": "Resulting resolution",
        },
    }

    co.print_selected_params_table(params_table, labels=labels)

    size_top = _resolve_fontsize(
        fs_top if fs_top is not None else fs_global, default_label="medium"
    )
    size_bottom = _resolve_fontsize(
        fs_bottom if fs_bottom is not None else fs_global, default_label="medium"
    )

    outline = "bordercolor=black:borderw=2"
    fontcolor = "white"

    total_jobs = len(jobs) if jobs else len(files)

    for i, job in enumerate(jobs, start=1):
        path = job.path
        out = fs.build_output_path(
            path,
            output_arg=getattr(args, "output", None),
            default_suffix="_meme",
            idx=i,
            total=total_jobs,
            target_ext=".gif",
        )

        pre_chain = job.pre_chain
        src_width = job.src_width

        scale_factor = 1.0
        if keep_quality and src_width > BASE_FONT_WIDTH:
            scale_factor = max(1.0, float(src_width) / float(BASE_FONT_WIDTH))

        eff_size_top = _scale_font(size_top, scale_factor)
        eff_size_bottom = _scale_font(size_bottom, scale_factor)

        draw: List[str] = []
        if text_top:
            safe = he.escape_drawtext(text_top)
            draw.append(
                f"drawtext={use_font_expr}:text='{safe}':"
                f"fontcolor={fontcolor}:fontsize={eff_size_top}:{outline}:"
                f"x=(w-text_w)/2:y=10"
            )
        if text_bottom:
            safe = he.escape_drawtext(text_bottom)
            draw.append(
                f"drawtext={use_font_expr}:text='{safe}':"
                f"fontcolor={fontcolor}:fontsize={eff_size_bottom}:{outline}:"
                f"x=(w-text_w)/2:y=h-th-10"
            )

        parts: List[str] = []
        if pre_chain and not keep_quality:
            parts.append(pre_chain)
        parts += draw
        pre_combined = ",".join(parts) if parts else "null"

        filter_complex = (
            f"[0:v]{pre_combined},split=2[pre_a][pre_b];"
            f"[pre_a]palettegen=stats_mode=diff[p];"
            f"[pre_b][p]paletteuse=dither=sierra2_4a"
        )

        cmd: List[str] = [
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
            "-filter_complex",
            filter_complex,
            "-loop",
            "0",
            str(out),
        ]

        pw.run_ffmpeg_with_progress(
            path.name,
            cmd,
            _("gif_creating_progress"),
            _("gif_created_done"),
            output_file=out,
            BATCH_MODE=BATCH_MODE,
        )

        if BATCH_MODE and not no_auto_open_flag:
            fs.open_file_crossplatform(out)

        co.print_finished(_("gif_method"))
