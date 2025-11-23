#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Mapping as AbcMapping
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TextIO,
    TypeAlias,
    cast,
)

# Drittanbieter: wcwidth – getypte Wrapper statt direkter Alias-Variablen
import wcwidth as _wcmod
import definitions as defin
import helpers as he  # noqa: F401
import i18n as _i18n
from ANSIColor import ANSIColor


def _WCSWIDTH(s: str) -> int:
    return cast(int, _wcmod.wcswidth(s))


def _WCWIDTH(ch: str) -> int:
    return cast(int, _wcmod.wcwidth(ch))


_ = cast(Callable[[str], str], _i18n._)
tr = cast(Callable[[Any], str], _i18n.tr)
get_lang = cast(Callable[[], str], _i18n.get_lang)

STARTED = False
ENDED = False

# ─────────────────────────────────────────────────────────────────────────────
# Typen / Protokolle
# ─────────────────────────────────────────────────────────────────────────────

SpecLeaf: TypeAlias = list[str]
SpecMap: TypeAlias = Mapping[str, "SpecNode"]  # parent_key -> SpecNode
SpecNode: TypeAlias = Mapping[
    Any, SpecLeaf | SpecMap
]  # parent_value -> (Leaf | tiefer)

# Styles können String, Sequenz von Strings oder None sein
StyleType: TypeAlias = str | Sequence[str] | None


# ANSI-Farbprovider als Protocol
class _AnsiProvider(Protocol):
    def get(self, color: str, background: bool = False) -> str: ...
    def combine(self, *names: str) -> str: ...


c: _AnsiProvider = ANSIColor()

# Standardmäßig übersprungene Color-Defaults (50 = neutral)
_COLOR_PARAMS_DEFAULT_SKIP: set[str] = {
    "warmth",
    "tint",
    "brightness",
    "contrast",
    "saturation",
}

# ANSI Regex & Helfer
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ─────────────────────────────────────────────────────────────────────────────
# String-/Terminal-Helfer
# ─────────────────────────────────────────────────────────────────────────────


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")


def visible_width(s: str) -> int:
    """Sichtbare Breite eines Strings (ANSI entfernt, Unicode-NFC, wcwidth-basiert)."""
    s = unicodedata.normalize("NFC", strip_ansi(str(s)))
    w = _WCSWIDTH(s)
    if w < 0:  # unprintables → Zeichenweise summieren
        w = sum(max(0, _WCWIDTH(ch)) for ch in s)
    return w


def wrap_vis(text: str, width: int) -> list[str]:
    """Weicher Umbruch nach sichtbarer Breite (ohne harte Wortabbrüche)."""
    parts = re.split(r"(\s+)", text)
    lines: list[str] = []
    cur = ""
    curw = 0
    for tok in parts:
        tw = visible_width(tok)
        if cur and curw + tw > width:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
            curw = visible_width(cur)
        else:
            cur += tok
            curw += tw
    if cur:
        lines.append(cur.rstrip())
    return lines


def _ansi_len(s: str) -> int:
    return len(strip_ansi(s))


def _is_langmap(v: Any) -> bool:
    # Erkenne {"de": "...", "en": "..."} (auch wenn nur eine Sprache drinsteht)
    return isinstance(v, dict) and set(v.keys()).issubset({"de", "en"}) and len(v) > 0


def _label_for(key: str, labels: Optional[Dict[str, Any]]) -> str:
    # 1) labels-Override (str ODER {"de","en"})
    # 2) i18n._(key)
    # 3) key selbst (Fallback)
    if labels and key in labels:
        spec = labels[key]
        return tr(spec) if _is_langmap(spec) else str(spec)
    lbl = _(key)
    return lbl if lbl != key else key


def _fmt_value(value: Any) -> Iterable[str]:
    # Sprach-Dicts
    if _is_langmap(value):
        return [tr(value)]
    # Sequenzen (aber keine Strings) → zeilenweise, Elemente ggf. tr()
    if isinstance(value, (list, tuple, set)) and not isinstance(value, (str, bytes)):
        out: list[str] = []
        for x in cast(Iterable[Any], value):
            out.append(tr(x) if _is_langmap(x) else str(x))
        return out or ["—"]
    # Dict → "key: value"
    if isinstance(value, dict):
        lines: list[str] = []
        # WICHTIG: Items als Iterable[tuple[Any, Any]] casten, damit k/v nicht Unknown sind
        for k_any, v_any in cast(Iterable[tuple[Any, Any]], value.items()):
            vv = tr(v_any) if _is_langmap(v_any) else str(v_any)
            kk = _(str(k_any))
            k_str = str(k_any)
            lines.append(f"{kk if kk != k_str else k_str}: {vv}")
        return lines or ["—"]
    # Single-Line
    return [str(value)]


def _keys_str(d: Mapping[Any, Any]) -> set[str]:
    """Hilfsfunktion: Keys als set[str]."""
    return {str(k) for k in d.keys()}


def count_leading_indent_spaces(s: str, tabsize: int = 4) -> int:
    # ersetzt Tabs durch tabsize Spaces und zählt dann Spaces am Anfang
    expanded = s.expandtabs(tabsize)
    return len(expanded) - len(expanded.lstrip(" "))


# ─────────────────────────────────────────────────────────────────────────────
# Dynamische Banner (einzeilig, 3er Außenabstand)
# ─────────────────────────────────────────────────────────────────────────────


def _build_banner_line(
    label: str,
    *,
    color: str,
    style: Optional[str] = "bold",
    left_margin: int = 3,
    right_margin: int = 3,
    cap: str = "|",
    filler: str = "=",
    gap: int = 2,  # Abstand zwischen Füllung und Label links/rechts
) -> str:
    cols = shutil.get_terminal_size((80, 20)).columns
    # Sichtbare Breite ohne Außenränder
    content_width = max(10, cols - left_margin - right_margin)
    # Innenbreite zwischen den Caps (|........|)
    inner_width = max(0, content_width - 2)

    # Label-Breite messen und ggf. kürzen (mit Ellipse), falls zu lang
    label_w = _WCSWIDTH(label)
    min_needed = label_w + (gap * 2 if gap > 0 else 0)

    if min_needed > inner_width:
        max_label_w = max(0, inner_width - (gap * 2 if gap > 0 else 0))
        if max_label_w > 0:
            trimmed = label
            # so lange kürzen, bis es passt (Platz für Ellipse berücksichtigen)
            while trimmed and _WCSWIDTH(trimmed) > max_label_w - 1:
                trimmed = trimmed[:-1]
            label = (trimmed + "…") if trimmed else "…"
            label_w = _WCSWIDTH(label)
        else:
            label = ""
            label_w = 0

        min_needed = label_w + (gap * 2 if gap > 0 else 0)
        if min_needed > inner_width:
            # gap notfalls reduzieren
            gap = 1 if inner_width - label_w >= 2 else 0
            min_needed = label_w + (gap * 2 if gap > 0 else 0)

    # Restbreite gleichmäßig mit Füllzeichen verteilen
    rem = max(0, inner_width - min_needed)
    left_fill = rem // 2 + (rem % 2)  # bei ungerade: ein Zeichen mehr links
    right_fill = rem // 2

    inner = (
        (filler * left_fill)
        + (" " * gap if gap else "")
        + label
        + (" " * gap if gap else "")
        + (filler * right_fill)
    )

    # Farbsequenz: style nur anhängen, wenn vorhanden (keine Übergabe von None)
    color_seq = (c.get(style) if style else "") + c.get(color)

    # komplette Zeile: 3 Leerzeichen + farbiges Banner + 3 Leerzeichen
    return (
        (" " * left_margin)
        + color_seq
        + cap
        + inner
        + cap
        + c.get("reset")
        + (" " * right_margin)
    )


def print_banner(
    label: str,
    *,
    color: str,
    style: Optional[str] = "bold",
    left_margin: int = 3,
    right_margin: int = 3,
    cap: str = "|",
    filler: str = "=",
    gap: int = 2,
    leading_newline: bool = False,  # optionales altes Verhalten
    trailing_newline: bool = True,
) -> None:
    if leading_newline:
        print()
    line = _build_banner_line(
        label,
        color=color,
        style=style,
        left_margin=left_margin,
        right_margin=right_margin,
        cap=cap,
        filler=filler,
        gap=gap,
    )
    print(line, end="\n" if trailing_newline else "")


# ─────────────────────────────────────────────────────────────────────────────
# Kopf-/Status-Ausgaben
# ─────────────────────────────────────────────────────────────────────────────


def print_package_headline() -> None:
    print(
        "\n"
        + c.combine("bold", "pearl")
        + c.get("deep_blue", background=True)
        + "          🎬 PhotonFabric VideoKit - "
        + _("package_description")
        + "            "
        + c.get("reset")
        + "\n"
    )


def print_start(method: str) -> None:
    # Einzeilig, dynamisch breit, 3er Außenabstand
    global STARTED
    if STARTED:
        return
    print()
    print_banner(f"[{method}] " + _("starting"), color="light_yellow", style="bold")
    print()
    STARTED = True


def print_error(message: str) -> None:
    print(
        "\n"
        + c.combine("bold", "red")
        + " ❌ "
        + _("error_prefix")
        + f" {message}"
        + c.get("reset")
        + "\n"
    )


def print_warning(message: str) -> None:
    print(
        "\n"
        + c.combine("bold", "orange")
        + " ⚠️ "
        + _("warning_prefix")
        + f" {message}"
        + c.get("reset")
        + "\n"
    )


def print_info(message: str, color: str = "khaki") -> None:
    """Hervorgehobene Info-Zeile (Label + eingerückter Text)."""
    print("\n" + c.combine("bold", "yellow") + " ℹ️ INFO: " + c.get("reset"), end="")
    print_text(message, indent=10, first_indent=False, color=color)
    print()


def print_success(message: str) -> None:
    print(" ✅ " + c.combine("bold", "forest_green") + f"{message}" + c.get("reset"))


def print_fail(message: str) -> None:
    print(" 🚫 " + c.combine("bold", "soft_red") + f"{message}" + c.get("reset"))


def print_file_info(filename: str, message: str) -> None:
    print(
        c.combine("bold", "sand")
        + f" 📄 {filename}:\n   "
        + c.get("reset")
        + f"{message}\n"
    )


def print_debug(context: str, **data: Any) -> None:
    """Kompakte Debug-Informationen."""
    ts = datetime.now().strftime("%H:%M:%S")
    header = (
        c.combine("bold", "magenta") + f"🪛 [{ts}] [DEBUG] {context}:" + c.get("reset")
    )
    print(header)
    for key, value in data.items():
        print(f"  {key}: {value}")
    print()


def print_value_info(label: str, value: str, color: str = "sky_blue") -> None:
    print("  " + c.combine("bold", color) + f"{label}: " + c.get("reset") + f"{value}")


def print_headline(text: str, color: str = "turquoise2") -> None:
    print(c.combine("bold", color) + f"{text}" + c.get("reset"))


def print_line(text: str, color: str = "reset", style: Optional[str] = None) -> None:
    if style is None:
        print(c.get(color) + f"{text}" + c.get("reset"))
    else:
        print(c.combine(color, style) + f"{text}" + c.get("reset"))


def print_promt(
    message: str, color: str = "light_yellow"
) -> None:  # Name beibehalten (API)
    print("\n" + c.combine("bold", color) + f" 👉 {message}" + c.get("reset"))


def print_seperator() -> None:
    print(
        c.get("bold")
        + "----------------------------------------------------------------"
        + c.get("reset")
    )


def print_multi_line(*chunks: Any, sep: str = "", end: str = "\n") -> None:
    """
    Druckt in einer Zeile mehrere farb-/style-formatierte Segmente.

    Jedes 'chunk' kann sein:
      - str
      - (text,)
      - (text, color)
      - (text, color, style)
      - {"text":..., "color":..., "style": ...} (style darf Liste sein)
    """
    # Kurzform: print_multi_line([...])
    if (
        len(chunks) == 1
        and isinstance(chunks[0], (list, tuple))
        and not isinstance(chunks[0], str)
    ):
        chunks = tuple(cast(Iterable[Any], chunks[0]))

    out_parts: list[str] = []

    for ch in chunks:
        # Normalisieren
        if isinstance(ch, str):
            text, color, style_val = ch, None, None
        elif isinstance(ch, dict):
            text = str(ch.get("text", ""))
            color = cast(Optional[str], ch.get("color"))
            style_val = cast(StyleType, ch.get("style"))
        elif isinstance(ch, (list, tuple)):
            seq_ch = cast(Sequence[Any], ch)
            if len(seq_ch) == 1:
                text, color, style_val = str(seq_ch[0]), None, None
            elif len(seq_ch) == 2:
                text, color, style_val = (
                    str(seq_ch[0]),
                    cast(Optional[str], seq_ch[1]),
                    None,
                )
            else:
                text, color, style_val = (
                    str(seq_ch[0]),
                    cast(Optional[str], seq_ch[1]),
                    cast(StyleType, seq_ch[2]),
                )
        else:
            text, color, style_val = str(ch), None, None

        seq: str = ""
        try:
            if isinstance(color, str) and color:
                seq += c.get(color)
        except Exception:
            pass

        # Style zu Iterable[str] normalisieren
        if style_val is None:
            styles: Iterable[str] = ()
        elif isinstance(style_val, str):
            styles = (style_val,)
        else:
            styles = (str(st) for st in cast(Sequence[Any], style_val))

        for st in styles:
            try:
                seq += c.get(st)
            except Exception:
                try:
                    if hasattr(c, "combine") and isinstance(color, str) and color:
                        seq = c.combine(color, st)
                except Exception:
                    pass

        try:
            reset = c.get("reset")
        except Exception:
            reset = "\033[0m"

        out_parts.append(f"{seq}{text}{reset}")

    print(sep.join(out_parts), end=end)


def delete_last_n_lines(n: int, stream: Optional[TextIO] = None) -> None:
    """
    Löscht die letzten n Zeilen im Terminal (inkl. der aktuellen Zeile).
    No-Op, wenn n <= 0 oder stream kein TTY ist.
    """
    if stream is None:
        stream = sys.stdout
    stream = cast(TextIO, stream)

    if not isinstance(n, int) or n <= 0:
        return
    if not getattr(stream, "isatty", lambda: False)():
        return

    try:
        stream.write("\r")
        for i in range(n):
            stream.write("\x1b[2K")  # gesamte Zeile löschen
            if i < n - 1:
                stream.write("\x1b[1A")  # eine Zeile nach oben
            stream.write("\r")
        stream.flush()
    except Exception:
        # Best effort: ignorieren, wenn das Terminal die Sequenzen nicht unterstützt
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Listen-/Tabellen-Ausgaben
# ─────────────────────────────────────────────────────────────────────────────


def print_list(
    items: Sequence[Any],
    color: str = "soft_blue",
    indent: int = 5,
    start: int = 1,
    descriptions: Optional[Mapping[Any, str] | Sequence[str]] = None,
    seperator: str = "#",
    comment_col: int = 10,
) -> None:
    """
    Hübsch formatierte Liste (mit optionaler Beschreibungsspalte).
    """
    total = len(items) + start - 1
    num_width = len(str(total))

    for j, key in enumerate(items, start):
        # Beschreibung ermitteln
        desc = ""
        if descriptions:
            if isinstance(descriptions, Mapping):
                desc = cast(Mapping[Any, str], descriptions).get(key, "")
            elif isinstance(descriptions, (list, tuple)):
                idx = j - start
                if 0 <= idx < len(descriptions):
                    desc = str(descriptions[idx])

        if isinstance(key, (tuple, list)):
            key_seq: Sequence[Any] = cast(Sequence[Any], key)
            key_str = " ".join(str(kk) for kk in key_seq)
        else:
            key_str = str(key)

        pre = (
            " " * indent
            + c.combine("khaki", "bold")
            + f"[{j:>{num_width}d}] "
            + c.get("reset")
            + c.combine(color, "bold")
            + key_str
            + c.get("reset")
        )

        keylen = visible_width(pre)

        fill_len = max(1, comment_col - keylen)
        line = pre + (" " * fill_len)
        print(line, end="")
        ind = comment_col + 2
        print_text(
            f"{seperator} {desc}",
            color="khaki",
            indent=ind,
            first_indent=False,
            style=None,
        )


def print_bullet_list(
    names: Sequence[str],
    descriptions: Sequence[str],
    color1: str = "light_yellow",
    color2: str = "khaki",
    indent: int = 2,
    seperator: str = "-",
) -> None:
    """
    Bullet-Liste mit umbrechenden Beschreibungen.
    """
    if not names:
        return

    max_names_len = max(visible_width(str(nam)) for nam in names)
    term_cols = shutil.get_terminal_size((80, 20)).columns - 4

    for i, name in enumerate(names):
        pad = " " * max(0, max_names_len - visible_width(name))
        hpad = " " * indent
        head = f"{hpad}• {name}{pad}"

        desc = descriptions[i] if i < len(descriptions) else ""
        if desc:
            body = f" {seperator} {desc}"
            head_w = visible_width(head)
            EXTRA = 3
            avail = max(10, term_cols - head_w)
            wrapped = wrap_vis(body, avail)
            print_multi_line(
                (head, color1, "bold"), ((wrapped[0] if wrapped else ""), color2)
            )
            for cont in wrapped[1:]:
                print_multi_line(
                    (" " * (head_w + EXTRA), color1, "bold"), (cont.lstrip(), color2)
                )
        else:
            print_line(head, color=color1)


def print_text(
    text: str,
    color: str = "khaki",
    indent: int = 3,
    first_indent: bool = True,
    style: Optional[str] = "italic",
    keep_spaces: Optional[bool] = False,
) -> None:
    term_cols = shutil.get_terminal_size((80, 20)).columns - 4
    pad = " " * indent
    avail = max(10, term_cols - indent)
    wrapped = wrap_vis(text, avail)
    first = True
    for cont in wrapped[0:]:
        c = cont if keep_spaces else cont.lstrip()
        if first:
            tpad = pad if first_indent else ""
            line = tpad + c
            first = False
        else:
            line = pad + cont.lstrip()
        if style is None or style == "normal":
            print_line((line if cont else ""), color=color)
        else:
            print_line((line if cont else ""), color=color, style=style)


def print_exit(indent: int = 5, text: Optional[str] = None) -> None:
    back_text = _("exit") if text is None else text
    out = (
        "\n"
        + " " * indent
        + c.combine("khaki", "bold")
        + "[0] 🚪 "
        + c.get("reset")
        + c.combine("bright_blue", "bold")
        + back_text
        + c.get("reset")
        + "\n"
    )
    print_line(out)


def return_promt(
    text: str, color: str = "cyan", style: Optional[str] = "bold", end: str = ""
) -> str:
    """Formatierten Prompt (z. B. für input) als String zurückgeben."""
    seq = ""
    if style is not None and style.strip():
        seq += c.get(style)
    seq += c.get(color)
    return seq + text + c.get("reset") + end


def unwrap(value: Any) -> str:
    """List/Tuple hübsch; Strings/Klammern säubern."""
    if isinstance(value, (list, tuple)):
        seq: Sequence[Any] = cast(Sequence[Any], value)
        if len(seq) == 1:
            return unwrap(seq[0])
        return ", ".join(unwrap(v) for v in seq)

    if value is None:
        return ""

    s = str(value).strip()
    # Optional: auch runde Klammern und doppelte Anführungszeichen säubern
    if (s.startswith("[") and s.endswith("]")) or (
        s.startswith("(") and s.endswith(")")
    ):
        s = s[1:-1].strip()
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]
    return s


def collect_relevant_keys(
    params: Mapping[str, Any], mapping: Mapping[str, Any] | None = None
) -> set[str]:
    """
    Ermittelt relevante Parameternamen basierend auf einem rekursiven Spezifikationsbaum.
    Akzeptiert an der Wurzel sowohl Mengen (immer zeigen) als auch Gate-Maps.
    """
    if mapping is None:
        base: Mapping[str, Any] = cast(Mapping[str, Any], defin.RELEVANT_PARAM_GROUPS)
    else:
        base = mapping

    out: set[str] = set()
    _collect_from_root(base, params, out)
    return out


def _collect_from_root(
    spec_map: Mapping[str, Any] | AbcMapping[str, Any],
    params: Mapping[str, Any],
    out: set[str],
) -> None:
    """Top-Ebene: parent_key -> value_map | set(keys)"""
    if not isinstance(spec_map, AbcMapping):
        return

    for parent_key, value_map in spec_map.items():
        # Mengen: direkt hinzufügen (immer anzeigen)
        if isinstance(value_map, set):
            out.update(str(x) for x in cast(Iterable[Any], value_map))
            continue

        # Rekursiv über Gate-Maps
        if isinstance(value_map, AbcMapping):
            _collect_for_parent(str(parent_key), value_map, params, out)
        # Sonst ignorieren


def _collect_for_parent(
    parent_key: str,
    value_map: AbcMapping[Any, Any],
    params: Mapping[str, Any],
    out: set[str],
) -> None:
    """
    value_map: parent_value -> (SpecLeaf | SpecMap)
    """
    parent_val = params.get(parent_key)
    if parent_val is None or parent_val not in value_map:
        return

    selected = value_map.get(parent_val)
    if selected is None:
        return

    # Fall 1: Liste von Kind-Schlüsseln
    if isinstance(selected, list):
        out.update(str(x) for x in cast(Iterable[Any], selected))
        return

    # Fall 2: Tieferer Baum: child_key -> child_value_map
    if isinstance(selected, AbcMapping):
        for child_key, child_value_map in cast(AbcMapping[Any, Any], selected).items():
            if isinstance(child_value_map, AbcMapping):
                _collect_for_parent(str(child_key), child_value_map, params, out)
        return
    # Sonst: ignorieren


def _to_str_key_set(val: Any) -> Set[str]:
    out: Set[str] = set()
    if val is None:
        return out
    if isinstance(val, (set, list, tuple)):
        for k in cast(Iterable[object], val):
            if k is not None:
                out.add(str(k))
        return out
    if isinstance(val, Mapping):
        # Falls jemand eine Map von Gruppen → Keys liefert, nimm ihre Keys
        for k in cast(Mapping[object, Any], val).keys():
            if k is not None:
                out.add(str(k))
        return out
    out.add(str(val))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ausgabetabellen (Parameter/Presets)
# ─────────────────────────────────────────────────────────────────────────────


def print_selected_params_table(
    params: Dict[str, Any],
    labelcolor: str = "sky_blue",
    *,
    labels: Optional[Dict[str, Any]] = None,
    relevant_groups: Optional[Dict[str, Any]] = None,
    show_files: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Generische Ausgabe-Tabelle für ausgewählte Parameter.
    Keine domänenspezifische Logik (kein Stabilize/Denoise/Color-Sonderfall).
    Ein Wrapper (z. B. im Enhance-Modul) kann params vorab "säubern/aufbereiten".
    """
    # ---- Files oben (optional) ----
    files_any: Any = params.get("files") or []
    files: list[str] = list(files_any) if isinstance(files_any, (list, tuple)) else []
    if show_files and files:
        print_line("\n      " + _("files"), color="light_yellow", style="bold")
        print_seperator()
        for f in files:
            fi = Path(f).name
            print_line("  " + str(fi), style="italic")
        print(" ")

    if title is None:
        title = _("selected_parameters_header")

    # ---- Header ----
    print_line("\n      " + title, color="light_yellow", style="bold")
    print_seperator()

    # ---- Relevanz bestimmen ----
    try:
        mapping: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (
                relevant_groups
                if relevant_groups is not None
                else defin.RELEVANT_PARAM_GROUPS
            ),
        )
        raw_any: Any = collect_relevant_keys(params, mapping)
    except Exception:
        raw_any = None

    show_keys_raw: Set[str] = _to_str_key_set(raw_any)
    all_param_keys = {k for k in _keys_str(params) if k != "files"}
    show_keys = show_keys_raw & all_param_keys
    if not show_keys:
        show_keys = all_param_keys

    # ---- Zeilen generisch aus params erzeugen (keine Sonderfälle) ----
    rows: list[tuple[str, str, Any]] = []
    for key, value in params.items():
        if key == "files":
            continue
        if key not in show_keys:
            continue
        if value is None:
            continue
        rows.append((key, _label_for(key, labels), value))

    if not rows:
        print_line("    " + _("no_parameters_changed"), color="dim")
        print(" ")
        return

    # ---- Ausgabe ----
    max_label_len = max(_ansi_len(lbl) for _, lbl, _ in rows)
    pad_left = " " * 4
    sep_col = " | "
    for _key, lbl, val in rows:
        pad = " " * max(0, max_label_len - _ansi_len(lbl))
        label_colored = c.combine("bold", labelcolor) + lbl + c.get("reset")
        lines = [val] if isinstance(val, str) else list(_fmt_value(val))
        print(
            pad_left
            + label_colored
            + pad
            + sep_col
            + c.get("bold")
            + lines[0]
            + c.get("reset")
        )
        if len(lines) > 1:
            cont_prefix = pad_left + " " * max_label_len + sep_col
            for ln in lines[1:]:
                print(cont_prefix + c.get("bold") + ln + c.get("reset"))
    print(" ")


def print_preset_params_table(preset_key: str, labelcolor: str = "sky_blue") -> None:
    """
    Gibt alle Einstellungen eines Presets aus ENHANCE_PRESETS im Tabellenstil aus.
    """
    preset = defin.ENHANCE_PRESETS.get(preset_key)
    if preset is None:
        print_error(f"{_('invalid_preset')}: {preset_key}")
        return

    # Name (de/en) auflösen
    name_field = preset.get("name")
    if isinstance(name_field, dict):
        preset_name = name_field.get(get_lang(), preset_key)
    else:
        preset_name = str(name_field) if name_field is not None else preset_key

    header = f"\n      Preset: {preset_name}"
    print_line(header, color="light_yellow", style="bold")
    print_seperator()

    entries: list[tuple[Any, str]] = []
    max_label_len = 2

    for key, val in preset.items():
        # Metadaten überspringen
        if key in ("name", "description", "virtual", "filter_chain"):
            continue
        # Farb-Defaults (50) überspringen
        if (
            key in _COLOR_PARAMS_DEFAULT_SKIP
            and isinstance(val, (int, float))
            and val == 50
        ):
            continue
        label = _(key)
        max_label_len = max(max_label_len, len(label))
        entries.append((val, label))

    max_label_len += 1
    for value, label in entries:
        spacing = " " * (max_label_len - len(label))
        print(
            "    "
            + c.combine("bold", labelcolor)
            + label
            + c.get("reset")
            + spacing
            + "| "
            + c.get("bold")
            + str(value)
            + c.get("reset")
        )
    print()


def parse_hash_sections(
    line: str, *, allow_indent: bool = False, sign: str = "#"
) -> tuple[str, str] | None:
    """
    Zerlegt eine Zeile der Form:
        "#Titel# Rest des Textes"
    in (titel, rest). Gibt None zurück, wenn das Muster nicht passt.

    Parameter
    ---------
    line : str
        Eingabezeile.
    allow_indent : bool, default False
        True: Beliebige führende Whitespaces (inkl. Tabs) vor dem ersten 'sign' erlaubt.
        False: Zeile muss direkt mit 'sign' beginnen.
    sign : str, default "#"
        Marker-Zeichen(folge), typischerweise "#".

    Regeln
    ------
    - Genau ein führendes 'sign' (kein doppeltes 'sign' unmittelbar danach).
    - Das nächste 'sign' beendet den Titel.
    - Rückgabe ist (titel.strip(), rest.strip()) oder None.
    """
    if not isinstance(line, str) or not sign:
        return None

    s = line.rstrip("\r\n")
    i0 = 0

    if allow_indent:
        # alle führenden Whitespaces inkl. Tabs erlauben/überspringen
        while i0 < len(s) and s[i0].isspace():
            i0 += 1

    # Muss mit 'sign' beginnen (nach optionalem Einrücken)
    if i0 >= len(s) or not s.startswith(sign, i0):
        return None

    # Doppeltes 'sign' direkt am Start (z. B. '##' bei sign='#') ausschließen
    after_first = i0 + len(sign)
    if after_first <= len(s) - len(sign) and s.startswith(sign, after_first):
        return None

    # Zweites 'sign' suchen
    i2 = s.find(sign, after_first)
    if i2 == -1:
        return None  # kein zweites 'sign'

    title = s[after_first:i2].strip()
    rest = s[i2 + len(sign) :].strip()
    return (title, rest)


# ─────────────────────────────────────────────────────────────────────────────
# Infofile-Ausgabe
# ─────────────────────────────────────────────────────────────────────────────


def show_infofile(file: Optional[str]) -> None:
    # Farben wie im Bash-Skript
    headcolor = "sky_blue"
    subheadcolor = "gold"
    speccolor = "pearl"
    listcolor = "fuchsia"  # "sea_green"
    textcol = "khaki"

    if not file:
        return

    extension = file.split(".")[-1] if "." in file else ""
    if not extension:
        file = file + ".info"
    else:
        if "info" not in extension:
            file = re.sub(r"\." + re.escape(extension) + r"$", ".info", file)

    if not os.path.isfile(file):
        return

    print("\n\n")
    with open(file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    n = len(lines)
    while i < n:
        sline = lines[i].rstrip("\n")

        # Header: ### am Zeilenanfang
        if sline.startswith("###"):
            headline = sline[3:].strip()
            print_banner(
                "=" * len(headline),
                color=headcolor,
                left_margin=6,
                right_margin=6,
                gap=0,
            )
            print_banner(headline, color=headcolor, left_margin=6, right_margin=6)
            print_banner(
                "=" * len(headline),
                color=headcolor,
                left_margin=6,
                right_margin=6,
                gap=0,
            )
            i += 1
            continue

        # Subheader: ## (aber nicht ###)
        if sline.startswith("##"):
            sub = sline[2:].strip()
            # print(subheader + sub + end)
            print_line(sub, color=subheadcolor, style="bold")
            i += 1
            continue

        # Einzeilige Spez mit '#Titel# Text'
        if sline.startswith("#"):
            res = parse_hash_sections(sline, allow_indent=False, sign="#")
            if res is not None:
                title, body = res
                f_title = return_promt(title, color=speccolor)
                print(f_title, end="")
                print_text(
                    " " + body,
                    indent=len(f_title) + 1,
                    first_indent=False,
                    style="bold",
                    keep_spaces=True,
                )
            else:
                print_text(sline, style="bold", keep_spaces=True)
            i += 1
            continue

        # Block: aufeinanderfolgende Zeilen, die mit genau einem '§' (nach Einrückung) starten
        t = sline.lstrip()
        if t.startswith("§") and not t.startswith("§§"):
            names: list[str] = []
            descs: list[str] = []

            while i < n:
                cur = lines[i].rstrip("\n")
                tc = cur.lstrip()
                if not (tc.startswith("§") and not tc.startswith("§§")):
                    break

                parsed = parse_hash_sections(cur, allow_indent=True, sign="§")
                if parsed is None:
                    # nicht wohlgeformt → Block beenden, vorher gesammeltes ausgeben
                    break

                name, body = parsed
                names.append(name)
                descs.append(body)
                i += 1  # zur nächsten Zeile des Blocks

            # Ausgabe des gesammelten Blocks
            if names:
                print_bullet_list(
                    names,
                    descs,
                    color1=listcolor,  # Namen z. B. in 'gold'
                    color2=textcol,  # Beschreibungen in 'khaki'
                    indent=count_leading_indent_spaces(sline),
                    seperator="-",
                )
            continue  # nicht i++ mehr; wir stehen schon auf erster Nicht-§-Zeile

        # Normaltext
        # print(textcol + sline + end)
        # print_line(sline, color=textcol)
        if sline.lstrip() == "":
            print("")
        else:
            print_text(
                sline,
                color=textcol,
                indent=count_leading_indent_spaces(sline),
                style=None,
            )
        i += 1

    print()


def show_info(subcommand: Optional[str] = None) -> None:
    """
    Sucht in <projekt>/infofiles/ eine .info-Datei mit Postfix
    [<subcommand>.]<lang>.info, wobei <lang> aus i18n.get_lang() kommt.
    Beispiel: PhotonFabric.convert.de.info  oder PhotonFabric.de.info

    - Ignoriert den 'base'-Namen (Präfix) der Datei.
    - Fallbacks:
        1) Wenn <subcommand>-spezifisch nichts gefunden → <lang>.info ohne subcommand
        2) Wenn Sprache ≠ 'en' und nichts gefunden → .en.info
    - Findet den Projektordner über den Skriptpfad (sys.argv[0]) und sucht 'infofiles'
      im aktuellen Ordner oder aufwärts in den Elternordnern. Als letzte Chance wird
      auch $CWD/infofiles geprüft.
    """
    import i18n

    # 1) Projektordner bestimmen (Start: Verzeichnis des Entry-Points)
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] not in ("", "-c") else __file__
    start_dir = Path(argv0).resolve().parent

    def find_info_dir(p: Path) -> Optional[Path]:
        # p/infofiles, dann in Elternordnern suchen
        for cur in [p] + list(p.parents):
            cand = cur / "infofiles"
            if cand.is_dir():
                return cand
        # letzte Chance: CWD/infofiles
        cwd_cand = Path.cwd() / "infofiles"
        return cwd_cand if cwd_cand.is_dir() else None

    info_dir = find_info_dir(start_dir)
    if not info_dir:
        # Kein infofiles-Ordner gefunden → nichts anzeigen
        show_infofile(None)
        sys.exit(0)

    lang = getattr(i18n, "get_lang", lambda: "de")()
    lang = (lang or "de").lower()

    # 2) Postfix(e) aufbauen – erst mit subcommand, dann Fallbacks
    wanted_suffixes = []
    if subcommand:
        wanted_suffixes.append(f".{subcommand}.{lang}.info")
    wanted_suffixes.append(f".{lang}.info")
    if lang != "en":
        wanted_suffixes.append(".en.info")  # letzter Fallback

    # 3) Kandidaten finden (Case-insensitive Endswith-Match auf Postfix)
    infos = sorted(info_dir.glob("*.info"))
    chosen: Optional[Path] = None
    lower_names = [(p, p.name.lower()) for p in infos]

    for suffix in wanted_suffixes:
        suf = suffix.lower()
        for p, lname in lower_names:
            if lname.endswith(suf):
                chosen = p
                break
        if chosen:
            break

    # 4) Anzeigen (oder still aussteigen, wenn nichts da ist)
    if chosen:
        show_infofile(str(chosen))
    else:
        show_infofile(None)

    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Kompakte Zusammenfassung / Abschlussbanner
# ─────────────────────────────────────────────────────────────────────────────


def print_finished(action_name: str) -> None:
    # Einzeilig, dynamisch breit, 3er Außenabstand
    global ENDED
    if ENDED:
        return
    print()
    print_banner(f"[{action_name}] " + _("finished"), color="emerald", style="bold")
    print()
    ENDED = True
