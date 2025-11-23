#!/usr/bin/env python3
from __future__ import annotations

import builtins
import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

import consoleOutput as co
import definitions as defin
import fileSystem as fs
import helpers as he
import imageProcessor as ip
import overwrites as ow
import VideoEncodersCodecs as vec
from ANSIColor import ANSIColor

# local modules
from i18n import _, tr
from imageDisplayer import ImageDisplayer

builtins.input = ow.safe_input


# ---- Typing helpers / aliases ------------------------------------------------
# make i18n._ explicit for Pylance
_: Callable[[str], str] = cast(Callable[[str], str], _)

floatNumberReturn: TypeAlias = Union[float, str, Tuple[float, str]]
NumberReturn = Union[
    float, str, Tuple[float, str]
]  # belassen; nur diese Funktion wird optional

# Grouped-tag selection helpers
SetOption: TypeAlias = Tuple[str, str, Any]  # (key, label, value-shown)
UnsetOption: TypeAlias = Tuple[str, str, Optional[str]]  # (key, label, description)
GroupedTagResult: TypeAlias = Tuple[
    Literal["edit", "add", "thumb", "del_thumb", "exit"], Optional[str]
]

T = TypeVar("T")

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # CSI-Sequenzen

displayer = ImageDisplayer()
c = ANSIColor()


def ask_user(
    prompt: str,
    options: Sequence[str],
    descriptions: Optional[Sequence[Optional[str]]] = None,
    default: int = 0,
    display_labels: Optional[Sequence[str]] = None,
    explanation: Optional[str] = None,
    back_button: bool = True,
) -> Optional[str]:
    noptions = len(options)

    if not 0 <= default < noptions:
        default = 0

    if display_labels and len(display_labels) == noptions:
        labels: List[str] = [str(x) for x in display_labels]
    else:
        labels = [str(x) for x in options]

    descs: Optional[List[Optional[str]]]
    if isinstance(descriptions, (list, tuple)):
        descs = list(descriptions)
    else:
        descs = None

    if back_button:
        labels.append("⬅️ " + _("back"))
        if descs is not None:
            if len(descs) < noptions:
                descs += [None] * (noptions - len(descs))
            descs.append(None)

    max_labels_length = max(co.visible_width(str(lbl)) for lbl in labels) + 1
    shown_prompt = f"{prompt} [{labels[default]}]"

    term_cols = shutil.get_terminal_size((80, 20)).columns - 4

    while True:
        co.print_promt(shown_prompt)
        if explanation:
            co.print_text("    " + _(explanation), color="sand", indent=4)
        co.print_seperator()

        index_digits = len(str(len(labels)))

        for i, label in enumerate(labels, start=1):
            label_str = str(label)
            pad = " " * max(0, max_labels_length - co.visible_width(label_str))
            col_numb = co.return_promt(f"[{i:>{index_digits}}]", color="khaki")
            col_label = co.return_promt(label_str, color="bright_blue")
            head = f"    {col_numb} {col_label}{pad}"
            desc = ""
            if descs and (i - 1) < len(descs) and descs[i - 1]:
                desc = str(descs[i - 1])

            if desc:
                body = " - " + desc
                head_w = co.visible_width(head)
                EXTRA = 3
                avail = max(10, term_cols - head_w)
                wrapped = co.wrap_vis(body, avail)
                co.print_line(
                    head
                    + (
                        co.return_promt(wrapped[0], color="khaki", style=None)
                        if wrapped
                        else ""
                    )
                )
                for cont in wrapped[1:]:
                    co.print_line(" " * (head_w + EXTRA) + cont.lstrip(), color="khaki")
            else:
                co.print_line(head)

        co.print_exit(indent=4)

        choice = input(co.return_promt("      " + _("choose_option") + ": ")).strip()
        print("")

        if choice == "0":
            co.print_line(f"  {_('exiting')} ...")
            sys.exit(0)

        if choice == "":
            return str(options[default])

        try:
            idx = int(choice) - 1
            if 0 <= idx < noptions:
                return str(options[idx])
            if back_button and idx == noptions:
                return None
        except ValueError:
            pass

        total_visible = noptions + (1 if back_button else 0)
        co.print_fail(_("invalid_ask_input").format(option_num=total_visible))


def ask_yes_no(
    prompt: str,
    default: bool = False,
    explanation: Optional[str] = None,
    back_option: bool = True,
) -> Optional[bool]:
    options = [_("yes"), _("no")]
    def_int = 0 if default else 1
    full_prompt = f"{_(prompt)}"
    choice = ask_user(
        full_prompt,
        options,
        default=def_int,
        explanation=explanation,
        back_button=back_option,
    )
    if choice is None:
        return None
    if choice == options[0]:
        return True
    if choice == options[1]:
        return False
    return default


def read_percent(prompt: str, default: int = 50) -> int:
    val = input(
        "  " + co.return_promt(_(prompt).format(default=default), color="salmon")
    ).strip() or str(default)

    try:
        val_int = int(val)
        if not (0 <= val_int <= 100):
            raise ValueError
        return val_int
    except ValueError:
        co.print_error(_("read_percent_error"))
        raise


def _get_time_input(defaults: List[float]) -> List[float]:
    co.print_promt(
        _("get_preview_time_input").format(
            default=",".join(str(int(p * 100)) for p in defaults)
        )
    )
    inp = input("  > ").strip()
    if not inp:
        return defaults
    pts: List[float] = []
    for part in re.split(r"[,; ]+", inp):
        try:
            v = float(part.replace("%", ""))
            if 0 <= v <= 100:
                pts.append(v / 100)
        except ValueError:
            pass
    return pts if pts else defaults


def ask_factor_or_number(
    prompt: Optional[str],
    error_message: Optional[str],
    user_input: Optional[str] = None,
    suffix: Optional[str] = "x",
) -> Optional[Tuple[bool, float]]:
    if user_input is None:
        if prompt:
            co.print_text(prompt, style="bold")
        raw = input("  : ").strip()
    else:
        raw = user_input.strip()

    if suffix is not None:
        # 1) Faktor mit 'x' (case-insensitive)
        pat = rf"(?i)^\s*([0-9]+(?:[.,][0-9]+)?)\s*{re.escape(suffix)}\s*$"
        m = re.fullmatch(pat, raw)
        if m:
            try:
                return True, float(m.group(1).replace(",", "."))
            except Exception:
                pass

    # 2) Bruch a/b am Zeilenanfang (erlaubt nachgestellte Einheiten, z. B. " fps")
    m = re.match(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*/\s*([+-]?\d+(?:[.,]\d+)?)", raw)
    if m:
        try:
            num = float(m.group(1).replace(",", "."))
            den = float(m.group(2).replace(",", "."))
            if den == 0:
                raise ZeroDivisionError
            return False, num / den
        except Exception:
            if error_message:
                co.print_fail(error_message)
            else:
                co.print_fail(f"Invalid fraction: '{raw}'")
            return None

    # 3) Sonst: Einheiten abtrennen und als Zahl interpretieren
    token = he.strip_trailing_units(raw)
    if not token:
        if error_message:
            co.print_fail(error_message)
        else:
            co.print_fail(f"Invalid input: '{raw}'")
        return None

    token = token.replace(",", ".").strip()
    try:
        return False, float(token)
    except Exception:
        if error_message:
            co.print_fail(error_message)
        else:
            co.print_fail(f"Invalid number: '{raw}'")
        return None


def ask_user_grouped_tags(
    prompt: str,
    set_options: Sequence[SetOption],
    unset_options: Sequence[UnsetOption],
    delete_thumb: bool = False,
    default_index: int = 0,
    back_button_text: str = _("exit"),
) -> GroupedTagResult:
    """
    Zweigeteilte Anzeige mit fortlaufendem Index:
      - set_options:  [(key, label, value), ...]    -> bereits gesetzte Tags; 'value' wird angezeigt (statt desc)
      - unset_options:[(key, label, desc),  ...]    -> ungesetzte Tags; 'desc' wird angezeigt
    Rückgabe:
      ("edit", key)   wenn ein gesetzter Tag gewählt wurde
      ("add",  key)   wenn ein ungesetzter Tag gewählt wurde
      ("thumb", None) bei '#'
      ("del_thumb", None) bei '*'
      ("exit",  None) bei '0'
    """
    n_set = len(set_options)
    n_unset = len(unset_options)
    total = n_set + n_unset

    # Default absichern
    if not (0 <= default_index < max(n_set, 1)):
        default_index = 0

    # sichtbare Breite über alle Labels (ohne Sequence-Concatenation)
    all_labels: List[str] = []
    all_labels.extend(str(lbl) for _, lbl, _ in set_options)
    all_labels.extend(str(lbl) for _, lbl, _ in unset_options)
    if not all_labels:
        all_labels = [""]

    max_labels_length = max(co.visible_width(lbl) for lbl in all_labels) + 1

    # Default-Label für Prompt
    if total > 0:
        if n_set:
            default_label: str = str(set_options[default_index][1])
        else:  # n_unset must be > 0 here
            default_label = str(unset_options[default_index][1])
    else:
        default_label = _("exit_and_save")

    full_prompt = f"{prompt} [{default_label}]"

    term_cols = shutil.get_terminal_size((80, 20)).columns - 4
    index_digits = len(str(max(1, total)))

    while True:
        co.print_promt(full_prompt)
        co.print_seperator()

        # ===== Gesetzte Tags (Beschreibung ausgeblendet, stattdessen VALUE anzeigen) =====
        if n_set:
            co.print_headline(
                "\n   ====== " + _("editable_tags_set") + " ======", "sky_blue"
            )
            for i, (key, label, value) in enumerate(set_options, start=1):
                label_str = str(label)
                pad = " " * max(0, max_labels_length - co.visible_width(label_str))
                number = f"    [{i:>{index_digits}}] "
                name = f"{label_str}:{pad}"
                head = number + name

                # 'value' wie Beschreibung behandeln (Wrapper/Einschub)
                if value is not None and value != "":
                    body = str(value)
                    head_w = co.visible_width(head)
                    EXTRA = 3
                    avail = max(10, term_cols - head_w)
                    wrapped = co.wrap_vis(body, avail)
                    co.print_multi_line(
                        (number, "khaki", "bold"),
                        (name, "bright_blue"),
                        (wrapped[0] if wrapped else "", "bright_blue", "bold"),
                    )
                    for cont in wrapped[1:]:
                        co.print_line(
                            " " * (head_w + EXTRA) + cont.lstrip(), color="khaki"
                        )
                else:
                    # Kein Wert -> nur Kopfzeile
                    co.print_multi_line(
                        (number, "khaki", "bold"), (name, "bright_blue")
                    )
            print()

        # ===== Ungesetzte Tags (Beschreibung normal anzeigen) =====
        if n_unset:
            co.print_headline(
                "   ====== " + _("editable_tags_unset") + " ======", "bright_green"
            )
            for j, (key, label, desc) in enumerate(unset_options, start=n_set + 1):
                label_str = str(label)
                pad = " " * max(0, max_labels_length - co.visible_width(label_str))
                number = f"    [{j:>{index_digits}}] "
                name = f"{label_str}{pad}"
                head = number + name

                if desc:
                    body = " - " + str(desc)
                    head_w = co.visible_width(head)
                    EXTRA = 3
                    avail = max(10, term_cols - head_w)
                    wrapped = co.wrap_vis(body, avail)
                    co.print_multi_line(
                        (number, "khaki", "bold"),
                        (name, "light_green"),
                        (wrapped[0] if wrapped else "", "khaki"),
                    )
                    for cont in wrapped[1:]:
                        co.print_line(
                            " " * (head_w + EXTRA) + cont.lstrip(), color="khaki"
                        )
                else:
                    co.print_multi_line(
                        (number, "khaki", "bold"), (name, "light_green")
                    )

        # Spezialzeilen
        print()
        co.print_multi_line(
            ("    [#] ", "khaki", "bold"), (_("set_thumbnail"), "salmon", "bold")
        )
        if delete_thumb:
            co.print_multi_line(
                ("    [*] ", "khaki", "bold"), (_("delete_thumbnail"), "salmon", "bold")
            )
        co.print_exit(indent=4, text=back_button_text)  # rendert [0] Exit/Zurück

        choice = input(co.return_promt("      " + _("choose_option") + ": ")).strip()

        if choice == "0":
            return ("exit", None)
        if choice == "":
            # Default zurückgeben
            if total == 0:
                return ("exit", None)
            if n_set:
                return ("edit", set_options[default_index][0])
            # hier muss n_unset > 0 sein
            return ("add", unset_options[default_index][0])
        if choice == "#":
            return ("thumb", None)
        if delete_thumb and choice == "*":
            return ("del_thumb", None)

        try:
            idx = int(choice)
        except ValueError:
            co.print_fail(_("invalid_ask_input").format(option_num=total))
            continue

        if 1 <= idx <= total:
            if idx <= n_set:
                return ("edit", set_options[idx - 1][0])
            else:
                return ("add", unset_options[idx - n_set - 1][0])

        co.print_fail(_("invalid_ask_input").format(option_num=total))


def _fmt_num(x: float) -> str:
    return str(int(x)) if abs(x - int(x)) < 1e-9 else str(x)


def _normalize_to_rational(value: str, denom_limit: int = 1001) -> str:
    """
    Normiert übliche Eingaben auf einen stabilen Rational-String.
    - erkennt NTSC-klassiker exakt (23.976, 29.97, 59.94)
    - akzeptiert Brüche a/b
    - wandelt Dezimalzahlen in gekürzte Brüche (limit_denominator)
    """
    value = value.strip()
    ntsc_map = {"23.976": "24000/1001", "29.97": "30000/1001", "59.94": "60000/1001"}
    if value in ntsc_map:
        return ntsc_map[value]
    if re.fullmatch(r"[+-]?\d+/\d+", value):
        # Bruch ggf. kürzen
        num, den = value.split("/")
        return f"{Fraction(int(num), int(den))}"
    # Dezimal → Bruch
    f = float(value.replace(",", "."))
    # Ganze Zahl bleibt ganz
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return str(Fraction(f).limit_denominator(denom_limit))


def _parse_number_token(token: str, denom_limit: int = 1001) -> Tuple[float, str]:
    """
    Gibt (float_wert, rational_string) zurück.
    Erlaubt: '12', '12.5', '12,5', '30000/1001', '29.97' → ('30000/1001').
    """
    token = token.strip()
    rat = _normalize_to_rational(token, denom_limit=denom_limit)
    fval = float(Fraction(rat))
    return fval, rat


# ── Overloads: binden Rückgabetyp an 'return_type' ───────────────────────────
@overload
def ask_float(
    prompt: str,
    *,
    default: Optional[Union[float, str]] = ...,
    min_value: Optional[float] = ...,
    max_value: Optional[float] = ...,
    back_button: bool = ...,
    return_type: Literal["float"] = "float",
    denom_limit: int = ...,
) -> float: ...
@overload
def ask_float(
    prompt: str,
    *,
    default: Optional[Union[float, str]] = ...,
    min_value: Optional[float] = ...,
    max_value: Optional[float] = ...,
    back_button: bool = ...,
    return_type: Literal["rational"],
    denom_limit: int = ...,
) -> str: ...
@overload
def ask_float(
    prompt: str,
    *,
    default: Optional[Union[float, str]] = ...,
    min_value: Optional[float] = ...,
    max_value: Optional[float] = ...,
    back_button: bool = ...,
    return_type: Literal["both"],
    denom_limit: int = ...,
) -> Tuple[float, str]: ...


def ask_float(
    prompt: str,
    *,
    default: Optional[Union[float, str]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    back_button: bool = False,
    return_type: Literal["float", "rational", "both"] = "float",
    denom_limit: int = 1001,
) -> floatNumberReturn:
    """
    Interaktive Zahleneingabe (int/float mit Punkt oder Komma, oder Bruch a/b).
      - ENTER: übernimmt 'default' (falls gesetzt), sonst erneut fragen
      - [0]: Exit (falls back_button=True)
      - Bereichsprüfung: min_value <= x <= max_value (je nach Angabe)
      - Rückgabe: float | rational 'a/b' | (float, 'a/b')

    Beispiele:
      ask_float("CRF wählen", default=23, min_value=0, max_value=51)
      ask_float("Ziel-FPS", default="29.97", min_value=1, max_value=480, return_type="rational")
    """
    # Bereich normalisieren
    if (min_value is not None) and (max_value is not None) and (max_value < min_value):
        min_value, max_value = max_value, min_value

    # Default vorbereiten (kann float oder string sein, z. B. "29.97")
    default_str: Optional[str] = None
    if default is not None:
        try:
            # numerisch formatiert anzeigen
            f, _r = _parse_number_token(str(default), denom_limit=denom_limit)
            default_str = _fmt_num(f)
        except Exception:
            default_str = None  # ungültiger Default → ignorieren

    # Prompt anzeigen
    hint = f" [{default_str}]" if default_str is not None else ""
    shown_prompt = f"{prompt}{hint}"

    # Bereichs-Hinweis
    range_hint_parts: List[str] = []
    if min_value is not None:
        range_hint_parts.append(f"≥ {_fmt_num(min_value)}")
    if max_value is not None:
        range_hint_parts.append(f"≤ {_fmt_num(max_value)}")
    range_hint = ("   " + " / ".join(range_hint_parts)) if range_hint_parts else ""

    while True:
        co.print_promt(shown_prompt)

        if range_hint:
            co.print_line(range_hint, "gold")
        if back_button:
            co.print_exit()
        co.print_seperator()
        raw = input(co.return_promt(" : ")).strip()

        # Exit?
        if back_button and raw == "0":
            co.print_line("  " + _("exiting"))
            sys.exit(0)

        # ENTER → nur bei vorhandenem Default
        if raw == "":
            if default_str is None:
                co.print_fail(_("enter_float_value"))
                continue
            # Default parsen
            fval, rstr = _parse_number_token(default_str, denom_limit=denom_limit)
        else:
            # Komma erlauben; Validierung/Parsing inkl. Brüchen
            try:
                fval, rstr = _parse_number_token(
                    raw.replace(",", "."), denom_limit=denom_limit
                )
            except Exception:
                co.print_fail(_("invalid_float_number"))
                continue

        # Bereich prüfen
        if (min_value is not None and fval < min_value) or (
            max_value is not None and fval > max_value
        ):
            lo = _fmt_num(min_value) if min_value is not None else "-∞"
            hi = _fmt_num(max_value) if max_value is not None else "+∞"
            co.print_fail(_("float_not_in_range") + f" => {lo} … {hi}")
            continue

        # Rückgabe nach Wunsch
        if return_type == "float":
            return fval
        if return_type == "rational":
            return rstr
        return (fval, rstr)


def _normalize_pair_default(default: object) -> Optional[Tuple[int, int]]:
    """
    Default kann Tuple[int,int] oder String ('1280x720', '1280:720', '1280 -2') sein.
    Gibt (w,h) zurück oder None, wenn nicht parsbar.
    """
    if default is None:
        return None
    if isinstance(default, tuple) and len(default) == 2:
        try:
            return int(default[0]), int(default[1])
        except Exception:
            return None
    if isinstance(default, str):
        nums = re.findall(r"[+-]?\d+", default)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
    return None


def ask_two_ints(
    prompt: str,
    output_sep: str = ":",
    default: Optional[object] = None,  # str oder (int,int)
    explanation: Optional[str] = None,
    back_button: bool = True,
) -> str:
    """
    Fragt interaktiv zwei Ganzzahlen ab und gibt sie als 'W{sep}H' zurück.
      - Akzeptiert beliebige Trennzeichen (':', ',', '/', 'x', 'X', ';', '-', '*', Space).
      - ENTER: übernimmt 'default' (falls gesetzt), sonst erneut fragen.
      - [0]: Exit (falls back_button=True).
    """
    # Default normalisieren
    d_pair = _normalize_pair_default(default)
    default_hint = f" [{d_pair[0]}{output_sep}{d_pair[1]}]" if d_pair else ""
    shown_prompt = f"{prompt}{default_hint}"

    while True:
        co.print_promt(shown_prompt)

        if explanation is not None:
            co.print_line("    " + _(explanation), "sand")

        if back_button:
            co.print_exit()  # zeigt "[0] Exit"
        co.print_seperator()
        raw = input(co.return_promt(" : ")).strip()

        # Exit?
        if back_button and raw == "0":
            co.print_line("  " + ("exiting"))
            sys.exit(0)

        # ENTER -> Default (falls vorhanden)
        if raw == "":
            if d_pair:
                w, h = d_pair
                return f"{w}{output_sep}{h}"
            co.print_fail(_("enter_2integer_numbers"))
            continue

        # Zahlen extrahieren (erlaubt beliebige Trenner)
        nums = re.findall(r"[+-]?\d+", raw)
        if len(nums) < 2:
            co.print_fail(_("enter_2integer_numbers"))
            continue

        try:
            w = int(nums[0])
            h = int(nums[1])
        except Exception:
            co.print_fail(_("enter_2integer_numbers"))
            continue

        return f"{w}{output_sep}{h}"


def read_frame_time() -> str | None:
    """
    Fragt wiederholt nach einem Zeitpunkt, bis das Format gültig ist.
    Rückgabewert ist der bereinigte (strip) Originalstring.
    """
    while True:
        frame_time = input(co.return_promt("\n" + _("enter_frame_time") + ": ")).strip()
        ans = he.is_valid_time_or_percent(frame_time)
        if ans is None:
            return None
        if ans:
            return frame_time
        co.print_fail(_("invalid_time_input"))


def ask_time_or_other_number(
    prompt: str,
    suffix: Optional[str] = None,
    suffixes: Optional[Sequence[str]] = None,  # NEU: mehrere Suffixe optional
    default: Optional[Union[str, float]] = None,
    loop: Optional[bool] = True,
    exit_on_0: Optional[bool] = True,
    return_raw_if_nonnumber: Optional[bool] = False,
) -> Optional[Tuple[bool, Union[str, float]]]:
    """
    Interaktive Eingabe: Zeit ODER andere Zahl (optional mit Suffix/Suffixen).
    Rückgabe:
      - (True,  time_str)  → Zeitwert als String (z. B. '00:00:02.5', '1:30', '12s' → '12')
      - (False, number)    → andere Zahl als float, z. B. '30fps' (bei suffix='fps') oder '30p', '85%'
      - None               → bei '0' (Exit)
    """

    # Alle erlaubten Suffixe sammeln (Reihenfolge/Kompatibilität wahren)
    allowed_suffixes: List[str] = []
    if suffix:
        allowed_suffixes.append(str(suffix))
    if suffixes:
        allowed_suffixes.extend([s for s in suffixes if s])

    # Für Anzeige: erstes vorhandenes Suffix verwenden (kompatibel zum bisherigen Verhalten)
    display_suffix = suffix or (allowed_suffixes[0] if allowed_suffixes else "")

    # Prompt mit Default-Hinweis bauen
    if default is None:
        shown_prompt = prompt
        default_token_for_apply: Optional[str] = None
    else:
        if isinstance(default, (int, float)):
            default_disp = f"{default}{display_suffix}"
            default_token_for_apply = default_disp
        else:
            default_disp = str(default)
            default_token_for_apply = default_disp
        shown_prompt = f"{prompt} [{default_disp}]"

    # Regex zum Entfernen eines der erlaubten Suffixe am Ende (nur zur Erkennung)
    if allowed_suffixes:
        # längste zuerst, damit z. B. "fps" vor "s" (falls jemals vorhanden) gematcht würde
        suf_alt = "|".join(
            re.escape(s) for s in sorted(set(allowed_suffixes), key=len, reverse=True)
        )
        tail_suf_regex = re.compile(rf"\s*(?:{suf_alt})\s*$")
    else:
        tail_suf_regex = None

    while True:
        co.print_promt(shown_prompt)
        co.print_seperator()
        raw_in = input(co.return_promt(" : ")).strip()

        # ENTER → Default nutzen (falls vorhanden)
        if raw_in == "" and default_token_for_apply is not None:
            raw = default_token_for_apply
        else:
            raw = raw_in

        # Exit/Validierung über helper (not_time_suffix bleibt für Rückwärtskompatibilität erhalten)
        chk = he.is_valid_time(raw, not_time_suffix=suffix, bare_means_time=False)
        if (raw == "0") and (not exit_on_0):
            return True, 0
        elif chk is None:
            if loop:
                continue
            else:
                return None

        # Trailing Units entfernen (Zeit behält ':', Zahl wird normalisiert)
        token = he.strip_trailing_units(raw)

        # Für die Zeit-/Zahl-Erkennung: falls bekanntes Suffix dran hängt, vorher entfernen
        raw_for_detect = raw
        if tail_suf_regex is not None:
            raw_for_detect = tail_suf_regex.sub("", raw_for_detect)

        # Zeit oder Zahl entscheiden anhand des bereinigten Inputs
        is_time = (":" in raw_for_detect) or bool(
            re.fullmatch(r"^\s*[+-]?\d+(?:[.,]\d+)?\s*[sS]\s*$", raw_for_detect)
        )

        if is_time:
            # Zeit als String zurück, bereits bereinigt (',' → '.' im Sekunden-Teil)

            if token:
                return (True, token)
            co.print_fail(_("invalid_input"))
            if loop:
                continue
            else:
                return None

        # Zahl (mit/ohne Suffix(e))
        try:
            if return_raw_if_nonnumber:
                return (False, raw)
            num = float(token)
            return (False, num)
        except Exception:
            # Fallback: erlaubte Suffixe manuell entfernen und erneut versuchen
            t2 = raw
            if tail_suf_regex is not None:
                t2 = tail_suf_regex.sub("", t2)
            t2 = he.strip_trailing_units(t2).replace(",", ".").strip()
            try:
                num = float(t2)
                return (False, num)
            except Exception:
                co.print_fail(_("invalid_input"))
                if loop:
                    continue
                else:
                    return None


def pick_preset(
    default: Optional[str] = None, back_button: Optional[bool] = True
) -> str | None:
    preset_keys = list(defin.CONVERT_PRESET)

    if default:
        def_index = preset_keys.index(default) if default in preset_keys else 3
    else:
        def_index = 3

    preset_descriptions = [
        str(tr(defin.CONVERT_PRESET[k].get("description") or "")) for k in preset_keys
    ]
    preset_labels = [str(defin.CONVERT_PRESET[k].get("name", k)) for k in preset_keys]
    preset_choice = ask_user(
        _("choose_quality_preset"),
        preset_keys,
        preset_descriptions,
        def_index,
        preset_labels,
        back_button=back_button if back_button else True,
    )
    return preset_choice


def pick_format(
    default: Optional[str] = None, back_button: Optional[bool] = True
) -> str | None:
    # Format-Auswahl (inkl. Lossless-Badges)
    format_keys = list(defin.CONVERT_FORMAT_DESCRIPTIONS)
    if default:
        def_index = format_keys.index(default) if default in format_keys else 0
    else:
        def_index = 0
    format_descriptions = [
        str(tr(defin.CONVERT_FORMAT_DESCRIPTIONS[k].get("description") or ""))
        for k in format_keys
    ]
    format_labels = [
        str(tr(defin.CONVERT_FORMAT_DESCRIPTIONS[fk].get("name", fk)))
        for fk in format_keys
    ]

    format_choice = ask_user(
        _("choose_format"),
        format_keys,
        format_descriptions,
        def_index,
        format_labels,
        back_button=back_button if back_button else True,
    )
    return format_choice


def pick_codec(
    container: str, default: Optional[str] = None, back_button: Optional[bool] = True
) -> Optional[str]:
    """
    Zeigt eine Codec-Auswahl passend zum Container an und liefert den gewählten Codec-Key
    (z. B. 'h264', 'hevc', 'av1', 'vp9', 'ffv1'). Gibt None zurück, wenn der Nutzer abbricht.

    Funktioniert mit der *alten* prepare_encoder_maps(files, format_choice, ...):
    - Wir übergeben files=[], format_choice=<container>, dadurch sind 'files' nicht erforderlich.
    """
    c = (container or "").lower()
    if c not in defin.CONVERT_FORMAT_DESCRIPTIONS:
        return None

    # 1) Encoder-Map pro Container über die alte prepare_encoder_maps holen
    #    format_choice != 'keep' ⇒ files werden ignoriert
    try:
        encoder_maps_by_container = vec.prepare_encoder_maps(
            files=[],  # wird ignoriert, weil format_choice != 'keep'
            format_choice=c,
            convert_format_descriptions=defin.CONVERT_FORMAT_DESCRIPTIONS,
            prefer_hw=True,
            ffmpeg_bin="ffmpeg",
        )
    except Exception:
        encoder_maps_by_container = {}

    emap = encoder_maps_by_container.get(c, {})  # Mapping codec_key -> encoder_name
    # 2) Kandidaten-Codec-Keys bestimmen (Fallback auf defs, falls Map leer ist)
    if emap:
        encoders_codec_keys: List[str] = list(emap.keys())
    else:
        # Fallback: definierte Codecs für den Container (ohne 'copy')
        encoders_codec_keys = [
            k
            for k in (defin.CONVERT_FORMAT_DESCRIPTIONS.get(c, {}).get("codecs") or [])
            if k != "copy"
        ]

    if not encoders_codec_keys:
        return None

    # 3) Labels & Descriptions aus defin.VIDEO_CODECS
    codec_labels = [
        str(defin.VIDEO_CODECS.get(k, {}).get("name", k)) for k in encoders_codec_keys
    ]
    # Wenn du tr() verwendest, nimm die nächste Zeile; ansonsten einfach ohne tr:
    try:
        codec_descriptions = [
            str(tr(defin.VIDEO_CODECS.get(k, {}).get("description") or ""))
            for k in encoders_codec_keys
        ]  # type: ignore[name-defined]
    except Exception:
        codec_descriptions = [
            str(defin.VIDEO_CODECS.get(k, {}).get("description", ""))
            for k in encoders_codec_keys
        ]

    # 4) Default-Index wählen
    def_order = ("h264", "hevc", "av1", "vp9", "ffv1")
    if default and default in encoders_codec_keys:
        def_idx = encoders_codec_keys.index(default)
    else:
        # bevorzugte Defaults, falls vorhanden
        def_idx = 0
        for pref in def_order:
            if pref in encoders_codec_keys:
                def_idx = encoders_codec_keys.index(pref)
                break

    # 5) UI
    ans = ask_user(
        (
            _("choose_codec_for_format").format(format=c.upper())
            if "_" in globals()
            else f"Choose codec for {c.upper()}"
        ),
        encoders_codec_keys,  # Rückgabewerte
        codec_descriptions,  # Beschreibungen
        def_idx,
        codec_labels,  # Labels
        back_button=bool(back_button),
    )

    return ans


# ==========================================================================================
# ======================================= PREVIEWS =========================================
# ==========================================================================================


def preview_and_confirm(
    sample_path: Path, filter_chain: Optional[str], method: str, value: Any
) -> Optional[bool]:
    with tempfile.TemporaryDirectory(prefix="preview_") as tmp:
        term_cols = shutil.get_terminal_size((80, 20)).columns
        min_width = 1280 // 22
        multi_mode = displayer.is_image_terminal() and term_cols > min_width

        time_points: List[float] = [1 / 3, 1 / 2] if multi_mode else [0.5]
        video_duration = he.get_duration_seconds(sample_path) or 0.0

        while True:
            montages: List[Path] = []

            print(" ")
            co.print_line(
                "  " + _("processing_filters_creating_image"),
                color="gold",
                style="italic",
            )

            for idx, pos in enumerate(time_points):
                orig_img = Path(tmp) / f"{sample_path.stem}_orig_{idx}.png"
                prev_img = Path(tmp) / f"{sample_path.stem}_preview_{idx}.png"
                try:
                    # Schnelle Seek-Variante:
                    #  - nutzt bereits bekannte video_duration
                    #  - precise=False → -ss vor -i, kein autotune_final_cmd
                    ip.extract_frame(
                        sample_path,
                        None,
                        orig_img,
                        pos,
                        duration=video_duration,
                        precise=False,
                    )
                    ip.extract_frame(
                        sample_path,
                        filter_chain,
                        prev_img,
                        pos,
                        duration=video_duration,
                        precise=False,
                    )
                except subprocess.CalledProcessError:
                    co.print_error(_("frame_extract_error"))
                    return False

                label_time = he.format_time(
                    video_duration * pos if video_duration > 0 else 0.0
                )
                labeled_orig = Path(tmp) / f"{sample_path.stem}_labeled_orig_{idx}.png"
                labeled_prev = (
                    Path(tmp) / f"{sample_path.stem}_labeled_preview_{idx}.png"
                )
                ip.make_labeled(orig_img, f"{label_time} ORIGINAL", labeled_orig)
                ip.make_labeled(prev_img, f"{label_time} PROCESSED", labeled_prev)
                montage_img = Path(tmp) / f"{sample_path.stem}_montage_{idx}.png"
                ip.montage(labeled_orig, labeled_prev, montage_img)
                montages.append(montage_img)

            co.delete_last_n_lines(2)
            print(" ")
            if multi_mode and len(montages) == 2:
                displayer.show_image(str(montages[0]))
                displayer.show_image(str(montages[1]))
            else:
                displayer.show_image(str(montages[0]))

            if method == "Preset":
                co.print_preset_params_table(str(value))
            else:
                print(" ")
                co.print_value_info(str(method), str(value))
                print(" ")

            preoptions = [_("accept"), _("reject"), _("new_preview")]
            answer = ask_user(_("apply_settings"), preoptions)
            if answer is None:
                return None

            if answer == preoptions[0]:
                return True
            elif answer == preoptions[1]:
                return False
            elif answer == preoptions[2]:
                time_points = _get_time_input(time_points)
            else:
                co.print_fail(_("invalid_input"))


# --- Hilfen -----------------------------------------------------------------


def _insert_cut_after_input(
    tokens: list[str], t_start: float, t_dur: float
) -> list[str]:
    """
    Fügt `-ss/-t` *nach* dem Input ein (Decode-Seeking). Stabiler für MKV+Subs/Attachments.
    """
    cmd = list(tokens)
    try:
        i_in = cmd.index("-i")
        # nach dem Eingabepfad einfügen
        insert_pos = min(len(cmd), i_in + 2)
        cmd[insert_pos:insert_pos] = ["-ss", f"{t_start:.3f}", "-t", f"{t_dur:.3f}"]
    except ValueError:
        # Falls wider Erwarten kein -i enthalten ist, vorne einfügen
        cmd[1:1] = ["-ss", f"{t_start:.3f}", "-t", f"{t_dur:.3f}"]
    return cmd


def _grab_any_frame(video_path: Path, t_rel: float, dur: float, out_img: Path) -> bool:
    # robustes Greifen: Wunschzeit → 0.0 → Mitte/Ende
    return (
        ip.grab_frame_at(video_path, t_rel, out_img)
        or ip.grab_frame_at(video_path, 0.0, out_img)
        or ip.grab_frame_at(video_path, max(0.0, min(dur - 0.05, 0.5 * dur)), out_img)
    )


# --------------------- MAIN PREVIEW ---------------------


def preview_and_confirm_compress(
    sample_path: Path, crf_value: float, codec: str = "libx264", preset: str = "slow"
) -> bool | None:
    """
    ORIGINAL vs. PROZESSIERT – Vorschau baut den *gleichen* Transcode-Plan wie `compress`,
    erzeugt daraus kurze Snippets (ohne Maps zu verändern) und greift Frames daraus ab.
    True  -> akzeptiert
    False -> verworfen
    None  -> abgebrochen
    """

    # CRF -> Qualitäts-% (18 → 100, 30 → 0)
    try:
        quality_percent: Optional[int] = max(
            0, min(100, round((30.0 - float(crf_value)) / 12.0 * 100.0))
        )
    except Exception:
        quality_percent = None

    with tempfile.TemporaryDirectory(prefix="preview_") as tmpdir:
        tmp = Path(tmpdir)

        term_cols = shutil.get_terminal_size((80, 20)).columns
        min_width = 1280 // 22
        multi_mode = displayer.is_image_terminal() and term_cols > min_width

        duration = he.get_duration_seconds(sample_path) or 0.0
        time_points = [1 / 3, 1 / 2] if (multi_mode and duration > 0.0) else [0.5]
        stem = sample_path.stem

        # ---- Container + Ziel-Codec wie im Haupt-'compress' bestimmen ----------
        src_container = (vec.detect_container_from_path(sample_path) or "").lower()
        if not src_container:
            src_container = sample_path.suffix.lower().lstrip(".") or "mkv"

        # Der UI-Flow übergibt 'codec' i. d. R. bereits als Encoder-Name (z. B. 'libx264').
        # Wir normalisieren auf unseren Codec-Key und wählen wie in 'compress' ggf. um.
        desired_codec_key_orig = vec.normalize_codec_key(codec) or "h264"

        # CRF-/qscale-fähigen Ziel-Codec je Container wählen (gleiches Verhalten wie 'compress')
        try:
            desired_codec_key = vec.pick_crf_codec_for_container(
                src_container, desired_codec_key_orig
            )
        except Exception:
            # Fallback: wenn die Kombi unzulässig ist, auf Container-Default zurückfallen
            if not vec.container_allows_codec(src_container, desired_codec_key_orig):
                desired_codec_key = vec.suggest_codec_for_container(src_container)
            else:
                desired_codec_key = desired_codec_key_orig

        # Dateiendung für Snippets passend zum Container
        ext_map = {
            "mp4": "mp4",
            "m4v": "m4v",
            "mkv": "mkv",
            "matroska": "mkv",
            "webm": "webm",
            "mov": "mov",
            "avi": "avi",
            "mpeg": "mpg",
            "mpegps": "mpg",
            "mpegts": "ts",
            "flv": "flv",
        }
        src_ext = sample_path.suffix.lower().lstrip(".")
        out_ext = ext_map.get(src_container, (src_ext or "mkv"))

        # ---- Preset wie im Haupt-'compress' patchen ----------------------------
        _preset_name = "casual"
        _store = defin.CONVERT_PRESET.get(_preset_name, {})
        _backup = copy.deepcopy(_store)

        try:
            patched = dict(_store)
            patched["quality"] = int(crf_value)
            patched["speed"] = str(preset or "slow")
            if src_container in ("mp4", "m4v"):
                patched["faststart"] = True
            defin.CONVERT_PRESET[_preset_name] = patched  # type: ignore[assignment]

            # Wir wollen einen echten Encode (kein Stream-Copy), Verhalten wie in 'compress'
            _preset_max_fps_noop = 9999

            while True:
                montages: List[Path] = []
                total_snippets = 0
                successful_snippets = 0

                for idx, pos_frac in enumerate(time_points):
                    total_snippets += 1

                    t_target = (duration * pos_frac) if duration > 0 else 0.0
                    dur_snip = 1.0 if duration >= 1.0 else max(0.2, duration or 0.2)
                    half = dur_snip / 2.0
                    t_start = max(
                        0.0,
                        min(max(t_target - half, 0.0), max(duration - dur_snip, 0.0)),
                    )
                    t_rel = max(0.0, min(t_target - t_start, max(dur_snip - 1e-3, 0.0)))

                    # ORIGINAL-Frame
                    orig_img = tmp / f"{stem}_orig_{idx}.jpg"
                    if not ip.grab_frame_at(sample_path, t_target, orig_img):
                        # Wenn das Original-Grab schon scheitert, abbrechen und Entscheidung einholen
                        co.print_error(_("frame_not_created"))
                        opts = [_("accept"), _("reject")]
                        ans = ask_user(_("apply_settings"), opts, back_button=False)
                        if ans is None:
                            return None
                        return ans is opts[0]

                    # --- PROCESSED-Snippet: exakt wie 'compress' den Plan bauen -------
                    snippet = tmp / f"{stem}_snippet_{idx}.{out_ext}"
                    snippet_ok = False
                    try:
                        plan = vec.build_transcode_plan(
                            input_path=sample_path,
                            target_container=src_container,
                            preset_name=_preset_name,
                            codec_key=desired_codec_key,
                            preferred_encoder=None,  # Encoderwahl regelt vec wie in 'compress'
                            req_scale=None,
                            src_w=None,
                            src_h=None,
                            src_fps=None,
                            user_fps_rational=None,
                            preset_max_fps=_preset_max_fps_noop,
                            force_key_at_start=True,
                        )
                        final_cmd = list(plan.final_cmd_without_output)
                        final_cmd = vec.postprocess_cmd_all_presets(final_cmd, plan)

                        # 🎯 Kurz-Snippet: -ss/-t NACH dem Input (stabile MKV-Subs/Attachments)
                        final_cmd = _insert_cut_after_input(
                            final_cmd, t_start, dur_snip
                        )

                        # Ziel anhängen und laufen lassen
                        final_cmd.append(str(snippet))
                        rc = subprocess.call(
                            final_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        snippet_ok = (
                            rc == 0 and snippet.exists() and snippet.stat().st_size > 0
                        )
                    except Exception:
                        snippet_ok = False

                    prev_img = tmp / f"{stem}_prev_{idx}.jpg"
                    if snippet_ok:
                        successful_snippets += 1
                        if not _grab_any_frame(snippet, t_rel, dur_snip, prev_img):
                            # selbst wenn der Frame-Grab misslingt → für Anzeige duplizieren
                            shutil.copyfile(orig_img, prev_img)
                    else:
                        # kein Snippet – trotzdem etwas anzeigen
                        shutil.copyfile(orig_img, prev_img)

                    # Labeln & Montage (Label *immer* QUALITY/PROCESSED, nie "Preview failed")
                    label_time = he.format_time(t_target) if duration > 0 else "n/a"
                    labeled_o = tmp / f"{stem}_labeled_o_{idx}.jpg"
                    labeled_p = tmp / f"{stem}_labeled_p_{idx}.jpg"
                    ip.make_labeled(orig_img, f"{label_time} ORIGINAL", labeled_o)
                    if quality_percent is not None:
                        ip.make_labeled(
                            prev_img,
                            f"{label_time} QUALITY {quality_percent}%",
                            labeled_p,
                        )
                    else:
                        ip.make_labeled(prev_img, f"{label_time} PROCESSED", labeled_p)
                    montage = tmp / f"{stem}_montage_{idx}.jpg"
                    ip.montage(labeled_o, labeled_p, montage)
                    montages.append(montage)

                # Anzeige
                print(" ")
                if multi_mode and len(montages) >= 2:
                    displayer.show_image(str(montages[0]))
                    displayer.show_image(str(montages[1]))
                else:
                    displayer.show_image(str(montages[0]))

                print(" ")
                if quality_percent is not None:
                    co.print_value_info("Quality", f"{quality_percent}%")
                co.print_value_info("Codec", desired_codec_key.upper())
                co.print_value_info("Container", src_container.upper())

                # Warnung NUR, wenn *kein einziges* Snippet gebaut werden konnte.
                if total_snippets > 0 and successful_snippets == 0:
                    co.print_warning(_("preview_encode_error"))

                print(" ")
                opts = [_("accept"), _("reject"), _("new_preview")]
                ans = ask_user(_("apply_settings"), opts, back_button=False)
                if ans is None:
                    return None
                if ans is opts[0]:
                    return True
                if ans is opts[1]:
                    return False
                if ans is opts[2]:
                    entered = _get_time_input(time_points)
                    norm: List[float] = []
                    for p in entered:
                        try:
                            x = float(p)
                            x = x / 100.0 if x > 1.0 else x
                            norm.append(min(max(x, 0.0), 0.999))
                        except Exception:
                            continue
                    if norm:
                        time_points = norm
                else:
                    co.print_fail(_("invalid_input"))

        finally:
            try:
                defin.CONVERT_PRESET[_preset_name] = _backup  # type: ignore[assignment]
            except Exception:
                pass


def _preview_enabled() -> bool:
    return (
        os.environ.get("VIDEO_DISABLE_IMAGES", "0") != "1"
        and os.environ.get("VIDEO_IMG_DRIVER", "auto").lower() != "none"
    )


def preview_trim_range(
    target: Path | Sequence[Path], start_str: str, duration_str: str
) -> None:
    """
    Zeigt erstes & (nahezu) letztes Frame des Trim-Bereichs.
    Respektiert VIDEO_DISABLE_IMAGES/VIDEO_IMG_DRIVER und begrenzt Label-Länge.
    """
    if not _preview_enabled():
        co.print_info(_("preview_disabled_no_inline"))
        return

    # Normalisieren
    if isinstance(target, (str, Path)):
        files = [Path(target)]
    else:
        files = [Path(p) for p in target]

    try:
        term_cols = shutil.get_terminal_size((80, 20)).columns
        min_width = 1280 // 22
        multi_mode = displayer.is_image_terminal() and term_cols > min_width

        with tempfile.TemporaryDirectory(prefix="preview_trim_") as tmpdir:
            tmp = Path(tmpdir)
            for fpath in files:
                duration_total = he.get_duration_seconds(fpath) or 0.0
                start_sec = he.to_seconds(start_str)
                dur_sec = he.to_seconds(duration_str)
                end_sec = min(start_sec + dur_sec, duration_total)

                eps = 0.040
                end_frame_t = max(
                    0.0, min(end_sec - eps, max(duration_total - eps, 0.0))
                )

                start_img = tmp / f"{fpath.stem}_start.jpg"
                end_img = tmp / f"{fpath.stem}_end.jpg"
                if not ip.grab_frame_at(fpath, start_sec, start_img):
                    co.print_warning(_("frame_not_created"))
                    continue
                if not ip.grab_frame_at(fpath, end_frame_t, end_img):
                    co.print_warning(_("frame_not_created"))
                    continue

                # Label kürzen → verhindert ausufernde Titel in Kitty
                def _short(s: str, lim: int = 80) -> str:
                    s = str(s)
                    return s if len(s) <= lim else (s[: lim - 1] + "…")

                lbl_start = tmp / f"{fpath.stem}_start_labeled.jpg"
                lbl_end = tmp / f"{fpath.stem}_end_labeled.jpg"
                ip.make_labeled(
                    start_img,
                    f"START {he.format_time(start_sec)}\n{_short(fpath.name)}",
                    lbl_start,
                )
                ip.make_labeled(
                    end_img,
                    f"END   {he.format_time(end_sec)}\n{_short(fpath.name)}",
                    lbl_end,
                )

                print(" ")
                if multi_mode:
                    combo = tmp / f"{fpath.stem}_montage.jpg"
                    ip.montage(lbl_start, lbl_end, combo)
                    displayer.show_image(str(combo))
                else:
                    displayer.show_image(str(lbl_start))
                    displayer.show_image(str(lbl_end))
    except Exception as ex:
        co.print_warning(f"Preview failed: {ex}")


def select_files_interactively(*extension_groups: Sequence[str]) -> List[str]:
    """
    Interaktive Auswahl:
      - Einzelne Dateien im aktuellen Ordner
      - Zusätzlich: ganze Unterordner (1 Ebene), wenn sie passende Dateien enthalten.
    Rückgabe immer: Liste von *Dateipfaden* (Ordner werden transparent in Dateien expandiert).
    """
    cwd = Path.cwd()

    # --- WICHTIG: Endungen robust auf ".ext" normalisieren ---
    def _norm_ext(e: str) -> str:
        s = str(e).strip().lower()
        return s if s.startswith(".") else f".{s}"

    if extension_groups and len(extension_groups) > 0:
        raw_exts = [ext for group in extension_groups for ext in group]
    else:
        raw_exts = list(getattr(defin, "VIDEO_EXTENSIONS", []))

    valid_exts = {_norm_ext(ext) for ext in raw_exts}

    # Dateien im aktuellen Ordner
    files_here: List[str] = [
        str(p)
        for p in sorted(cwd.iterdir())
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    # Unterordner (1 Ebene), nur wenn sie passende Dateien enthalten
    dir_entries: List[Tuple[Path, List[str]]] = []
    for d in sorted([x for x in cwd.iterdir() if x.is_dir()]):
        matches: List[str] = [
            str(child)
            for child in sorted(d.iterdir())
            if child.is_file() and child.suffix.lower() in valid_exts
        ]
        if matches:
            dir_entries.append((d, matches))

    if not files_here and not dir_entries:
        co.print_error(_("no_matching_file_found"))
        sys.exit(0)

    # Anzeige-Liste bauen (Index 1..N)
    items: List[Tuple[str, str]] = []  # ("file"| "dir", path)
    display_lines: List[str] = []

    idx = 1
    if dir_entries:
        for d, matches in dir_entries:
            items.append(("dir", str(d)))
            display_lines.append(
                f"[{idx:>2}] "
                + co.return_promt(
                    f"[DIR] {d.name}/  ({len(matches)} files)", color="mint"
                )
            )
            idx += 1
        print("")

    if files_here:
        co.print_promt(_("found_files") + ":")
        co.print_seperator()
        for f in files_here:
            items.append(("file", f))
            display_lines.append(
                f"[{idx:>2}] "
                + co.return_promt(f"[FILE] {Path(f).name}", color="soft_blue")
            )
            idx += 1
        print("")

    # Ausgabe gesammelt zeigen (geordnet), plus Bedienhinweise
    for line in display_lines:
        co.print_line("  " + line)
    co.print_exit(2)
    co.print_line("  " + _("enter_numbers_to_select_files"), color="light_yellow")
    print("")

    while True:
        user_input = input("  > ").strip()
        print("")
        if user_input == "0":
            sys.exit(0)

        indices = fs.parse_file_selection(user_input, len(items))  # ENTER ⇒ alle

        if indices and all(0 <= i < len(items) for i in indices):
            # Auswahl expandieren (Ordner -> Dateien)
            result: List[str] = []
            seen: set[str] = set()
            for i in indices:
                kind, path = items[i]
                if kind == "file":
                    if path not in seen:
                        result.append(path)
                        seen.add(path)
                else:
                    # Ordner: passende Dateien (1 Ebene) sammeln
                    d = Path(path)
                    for child in sorted(d.iterdir()):
                        if child.is_file() and child.suffix.lower() in valid_exts:
                            sp = str(child)
                            if sp not in seen:
                                result.append(sp)
                                seen.add(sp)

            if result:
                co.print_promt(_("selected_files") + ":")
                for p in result:
                    co.print_line(
                        f"  - {Path(p).name}", color="soft_blue", style="bold"
                    )
                print(" ")
                return result

            co.print_error(_("no_matching_file_found") + "\n")
        else:
            # (Nur als Fallback; eigentlich liefert parse_file_selection bei ENTER schon alle.)
            if not user_input:
                indices = list(range(len(items)))
                continue
            co.print_error(_("invalid_input") + "\n")


def ask_for_name(promt: Optional[str] = None, default: Optional[str] = None) -> str:
    quest = promt if promt else _("enter_name")
    stand = default if default else "title"

    inputstr = "   " + quest + f" [{stand}]: "
    uinput = input(co.return_promt(inputstr)).strip()

    if not uinput:
        return stand
    else:
        return uinput


def ask_for_filename(
    original_file: Union[str, Path],
    default: Optional[str] = None,
    promt: Optional[str] = None,
) -> str:
    """
    Fragt neuen Dateinamen ab und sorgt für gültige Video-Extension.
    """
    original_path = Path(original_file)
    original_ext = original_path.suffix
    video_exts = {ext.lower().lstrip(".") for ext in defin.VIDEO_EXTENSIONS}

    if default:
        default_name = default
    else:
        default_name = _("overwrite_existing_file")

    print()
    if promt:
        quest = promt
    else:
        quest = _("new_filename")

    user_input = ask_for_name(quest, default_name)

    if not user_input:
        if default:
            return default
        else:
            return original_path.name

    new_path = Path(user_input)
    user_ext = new_path.suffix.lstrip(".").lower()

    if user_ext in video_exts:
        return new_path.name
    else:
        return new_path.name + original_ext
