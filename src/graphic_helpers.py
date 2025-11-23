#!/usr/bin/env python3
# graphic_helpers
from __future__ import annotations

import sys
import threading
from typing import Optional

import definitions as defin
import helpers as he
import mem_guard as mg
from ANSIColor import ANSIColor
from i18n import _

c = ANSIColor()

_UI_LOCK = threading.Lock()
_LAST_LINES = 0
_UI_PHASE_ID_LOCK = threading.Lock()  # NEW
_UI_PHASE_ID = 0


def _move_up(n: int) -> None:
    if n > 0:
        sys.stdout.write(
            f"\x1b[{n}F"
        )  # Cursor n Zeilen nach oben (an den Zeilenanfang)


# --- robuster Output-Kanal + Lock (verhindert Zeilenflackern bei Parallelität)
_UI_LOCK = threading.Lock()


def draw_chunk_block(
    *, two_bars: bool, title: str, top_bar: str | None, bot_bar: str, hint: str
):
    global _LAST_LINES
    lines = [f"   {title}"]
    if two_bars and top_bar is not None:
        lines.append(top_bar)
    lines.append(bot_bar)
    lines.append(_("cancel_hint"))
    block = "\n".join(lines)

    with _UI_LOCK:
        if sys.stdout.isatty():
            _move_up(_LAST_LINES)  # alten Block übermalen
            sys.stdout.write("\x1b[J")  # nach unten löschen
            sys.stdout.write(block + "\n")
            sys.stdout.flush()
            _LAST_LINES = len(lines)
        else:
            # Nicht-TTY: seltener drucken, kein Überschreiben möglich
            sys.stdout.write(block + "\n")
            sys.stdout.flush()


def ui_new_phase() -> int:
    global _UI_PHASE_ID
    with _UI_PHASE_ID_LOCK:
        _UI_PHASE_ID += 1
        return _UI_PHASE_ID


def ui_phase_active(phase_id: Optional[int]) -> bool:
    if not phase_id:
        return True
    with _UI_PHASE_ID_LOCK:
        return phase_id == _UI_PHASE_ID and not mg.is_cancelled()


def draw_chunk_block_cond(
    *,
    two_bars: bool,
    title: str,
    top_bar,
    bot_bar,
    hint: str,
    ui_phase_id: Optional[int],
) -> None:
    if not ui_phase_active(ui_phase_id):
        return
    draw_chunk_block(
        two_bars=two_bars, title=title, top_bar=top_bar, bot_bar=bot_bar, hint=hint
    )


def make_bar(progress: float, bar_len: int) -> tuple[str, str]:
    progress = max(0.0, min(1.0, progress))
    filled = int(bar_len * progress)

    # 1) Truecolor
    if getattr(c, "supports_8", False) and getattr(c, "mode", "") == "truecolor":
        r, g, b = he.gradient_colour(progress)
        colour = f"\033[38;2;{r};{g};{b}m"
    # 2) 256-Farben
    elif getattr(c, "supports_256", False):
        start, end = 196, 46
        code = int(start + (end - start) * progress)
        colour = f"\033[38;5;{code}m"
    # 3) 16-Farben
    elif getattr(c, "supports_16", False):
        name = "green" if progress > 0.5 else "red"
        colour = c.get(name)
    else:
        colour = ""

    inner = f"{colour}{'█' * filled}{c.get('reset')}{' ' * (bar_len - filled)}"
    return f"[{inner}] {progress * 100:3.0f}%", colour


def print_progress_block(
    lines: list[str], bar_line: str, hint_line: str | None = None
) -> None:
    """
    Overwrite a block:
      N text lines
      + 1 bar line
      + optional 1 hint line
    located above the cursor.
    """
    extra = 1 if hint_line is not None else 0
    num_lines = len(lines)
    sys.stdout.write(f"\033[{num_lines + 1 + extra}F")  # Move cursor up
    for line in lines:
        sys.stdout.write(defin.CLEAR_LINE + line + "\n")
    sys.stdout.write(defin.CLEAR_LINE + bar_line + "\n")
    if hint_line is not None:
        sys.stdout.write(defin.CLEAR_LINE + hint_line + "\n")
    sys.stdout.flush()
