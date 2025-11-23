#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from typing import ClassVar, Dict


class ANSIColor:
    """
    ANSIColor: Terminalfarben & -styles mit Fallback, 8/16/256-Farben-Unterstützung
    """

    _ATTRIBUTES: ClassVar[Dict[str, str]] = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "inverse": "\033[7m",
        "hidden": "\033[8m",
        "strike": "\033[9m",
    }

    _STD_COLORS: ClassVar[Dict[str, int]] = {
        "black": 0,
        "red": 1,
        "green": 2,
        "yellow": 3,
        "blue": 4,
        "magenta": 5,
        "cyan": 6,
        "white": 7,
    }

    # 256 Colors – erweiterte Palette (>40 Farben), viele gängige Namen
    _COLORS_256: ClassVar[Dict[str, int]] = {
        # Rot/Orange/Gelb
        "red": 196,
        "soft_red": 203,
        "light_red": 210,
        "rose": 204,
        "orange": 208,
        "bright_orange": 214,
        "deep_orange": 202,
        "gold": 220,
        "yellow": 226,
        "light_yellow": 228,
        "khaki": 222,
        "peach": 217,
        # Grün-Töne
        "green": 46,
        "bright_green": 82,
        "light_green": 120,
        "mint": 121,
        "olive": 100,
        "teal": 37,
        "spring_green": 48,
        "lime": 118,
        "sea_green": 29,
        "emerald": 40,
        "dark_green": 22,
        # Blau-Töne
        "blue": 21,
        "soft_blue": 75,
        "bright_blue": 117,
        "deep_blue": 20,
        "cyan": 51,
        "aqua": 45,
        "turquoise": 44,
        "sky_blue": 111,
        "navy": 18,
        "indigo": 54,
        "azure": 39,
        # Lila/Violett-Töne
        "purple": 141,
        "deep_purple": 56,
        "violet": 99,
        "magenta": 201,
        "pink": 213,
        "fuchsia": 13,
        "orchid": 170,
        "lavender": 183,
        # Grau/Braun/Weiß
        "grey": 244,
        "light_grey": 250,
        "dark_grey": 239,
        "charcoal": 237,
        "silver": 7,
        "white": 15,
        "off_white": 255,
        "brown": 130,
        "tan": 180,
        "beige": 230,
        "maroon": 88,
        # Extra/Fun
        "salmon": 216,
        "apricot": 216,
        "plum": 134,
        "sand": 180,
        "slate": 66,
        "forest_green": 28,
        "mustard": 143,
        "turquoise2": 80,
        "pearl": 254,
        "eggplant": 90,
        "rosewood": 131,
    }

    # Fallback-Mapping: 256er-Farben → Standardfarbe
    _FALLBACK_256_TO_STD: ClassVar[Dict[str, str]] = {
        # ROT
        "red": "red",
        "soft_red": "red",
        "light_red": "red",
        "rose": "red",
        # ORANGE
        "orange": "yellow",
        "bright_orange": "yellow",
        "deep_orange": "yellow",
        "gold": "yellow",
        "peach": "yellow",
        # GELB
        "yellow": "yellow",
        "light_yellow": "yellow",
        "khaki": "yellow",
        # GRÜN
        "green": "green",
        "bright_green": "green",
        "light_green": "green",
        "mint": "green",
        "lime": "green",
        "olive": "green",
        "spring_green": "green",
        "emerald": "green",
        "sea_green": "green",
        "dark_green": "green",
        "forest_green": "green",
        # BLAU
        "blue": "blue",
        "soft_blue": "blue",
        "bright_blue": "blue",
        "deep_blue": "blue",
        "navy": "blue",
        "indigo": "blue",
        "azure": "blue",
        "cyan": "cyan",
        "aqua": "cyan",
        "turquoise": "cyan",
        "turquoise2": "cyan",
        "sky_blue": "cyan",
        # LILA/VIOLETT/PINK
        "purple": "magenta",
        "deep_purple": "magenta",
        "violet": "magenta",
        "magenta": "magenta",
        "orchid": "magenta",
        "lavender": "magenta",
        "pink": "magenta",
        "fuchsia": "magenta",
        "plum": "magenta",
        # BRAUN/BEIGE
        "brown": "yellow",
        "maroon": "red",
        "tan": "yellow",
        "beige": "white",
        "sand": "yellow",
        # GRAU/WEISS/SILBER
        "grey": "white",
        "light_grey": "white",
        "dark_grey": "black",
        "charcoal": "black",
        "silver": "white",
        "white": "white",
        "off_white": "white",
        "pearl": "white",
        # EXTRA
        "salmon": "red",
        "apricot": "yellow",
        "mustard": "yellow",
        "eggplant": "magenta",
        "slate": "blue",
        "rosewood": "red",
    }

    def __init__(self) -> None:
        self.mode: str = self._detect_color_mode()

    def _detect_color_mode(self) -> str:
        if sys.platform == "win32":
            term = os.environ.get("TERM", "")
            if "xterm" in term or "256" in term or os.environ.get("WT_SESSION"):
                return "256"
            return "16"
        term = os.environ.get("TERM", "")
        if "256color" in term:
            return "256"
        if term in ("xterm", "screen", "linux", "vt100"):
            return "16"
        if "color" in term:
            return "16"
        if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
            return "truecolor"
        return "8"

    def get(self, color: str, background: bool = False) -> str:
        """
        Gibt den passenden ANSI-Code für eine Farbe (mit Fallback) zurück.
        color: Farbnamen-String, z.B. 'orange', 'khaki', 'slate'
        background: Wenn True → Hintergrundfarbe
        """
        if not color:
            return ""
        color = color.lower()
        code: str

        # 256 Farben werden nur genommen, wenn Terminal es kann und Farbe existiert
        if self.mode in ("256", "truecolor") and color in self._COLORS_256:
            n = self._COLORS_256[color]
            code = f"\033[{48 if background else 38};5;{n}m"
        elif color in self._COLORS_256:
            # Fallback-Logik
            std_fallback = self._FALLBACK_256_TO_STD.get(color, "white")
            return self.get(std_fallback, background=background)
        elif color.startswith("bright_") and color[7:] in self._STD_COLORS:
            base = 90 + self._STD_COLORS[color[7:]]
            code = f"\033[{base + (10 if background else 0)}m"
        elif color in self._STD_COLORS:
            base = 30 + self._STD_COLORS[color]
            code = f"\033[{base + (10 if background else 0)}m"
        elif color == "reset":
            code = self._ATTRIBUTES["reset"]
        elif color in self._ATTRIBUTES:
            code = self._ATTRIBUTES[color]
        else:
            raise ValueError(f"Unknown color/style: '{color}'")
        return code

    def style(self, *args: str) -> str:
        """Kombiniert beliebig viele Styles/Farben zu einer Sequenz."""
        return "".join(self.get(arg) for arg in args if arg)

    def combine(self, *args: str) -> str:
        """Alias für .style() – für Konsistenz."""
        return self.style(*args)

    def demo(self) -> None:
        print(f"=== Demo: Styles/Colors (detected: {self.mode}) ===")
        for style in self._ATTRIBUTES:
            print(
                self.get(style) + style.ljust(14) + self._ATTRIBUTES["reset"], end="  "
            )
        print("\n\nStandardfarben:")
        for bright in (False, True):
            for name in self._STD_COLORS:
                colname = ("bright_" if bright else "") + name
                fg = self.get(colname)
                print(fg + colname.ljust(14) + self._ATTRIBUTES["reset"], end="  ")
            print()
        if self.mode == "256":
            print("\n256-Farben-Beispiele:")
            for name in sorted(self._COLORS_256, key=lambda n: self._COLORS_256[n]):
                fg = self.get(name)
                print(fg + name.ljust(14) + self._ATTRIBUTES["reset"], end="  ")
                if (self._COLORS_256[name] % 10) == 0:
                    print()
            print()

    @property
    def supports_256(self) -> bool:
        return self.mode == "256"

    @property
    def supports_16(self) -> bool:
        return self.mode in ("16", "256", "truecolor")

    @property
    def supports_8(self) -> bool:
        return True
