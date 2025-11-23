#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from typing import Any, Callable, Optional, Tuple

import consoleOutput as co  # optionales Projekt-Logging
from loghandler import print_log

try:
    from PIL import Image  # conda/pip installiert
except Exception:
    Image = None  # type: ignore


def _pil_lanczos() -> Any:
    """Kompatibler LANCZOS/ANTIALIAS-Filter für unterschiedliche Pillow-Versionen."""
    if Image is None:
        return None
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None and hasattr(resampling, "LANCZOS"):
        return getattr(resampling, "LANCZOS")
    return getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", None))


class ImageDisplayer:
    """
    Driver ladder (wie früher, aber robuster):
      1) Kitty/WezTerm → zuerst: kitty +kitten icat --transfer-mode=stream (ohne --place)
         Fallback: rohes Kitty Graphics Protocol (t=d) ohne Kitten
      2) iTerm2 (OSC 1337)
      3) chafa (ASCII/Unicode)
      4) viu (sixel/kitty & ASCII)
      5) extern (xdg-open/open/start)

    Steuerung:
      - VIDEO_IMG_DRIVER = auto|kitty|chafa|viu|none  (Default: auto)
      - VIDEO_DISABLE_IMAGES=1  → aus
      - VIDEO_IMG_COLS / VIDEO_IMG_ROWS → Größe für chafa/viu
    """

    def __init__(
        self, auto_resize: bool = True, max_size: Tuple[int, int] = (2048, 2048)
    ) -> None:
        self.image_id: int = 1
        self.auto_resize = auto_resize
        self.max_size = max_size
        self.driver = self._select_driver()
        try:
            if hasattr(co, "print_log"):
                print_log(f"[ImageDisplayer] driver={self.driver}")
        except Exception:
            pass

    # ───────────── Terminal-Erkennung ─────────────
    @staticmethod
    def _term_is_kitty() -> bool:
        return (
            os.environ.get("TERM", "").lower() == "xterm-kitty"
            or "KITTY_WINDOW_ID" in os.environ
        )

    @staticmethod
    def _term_is_wezterm() -> bool:
        t = os.environ.get("TERM", "").lower()
        return t == "wezterm" or "WEZTERM_EXECUTABLE" in os.environ

    @staticmethod
    def _term_is_iterm2() -> bool:
        return "ITERM_SESSION_ID" in os.environ

    def _kitty_usable(self) -> bool:
        return self._term_is_kitty() or self._term_is_wezterm()

    def _select_driver(self) -> str:
        if os.environ.get("VIDEO_DISABLE_IMAGES") == "1":
            return "none"
        want = os.environ.get("VIDEO_IMG_DRIVER", "auto").lower()
        if want in {"none", "off"}:
            return "none"
        if want == "kitty":
            return "kitty" if self._kitty_usable() else "none"
        if want == "chafa":
            return "chafa" if shutil.which("chafa") else "none"
        if want == "viu":
            return "viu" if shutil.which("viu") else "none"
        # auto
        if self._kitty_usable():
            return "kitty"
        if shutil.which("chafa"):
            return "chafa"
        if shutil.which("viu"):
            return "viu"
        return "none"

    # ───────────── Helpers ─────────────
    @staticmethod
    def _open_external(path: str) -> None:
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", path], check=False)
            elif os.name == "nt":
                try:
                    os.startfile(os.path.normpath(path))  # type: ignore[attr-defined]
                except Exception:
                    subprocess.run(
                        [
                            "powershell",
                            "-NoProfile",
                            "-Command",
                            f"Start-Process -FilePath '{path}'",
                        ],
                        check=False,
                    )
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            print(f"[preview skipped] external viewer failed: {e}")

    def _maybe_resize(self, image_path: str) -> str:
        """Optional: Bild verkleinern, wenn > max_size (wie früher)."""
        if not self.auto_resize or Image is None:
            return image_path
        try:
            with Image.open(image_path) as img:  # type: ignore
                w, h = img.size
                mw, mh = self.max_size
                if w > mw or h > mh:
                    out = image_path + ".resized_for_terminal.png"
                    resample = _pil_lanczos()
                    if resample is None:
                        img.thumbnail((mw, mh))
                    else:
                        img.thumbnail((mw, mh), resample=resample)  # type: ignore
                    img.save(out, "PNG")
                    return out
        except Exception as e:
            print(f"[preview skipped] resize failed: {e}")
        return image_path

    # ───────────── Kitty-Anzeigen: Plain icat (wie alte Version) ─────────────
    def _show_kitty_icat_plain(self, path: str) -> bool:
        """
        1. Wahl für Kitty/WezTerm: identisch zur alten Version
        → keine Platzierung, keine Bildschirm-Manipulation, einfach inline rendern.
        """
        if not shutil.which("kitty"):
            return False
        env = os.environ.copy()
        # Embedded-Python-Konflikte vermeiden (Kitten nutzt eigene Python-Umgebung):
        for k in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
            env.pop(k, None)
        try:
            # --scale-up macht kleine Previews besser sichtbar (schadet nicht)
            subprocess.run(
                [
                    "kitty",
                    "+kitten",
                    "icat",
                    "--transfer-mode",
                    "stream",
                    "--scale-up",
                    str(path),
                ],
                check=True,
                env=env,
            )
            return True
        except Exception:
            return False

    # ───────────── Kitty-Fallback: Rohes Graphics Protocol ─────────────
    def _show_kitty_raw(
        self,
        image_path: str,
        image_id: Optional[int] = None,
        delete_after: bool = False,
    ) -> None:
        """
        Fallback ohne +kitten: sendet die Bilddaten direkt (t=d), robust gegen kaputtes Kitten-Python.
        """
        if image_id is None:
            image_id = self.image_id
        if not sys.stdout.isatty():
            print("[preview skipped] stdout is not a TTY (no terminal).")
            return

        img_path = self._maybe_resize(image_path)
        try:
            with open(img_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
        except Exception as e:
            print(f"[preview skipped] read failed: {e}")
            return

        write: Callable[[str], int] = sys.stdout.write
        flush: Callable[[], None] = sys.stdout.flush

        write("\n")  # Bild auf neue Zeile wie früher

        CHUNK = 4096
        total = len(b64)
        i = 0
        while i < total:
            part = b64[i : i + CHUNK]
            more = 1 if i + CHUNK < total else 0
            # a=T → transmit; t=d → Daten (nicht Dateiname); f=100%; q=2 → quiet
            header = (
                f"\x1b_Ga=T,t=d,f=100,i={image_id},m={more},q=2"
                + (f",s={total}" if i == 0 else "")
                + ";"
            )
            write(header + part + "\x1b\\")
            i += CHUNK
        flush()

        if delete_after:
            write(f"\x1b_Ga=d,i={image_id}\x1b\\")
            flush()

        if img_path != image_path and img_path.endswith(".resized_for_terminal.png"):
            try:
                os.remove(img_path)
            except Exception:
                pass

    # ───────────── Andere Renderer ─────────────
    def _show_iterm2(self, image_path: str) -> None:
        """Inline-Image via iTerm2 OSC 1337."""
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("ascii")
            # preserveAspectRatio=1 → hübscheres Verhalten bei schmalen Fenstern
            sys.stdout.write(
                f"\033]1337;File=inline=1;width=auto;height=auto;preserveAspectRatio=1:{img_data}\a"
            )
            sys.stdout.flush()
        except Exception as e:
            print(f"[preview skipped] iTerm2 failed: {e}")

    def _show_chafa(self, path: str) -> None:
        cols = os.environ.get("VIDEO_IMG_COLS", "80")
        rows = os.environ.get("VIDEO_IMG_ROWS", "24")
        try:
            subprocess.call(
                ["chafa", f"--size={cols}x{rows}", path],
                stdout=sys.stdout,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[preview skipped] chafa failed: {e}")

    def _show_viu(self, path: str) -> None:
        cols = os.environ.get("VIDEO_IMG_COLS", "80")
        rows = os.environ.get("VIDEO_IMG_ROWS", "24")
        try:
            subprocess.call(
                ["viu", "-w", cols, "-h", rows, path],
                stdout=sys.stdout,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[preview skipped] viu failed: {e}")

    # ───────────── Public API ─────────────
    def is_image_terminal(self) -> bool:
        return self.driver != "none" or self._term_is_iterm2()

    def show_image(self, path: str, delete_after: bool = False) -> None:
        """
        Genau wie früher:
        - Wenn Kitty/WezTerm: zuerst icat (plain, stream), bei Fehler → rohes KGP.
        - Wenn iTerm2: OSC 1337.
        - Sonst Fallbacks (chafa/viu/extern).
        """
        if self._kitty_usable():
            ok = self._show_kitty_icat_plain(path)
            if not ok:
                self._show_kitty_raw(
                    path, image_id=self.image_id, delete_after=delete_after
                )
            self.image_id += 1
            return

        if self._term_is_iterm2():
            self._show_iterm2(path)
            return

        if self.driver == "chafa":
            self._show_chafa(path)
            return
        if self.driver == "viu":
            self._show_viu(path)
            return

        print("[i] Kein grafikfähiges Terminal erkannt – öffne extern.")
        self._open_external(path)

    def delete_all_kitty_images(self) -> None:
        """Entfernt alle bisher angezeigten Bilder (funktioniert für beide Kitty-Wege)."""
        # icat-Bilder zuverlässig leeren (wenn kitty vorhanden)
        if shutil.which("kitty"):
            try:
                subprocess.run(
                    ["kitty", "+kitten", "icat", "--clear"],
                    check=False,
                    stdout=sys.stdout,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        # Raw-KGP Bilder mit unseren IDs löschen
        for i in range(1, self.image_id):
            sys.stdout.write(f"\x1b_Ga=d,i={i}\x1b\\")
        sys.stdout.flush()

    def reset_image_id(self, start: int = 1) -> None:
        self.image_id = start
