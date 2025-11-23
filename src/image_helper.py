# image_helper.py
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# ─────────────────────────────────────────────────────────────────────────────
# Pillow optional einbinden (fallbacks, damit Import-Errors nicht alles sprengen)
# ─────────────────────────────────────────────────────────────────────────────
from pil_image import Image, PILImageType

# ─────────────────────────────────────────────────────────────────────────────
# NumPy optional einbinden und als _np verfügbar machen (wie in ai_helpers.py)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as _np  # runtime

    _NP_OK = True
except Exception:
    _NP_OK = False
    _np = None  # type: ignore[assignment]

# Externer Datei-Helper beibehalten (bestehendes Verhalten)
import consoleOutput as co
import graphic_helpers as gh
import helpers as he
import mem_guard as mg
from fileSystem import save_file
from i18n import _
from loghandler import print_log

_FR_RE = re.compile(r"frame_(\d+)")


# ─────────────────────────────────────────────────────────────────────────────
# Intern: Helfer
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_rgb(img: Any) -> Any:
    """
    Stellt sicher, dass das Bild im RGB-Mode vorliegt.
    Konvertiert CMYK, RGBA, L, LA etc. sauber nach RGB.
    """
    try:
        mode = getattr(img, "mode", None)
        if mode != "RGB":
            return img.convert("RGB")
        return img
    except Exception:
        # Falls Pillow fehlt oder ein exotischer Typ vorliegt
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Öffentliche API (wird von ai_helpers.py genutzt)
# ─────────────────────────────────────────────────────────────────────────────
def load_png(path: Path) -> Any:
    """
    PNG laden und als RGB zurückgeben.
    """
    im = Image.open(str(path))
    return _ensure_rgb(im)


def save_png(img: Any, path: Path) -> None:
    """
    PNG robust speichern (sorgt für RGB, nutzt deinen save_file-Helper).
    """
    img = _ensure_rgb(img)
    # Falls save_file die Verzeichnisse nicht selbst anlegt, hier absichern:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Kompatibel zum bisherigen Verhalten (Format explizit angeben):
    save_file(img, path, format="PNG")


# === NEW: fast PNG saver (falls ih.save_png langsamer ist) ===
def png_save_fast(img_or_np: Any, out_path: Path) -> None:
    """Schneller PNG-Writer (Pillow), akzeptiert PIL.Image **oder** NumPy-Array."""
    lvl = 1
    try:
        lvl = int(os.environ.get("PNG_LEVEL", "1") or "1")
    except Exception:
        pass

    try:
        if isinstance(img_or_np, Image.Image):
            im = cast(PILImageType, img_or_np)
        else:
            im = cast(PILImageType, img_from_np(img_or_np))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(str(out_path), format="PNG", compress_level=lvl, optimize=False)
        return
    except Exception:
        pass
    # Fallbacks
    try:
        save_png(
            img_or_np if isinstance(img_or_np, Image.Image) else img_from_np(img_or_np),
            out_path,
        )
        return
    except Exception:
        pass
    # letzter Versuch (roh)
    if isinstance(img_or_np, Image.Image):
        Image.fromarray(np_from_img(img_or_np)).save(str(out_path))
    else:
        Image.fromarray(img_or_np).save(str(out_path))


def np_from_img(img: Any) -> Any:
    """
    PIL.Image -> NumPy-Array
    - garantiert 3 Kanäle (RGB)
    - Graustufen werden auf 3 Kanäle erweitert
    - Alpha-Kanal (RGBA/LA) wird entfernt
    Hinweis: dtype bleibt wie von Pillow geliefert (typischerweise uint8).
    """
    if not _NP_OK or _np is None:
        # exakt die Exception, die im Log sichtbar war → klarere Ursache
        raise NameError("NumPy (_np) not available in image_helper.py")

    img = _ensure_rgb(img)
    arr = _np.asarray(img)

    # H×W (Graustufe) → H×W×3
    if arr.ndim == 2:
        arr = _np.stack([arr, arr, arr], axis=-1)

    # H×W×4 (RGBA) → H×W×3 (RGB)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    return arr


def img_from_np(arr: Any) -> Any:
    """
    NumPy-Array -> PIL.Image (RGB)
    - akzeptiert H×W, H×W×1, H×W×3, H×W×4
    - clamp & cast auf uint8
    """
    if not _NP_OK or _np is None:
        raise NameError("NumPy (_np) not available in image_helper.py")

    a = arr
    # H×W → H×W×3
    if getattr(a, "ndim", None) == 2:
        a = _np.stack([a, a, a], axis=-1)
    # Falls mehr als 3 Kanäle vorhanden sind → auf 3 Kanäle reduzieren
    if getattr(a, "ndim", None) == 3 and a.shape[-1] > 3:
        a = a[:, :, :3]

    if getattr(a, "dtype", None) is not _np.uint8:
        a = _np.clip(a, 0, 255).astype(_np.uint8, copy=False)

    return Image.fromarray(a, mode="RGB")


def read_frames_meta_from_dir(frames_dir: Path) -> Dict[str, Any]:
    import json

    p = frames_dir / "frames_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count_produced_frames(
    up_dir: Path, input_dir: Optional[Path], esrgan_root: Path
) -> int:
    try:
        raw_stems = {p.stem for p in list_raw_frames(input_dir or up_dir)}
    except Exception:
        raw_stems = set()

    def _belongs(p: Path) -> bool:
        s = p.stem[:-4] if p.stem.endswith("_out") else p.stem
        return (s in raw_stems) or p.name.startswith("frame_")

    produced: set[str] = set()
    # 1) up_dir (ohne Rekursion)
    for pat in ("*_out.png", "frame_*.png"):
        for p in up_dir.glob(pat):
            if p.is_file() and _belongs(p):
                produced.add(p.name)
    # 2) results/
    results_dir = esrgan_root / "results"
    if results_dir.exists():
        for pat in ("*_out.png", "frame_*.png"):
            for p in results_dir.rglob(pat):
                if p.is_file() and _belongs(p):
                    produced.add(p.name)
    # 3) input_dir (nur *_out.png)
    if input_dir and input_dir.exists():
        for p in input_dir.glob("*_out.png"):
            if p.is_file() and _belongs(p):
                produced.add(p.name)
    n = len(produced)
    samples = ", ".join(sorted(list(produced))[:6])
    print_log(
        f"[COUNT] produced frames: {n} (samples: {samples}{' …' if n > 6 else ''})"
    )
    return n


def get_wh_for_input(p: Path) -> Tuple[int, int]:
    if p.suffix.lower() == ".bmp":
        return _read_bmp_wh(p)
    if p.suffix.lower() == ".png":
        return read_png_wh(p)
    return (0, 0)


def original_frame_path(raw_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".bmp", ".jpg", ".jpeg"):
        p = raw_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def frame_index_from_name(p: Path) -> int:
    m = _FR_RE.search(p.stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 10**9


def list_raw_frames(raw_dir: Path) -> List[Path]:
    bmps = sorted(raw_dir.glob("frame_*.bmp"), key=frame_index_from_name)
    if bmps:
        return bmps
    pngs = sorted(raw_dir.glob("frame_*.png"), key=frame_index_from_name)
    if pngs:
        return pngs
    jpgs = sorted(raw_dir.glob("frame_*.jpg"), key=frame_index_from_name)
    if jpgs:
        return jpgs
    return []


def count_raw_frames(raw_dir: Path) -> int:
    n = len(list_raw_frames(raw_dir))
    print_log(f"[count_raw_frames] dir={raw_dir} → {n}")
    return n


def _read_bmp_wh(p: Path) -> Tuple[int, int]:
    try:
        with p.open("rb") as f:
            hdr = f.read(26)
        if len(hdr) >= 26 and hdr[:2] == b"BM":
            w = int.from_bytes(hdr[18:22], "little", signed=True)
            h = abs(int.from_bytes(hdr[22:26], "little", signed=True))
            return (abs(w), h) if w and h else (0, 0)
    except Exception:
        pass
    return (0, 0)


def _write_frames_meta(raw_dir: Path, meta: Dict[str, Any]) -> None:
    try:
        (raw_dir / "frames_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print_log(f"[frames_meta] wrote → {raw_dir / 'frames_meta.json'}")
    except Exception as e:
        co.print_warning(f"[frames_meta] write failed: {e}")


def read_png_wh(p: Path) -> Tuple[int, int]:
    try:
        with p.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return (0, 0)
            _tmp = f.read(4)
            typ = f.read(4)
            if typ != b"IHDR":
                return (0, 0)
            w = int.from_bytes(f.read(4), "big")
            h = int.from_bytes(f.read(4), "big")
            return (w, h) if w and h else (0, 0)
    except Exception:
        pass
    return (0, 0)


def estimate_input_megapixels(raw_dir: Path) -> float:
    frames = list_raw_frames(raw_dir)
    if not frames:
        return 0.0
    first = frames[0]
    w = h = 0
    if first.suffix.lower() == ".bmp":
        w, h = _read_bmp_wh(first)
    elif first.suffix.lower() == ".png":
        w, h = read_png_wh(first)
    if w > 0 and h > 0:
        mp = (w * h) / 1e6
        print_log(f"[estimate_mp] {w}x{h} → {mp:.3f}MP")
        return mp
    return 0.0


def copy_samples(src: Path, dst: Path) -> None:
    """Kopiert ein paar kleine Beispiel-Dateien (Metadaten, erste Frames, Logs)."""
    try:
        dst.mkdir(parents=True, exist_ok=True)
        # frames_meta.json
        m = src / "frames_meta.json"
        if m.exists():
            shutil.copy2(m, dst / "frames_meta.json")
        # erste 3 PNG/BMP
        cnt = 0
        for pat in ("frame_*.png", "frame_*.bmp"):
            for p in sorted(src.glob(pat), key=frame_index_from_name):
                shutil.copy2(p, dst / p.name)
                cnt += 1
                if cnt >= 3:
                    break
            if cnt >= 3:
                break
        # __logs__ falls vorhanden
        lg = src / "__logs__"
        if lg.exists():
            (dst / "__logs__").mkdir(exist_ok=True)
            copied = 0
            for p in sorted(lg.glob("*.log")):
                shutil.copy2(p, dst / "__logs__" / p.name)
                copied += 1
                if copied >= 20:
                    break
    except Exception as e:
        co.print_warning(f"[persist] sample copy failed: {e}")


def extract_frames_to_bmp(
    *,
    video_path: Path,
    raw_dir: Path,
    start_frame: int,
    end_frame: int,
    idx: int,
    input_name: str,
    BATCH_MODE: bool,
    total_chunks: Optional[int] = None,
) -> bool:
    import os
    import select
    import shutil as _shutil
    import signal
    import termios
    import threading
    import time
    import tty

    print_log(
        f"[extract] begin video={video_path} range={start_frame}-{end_frame} raw_dir={raw_dir}"
    )
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if end_frame < start_frame:
        co.print_error(_("ai_chunk_no_frames").format(idx=idx))
        return False

    total_out = int(end_frame) - int(start_frame) + 1
    vf_expr = f"select=between(n\\,{start_frame}\\,{end_frame})"

    ff = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-progress",
        "pipe:2",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vf",
        vf_expr,
        "-pix_fmt",
        "bgr24",
        "-vsync",
        "0",
        "-start_number",
        "1",
        "-f",
        "image2",
        "-frames:v",
        str(total_out),
        str(raw_dir / "frame_%06d.bmp"),
    ]

    two_bars = isinstance(total_chunks, int) and (total_chunks or 0) > 1
    cur_chunk = (idx or 0) + 1
    chunk_title = f"Chunk {cur_chunk}/{total_chunks or 1} – Frames extrahieren…"
    hint = _("cancel_hint")

    try:
        term_cols = _shutil.get_terminal_size((80, 20)).columns
    except Exception:
        term_cols = 80
    static_len = len(" 100% []")
    bar_len = max(20, min(80, max(20, term_cols - static_len)))

    p_chunk = max(0.0, min((cur_chunk - 1) / float(total_chunks or 1), 1.0))
    top_bar_val, _tmp = gh.make_bar(p_chunk, bar_len) if two_bars else (None, None)
    bot_bar_val, _tmp = gh.make_bar(0.0, bar_len)
    gh.draw_chunk_block(
        two_bars=two_bars,
        title=chunk_title,
        top_bar=top_bar_val,
        bot_bar=bot_bar_val,
        hint=hint,
    )

    class _CancelWatcher:
        def __init__(self, proc: subprocess.Popen):
            self.proc = proc
            self.ev = threading.Event()
            self.fd = None
            self.old = None
            self.fh = None
            self.th = None

        def start(self) -> None:
            if os.name != "posix":
                return
            try:
                self.fh = open("/dev/tty", "rb", buffering=0)
                self.fd = self.fh.fileno()
                self.old = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:
                self._restore()
                return

            def _run():
                try:
                    while self.proc.poll() is None and not self.ev.is_set():
                        r, _tmp1, _tmp2 = select.select([self.fh], [], [], 0.1)
                        if not r:
                            continue
                        try:
                            fd_i: int = self.fd if isinstance(self.fd, int) else -1
                            if fd_i < 0:
                                break
                            ch = os.read(fd_i, 1)
                        except Exception:
                            break
                        if ch in (b"\x1b", b"q"):
                            self.ev.set()
                            mg.CANCEL.set()
                            try:
                                self.proc.send_signal(signal.SIGINT)
                            except Exception:
                                pass
                            break
                finally:
                    self._restore()

            self.th = threading.Thread(target=_run, daemon=True)
            self.th.start()

        def stop(self) -> None:
            self.ev.set()
            try:
                if self.th and self.th.is_alive():
                    self.th.join(timeout=1.0)
            finally:
                self._restore()

        def _restore(self) -> None:
            try:
                if self.fd is not None and self.old is not None:
                    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass
            try:
                if self.fh is not None:
                    self.fh.close()
            except Exception:
                pass

    proc = mg.popen(ff, text=True)  # eigene Prozessgruppe via mem_guard
    cw = _CancelWatcher(proc)
    cw.start()
    try:
        proc.wait()
    finally:
        cw.stop()

    if not list_raw_frames(raw_dir):
        co.print_error(_("ai_chunk_no_frames").format(idx=idx))
        return False

    vm = he.ffprobe_video_meta(video_path)
    meta = {
        "source": str(video_path),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frames_extracted": int(total_out),
        "fps_num": int(vm.get("fps_num", 0)),
        "fps_den": int(vm.get("fps_den", 1)),
        "fps": float(vm.get("fps", 0.0)),
        "nb_frames_src": int(vm.get("nb_frames", 0)),
        "duration_src": float(vm.get("duration", 0.0)),
        "time_base": str(vm.get("time_base", "1/1000")),
        "written_at": time.time(),
    }
    _write_frames_meta(raw_dir, meta)

    return True
