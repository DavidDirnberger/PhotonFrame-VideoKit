#!/usr/bin/env python3
# metadata_support.py
# -*- coding: utf-8 -*-
"""
metadata_support.py – Kanonische Lese-/Schreib-Helfer für Container-Metadaten.

Öffentliche API:
- read_editable_metadata(file, to_json=None) -> Dict[str, str]
- write_editable_metadata(src_file, out_file, data) -> Path
- verify_metadata_written(file, desired) -> (ok: bool, missing: set[str])
- copy_all_metadata(src_file, out_file) -> Path

Design:
- Harmonisierte, containerunabhängige Keys (gemäß definitions.META_TAGS).
- AVI/RIFF: IARL (ArchivalLocation) ist unzuverlässig → Location/Unicode nach XMP.
- Sanity-Check: Nicht-ASCII in AVI/RIFF wird ASCII-sanitisiert + Original nach XMP.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

# Projektmodule
import definitions as defin

# Optionale Projektmodule
try:
    import video_thumbnail as vt
except Exception:
    vt = None  # type: ignore

try:
    import process_wrappers as pw
except Exception:
    pw = None  # type: ignore

# global switch (default off). Kann via Env überschrieben werden.
ALLOW_XMP_SIDECAR = bool(int(os.getenv("PFVID_ALLOW_XMP", "0")))

# --------------------------- Utilities ---------------------------


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kw)


def _ext(path: str | Path) -> str:
    return Path(path).suffix.lstrip(".").lower()


def _is_mp4_family(ext: str) -> bool:
    return ext in {"mp4", "m4v", "mov"}


def _is_mkv_family(ext: str) -> bool:
    return ext in {"mkv", "webm"}


def _is_mpeg_family(ext: str) -> bool:
    return ext in {"mpg", "mpeg", "ts", "m2ts"}


def _has_exiftool() -> bool:
    return shutil.which("exiftool") is not None


# Ersetze in metadata_support.py
def _first_year(text: str) -> str:
    """
    Extrahiert einen vierstelligen Jahrgang (YYYY), wenn möglich.
    Unterstützt u.a.:
      - 1999, 2024
      - 10.10.24, 10/10/24, 24-10-10, 24.10.10
      - 2024-10-10, 2024/10/10
    Rückgabe: 'YYYY' oder originaler Text, falls kein Jahr ermittelbar.
    """
    s = str(text or "").strip()

    # 1) Bevorzugt echte 4-stellige Jahre (19xx/20xx)
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return m.group(0)

    # 2) dd.mm.yy / dd-mm-yy / dd/mm/yy (Jahr 2-stellig → Pivot 70)
    m = re.search(r"(\d{1,2})[.\-\/](\d{1,2})[.\-\/](\d{2})(?!\d)", s)
    if m:
        yy = int(m.group(3))
        year = 1900 + yy if yy >= 70 else 2000 + yy
        return str(year)

    # 3) yy-mm-dd / yy.mm.dd / yy/mm/dd (Jahr vorne, 2-stellig)
    m = re.search(r"(^|\D)(\d{2})[.\-\/]\d{1,2}[.\-\/]\d{1,2}(?!\d)", s)
    if m:
        yy = int(m.group(2))
        year = 1900 + yy if yy >= 70 else 2000 + yy
        return str(year)

    # 4) Fallback: irgendeine 2-stellige "Jahr" Angabe allein
    m = re.search(r"(^|\D)(\d{2})(?!\d)($|\D)", s)
    if m:
        yy = int(m.group(2))
        year = 1900 + yy if yy >= 70 else 2000 + yy
        return str(year)

    # Kein Jahr ermittelbar → Original zurück
    return s


def _defin_keys() -> set[str]:
    try:
        return set(getattr(defin, "META_TAGS", {}).keys())
    except Exception:
        return set()


def _editable_keys() -> set[str]:
    try:
        meta = getattr(defin, "META_TAGS", {})
        return {k for k, v in meta.items() if not v.get("protected", False)}
    except Exception:
        return set()


def _filter_editable(meta: Mapping[str, str]) -> Dict[str, str]:
    keys = _editable_keys()
    out: Dict[str, str] = {}
    for k, v in meta.items():
        if k in keys:
            s = str(v).strip()
            if s != "":
                out[k] = s
    return out


def _has_non_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return False
    except Exception:
        return True


def _sanitize_ascii_heuristic(s: str) -> str:
    # Minimal robust for DE + gängige Zeichen
    mapping = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "é": "e",
        "è": "e",
        "ê": "e",
        "á": "a",
        "à": "a",
        "â": "a",
        "ó": "o",
        "ò": "o",
        "ô": "o",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Á": "A",
        "À": "A",
        "Â": "A",
        "Ó": "O",
        "Ò": "O",
        "Ô": "O",
        "€": "EUR",
        "’": "'",
        "„": '"',
        "“": '"',
        "”": '"',
        "‚": "'",
        "–": "-",
        "—": "-",
    }
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
        else:
            out.append(mapping.get(ch, "?"))
    return "".join(out)


# --- NEU: PFVID-Fallback im Kommentar (immer in der Videodatei, nie Sidecar) ---
_PFVID_MARKER = "PFVID:"


def _encode_pfvid_blob(d: Mapping[str, str]) -> str:
    try:
        blob = json.dumps(dict(d), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # best effort
        blob = json.dumps({k: str(v) for k, v in d.items()}, ensure_ascii=False)
    return f"{_PFVID_MARKER}{blob}"


def _decode_pfvid_blob(s: str) -> Dict[str, str]:
    """
    Extrahiert den *letzten* PFVID:{...}-Block aus einem Kommentartext.
    Gibt {} zurück, wenn keiner vorhanden/parsbar.
    """
    if not s:
        return {}
    idx = s.rfind(_PFVID_MARKER)
    if idx < 0:
        return {}
    j = s[idx + len(_PFVID_MARKER) :].strip()
    try:
        data = json.loads(j)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _extract_pfvid_from_raw_tags(
    raw_tags: Mapping[str, Any], ext: str
) -> Dict[str, str]:
    # Kommentar-Feld je nach Container
    candidates = []
    # ffprobe liefert lowercase Keys
    for k in (
        "comment",
        "icmt",
    ):  # AVI icmt wird via ffmpeg als "comment" gehandhabt; wir prüfen beides
        v = raw_tags.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v)
    for text in reversed(candidates):
        d = _decode_pfvid_blob(text)
        if d:
            return d
    return {}


# --- Container-Fähigkeiten & native Schreibbarkeit ----------------------

# Kanonische Keys, die wir *nativ* gut in MOV/MP4 (iTunes) schreiben können.
# (Wir setzen sowohl generische als auch iTunes-4CC, wo sinnvoll.)
_MOV_NATIVE = {
    "title",
    "artist",
    "album",
    "album_artist",
    "genre",
    "comment",
    "description",
    "publisher",
    "website",
    "copyright",
    "encoded_by",
    "date",
    "show",
    "season_number",
    "episode_id",
    "network",
    "rating",
    "track",
    "track_total",
    "disc",
    "disc_total",
    "production_year",
}


# MKV/WEBM akzeptiert generische Key=Value-Tags breit; wir markieren alles als nativ.
def _mkv_native(keys: Iterable[str]) -> set[str]:
    return set(keys)


# AVI/RIFF – was als RIFF INFO existiert oder via _CANON_TO_RIFF_OUT gemapped wird
_AVI_NATIVE_EXTRA = {
    "title",
    "artist",
    "album",
    "comment",
    "copyright",
    "genre",
    "website",
    "encoded_by",
    "director",
    "publisher",
    "location",
    "production_year",
    "date",
}


def _native_supported_keys(ext: str) -> set[str]:
    ext = ext.lower()
    all_def = _defin_keys()
    if _is_mp4_family(ext):
        return {k for k in all_def if k in _MOV_NATIVE}
    if _is_mkv_family(ext) or _is_mpeg_family(ext):
        return _mkv_native(all_def)
    if ext == "avi":
        return set(_CANON_TO_RIFF_OUT.keys()) | _AVI_NATIVE_EXTRA
    # Sonstige Container: konservativ – nur generisch verbreitete
    return {
        "title",
        "artist",
        "album",
        "genre",
        "comment",
        "description",
        "publisher",
        "website",
        "copyright",
        "encoded_by",
        "date",
        "production_year",
    }


# --- Helper für Track/Disc X/Y Syntax (MP4/MOV & auch MKV ok) -----------
def _compose_fraction_pair(a: Optional[str], b: Optional[str]) -> Optional[str]:
    try:
        x = int(str(a)) if a is not None and str(a).strip() != "" else None
    except Exception:
        x = None
    try:
        y = int(str(b)) if b is not None and str(b).strip() != "" else None
    except Exception:
        y = None
    if x is None and y is None:
        return None
    if x is None:
        x = 0
    if y is None:
        return str(x)
    return f"{x}/{y}"


# -------------------------------------------------------------------------
def _normalize_language_code(label: str) -> Optional[str]:
    """
    Normalisiert Sprache → ISO-639-2/T (3-letter), z.B. 'Englisch' → 'eng'.
    Nutzt definitions.LANG_ISO3, fällt sonst auf einfache Heuristik zurück.
    """
    s = (label or "").strip().lower()
    if not s:
        return None

    # Tokens wie "en-us" → "en-us" (erstes Token reicht hier)
    token = re.split(r"[,;|/_\-\s]+", s)[0]

    try:
        iso3_map = getattr(defin, "LANG_ISO3", {}) or {}
    except Exception:
        iso3_map = {}

    # Direkt aus Map
    iso = iso3_map.get(token) or (iso3_map.get(token[:2]) if len(token) >= 2 else None)
    if iso:
        return iso.lower()

    # Minimaler Fallback
    basic = {
        "de": "deu",
        "ger": "deu",
        "deu": "deu",
        "german": "deu",
        "deutsch": "deu",
        "en": "eng",
        "eng": "eng",
        "english": "eng",
        "fr": "fra",
        "fre": "fra",
        "fra": "fra",
        "french": "fra",
        "es": "spa",
        "spa": "spa",
        "spanish": "spa",
        "esp": "spa",
        "it": "ita",
        "ita": "ita",
        "italian": "ita",
    }
    return basic.get(token)


def _ffprobe_stream_langs(path: Path) -> dict[str, list[tuple[int, str]]]:
    """
    Liefert pro Typ ('a' Audio, 's' Subtitle) eine Liste (index, lang) mit
    3-letter Codes oder "" wenn nicht gesetzt.
    """
    # Audio
    ra = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index:stream_tags=language",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    rs = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index:stream_tags=language",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    out: dict[str, list[tuple[int, str]]] = {"a": [], "s": []}
    try:
        da = json.loads(ra.stdout or "{}")
        for st in da.get("streams") or []:
            idx = int(st.get("index"))
            tag = (st.get("tags") or {}).get("language") or ""
            out["a"].append((idx, str(tag).strip().lower()))
    except Exception:
        pass
    try:
        ds = json.loads(rs.stdout or "{}")
        for st in ds.get("streams") or []:
            idx = int(st.get("index"))
            tag = (st.get("tags") or {}).get("language") or ""
            out["s"].append((idx, str(tag).strip().lower()))
    except Exception:
        pass
    return out


def _streams_languages_match(path: Path, want_iso3: str) -> bool:
    got = _ffprobe_stream_langs(path)
    pairs = got["a"] + got["s"]
    if not pairs:
        return True  # nichts zu prüfen
    # bisher: alle müssen == want_iso3 sein
    # Alternative (lockerer): mindestens einer passt
    return any(lang == want_iso3 for _, lang in pairs)


# Aliasse für Eingabe/Schreiben: gruppiere semantisch ähnliche Keys
_ALIAS_GROUPS: list[tuple[str, set[str]]] = [
    ("location", {"location", "ort", "place", "archival_location", "iarl"}),
    ("production_year", {"production_year", "year", "jahr", "date"}),
]


def _coerce_aliases_to_primary(meta: Mapping[str, str]) -> Dict[str, str]:
    out = dict(meta)
    for primary, group in _ALIAS_GROUPS:
        if primary in out:
            continue
        for k in group:
            if k in out:
                out[primary] = out.pop(k)
                break
    return out


def _prefer_keys_for_defin(meta: Mapping[str, str]) -> Dict[str, str]:
    """Mappt gelesene Gruppeneinträge auf in defin vorhandene Schlüssel."""
    keys_def = _defin_keys()
    out = dict(meta)
    for primary, group in _ALIAS_GROUPS:
        present_key = next(
            (k for k in group if k in out and str(out[k]).strip() != ""), None
        )
        if not present_key:
            continue
        target = next((k for k in group if k in keys_def), primary)
        val = out[present_key]
        for k in group:
            if k in out and k != target:
                del out[k]
        out[target] = val
    return out


def _merge_preferring(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    out = dict(a)
    out.update({k: v for k, v in b.items() if v is not None and str(v).strip() != ""})
    return out


# --------------------- Lesen: ffprobe + ExifTool (AVI) ---------------------


def _ffprobe_read_format_tags(path: Path) -> Dict[str, str]:
    r = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        fmt: Dict[str, Any] = json.loads(r.stdout or "{}").get("format", {}) or {}
        tags_in: Dict[str, Any] = fmt.get("tags", {}) or {}
    except Exception:
        tags_in = {}
    return {str(k).lower(): str(v) for k, v in tags_in.items()}


# RIFF/INFO → kanonisch
_RIFF_IN_TO_CANON = {
    "software": "encoder",
    "iarl": "location",
    "icrd": "production_year",
    "date": "production_year",
    "inam": "title",
    "iart": "artist",
    "iprd": "album",
    "icmt": "comment",
    "icop": "copyright",
    "idir": "director",
    "ipub": "publisher",
    "iweb": "website",
    "igen": "genre",
    "ignr": "genre",
    "ienc": "encoded_by",
}

# ExifTool → RIFF-ähnliche Keys rückmappen
_EXIF_RIFF_BACKMAP = {
    "archivallocation": "iarl",
    "datecreated": "icrd",
    "artist": "iart",
    "title": "inam",
    "album": "iprd",
    "comment": "icmt",
    "copyright": "icop",
    "director": "idir",
    "publisher": "ipub",
    "genre": "ignr",
    "software": "isft",
    "encodedby": "ienc",
}

# Kanonisch → ExifTool generisch (Matroska/MP4/MOV/WebM/MPEG)
_CANON_TO_EXIF_GENERIC = {
    "title": "Title",
    "artist": "Artist",
    "album": "Album",
    "album_artist": "AlbumArtist",
    "comment": "Comment",
    "description": "Description",
    "genre": "Genre",
    "publisher": "Publisher",
    "website": "URL",
    "encoded_by": "Encoder",
    "copyright": "Copyright",
    "director": "Director",
    "show": "TVShow",
    "season_number": "TVSeason",
    "episode_id": "TVEpisode",
    "network": "TVNetwork",
    "rating": "Rating",
    "track": "Track",
    "track_total": "TrackTotal",
    "disc": "DiscNumber",
    "disc_total": "DiscTotal",
    "production_year": "Date",  # Jahr wird über ExifTool normalisiert
}


def _exiftool_write_generic(
    dst: Path, ext: str, canon_editable: Mapping[str, str]
) -> bool:
    if not _has_exiftool():
        return False
    args: list[str] = ["exiftool", "-overwrite_original", "-P", "-q", "-q"]
    wrote = False
    for k, v in canon_editable.items():
        if k == "language":
            continue  # Stream-Thema, nicht als Format-Tag
        tag = (
            _CANON_TO_EXIF_GENERIC.get(k) if ext != "avi" else _CANON_TO_EXIF_AVI.get(k)
        )
        if not tag:
            continue
        val = _first_year(v) if k == "production_year" else str(v)
        if val.strip() == "":
            continue
        args.append(f"-{tag}={val}")
        wrote = True
    if not wrote:
        return True
    args.append(str(dst))
    r = _run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode == 0


def _exiftool_read_tags(path: Path) -> Dict[str, str]:
    """
    Liest ExifTool-JSON und mappt RIFF + XMP-Felder auf *RIFF-ähnliche* Keys,
    die anschließend in _harmonize_in(...) zu kanonischen Keys umgebogen werden.
    Zusätzlich werden XMP-Felder direkt auf RIFF-Äquivalente zurückgemappt,
    damit der nachgelagerte Pfad unverändert bleiben kann.
    """
    if not _has_exiftool():
        return {}

    r = _run(
        ["exiftool", "-j", "-n", "-charset", "filename=utf8", str(path)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return {}

    try:
        arr = json.loads(r.stdout or "[]")
        if not isinstance(arr, list) or not arr:
            return {}
        meta = arr[0]
    except Exception:
        return {}

    out: Dict[str, str] = {}

    # 1) RIFF zurück auf RIFF-Keys mappen (wie gehabt)
    for k, v in meta.items():
        lk = str(k).lower()
        if lk in _EXIF_RIFF_BACKMAP:
            out[_EXIF_RIFF_BACKMAP[lk]] = str(v)

    # 2) XMP → RIFF-Äquivalente (Namespace tolerant)
    def _norm_xmp_key(s: str) -> str:
        s = s.lower()
        # namespace & dekor entfernen
        s = re.sub(r"^(xmp[-:]|xmp-dc:|xmpdm:|xmp-exif:|xmp-xmpmm:)", "", s)
        s = s.replace(" ", "").replace("_", "")
        return s

    XMP_TO_RIFF = {
        "title": "inam",  # XMP-dc:Title → RIFF:INAM
        "creator": "iart",  # XMP-dc:Creator → RIFF:IART
        "artist": "iart",
        "album": "iprd",
        "description": "icmt",  # XMP-dc:Description → Comment
        "comment": "icmt",
        "coverage": "iarl",  # XMP-dc:Coverage → Location
        "url": "iweb",
        "publisher": "ipub",
        "genre": "ignr",
        "encodedby": "ienc",
        "date": "icrd",  # Jahr wird später in _harmonize_in normalisiert
        "language": "language",  # nur zur Durchreichung (nicht Container-Format!)
        "subject": "keywords",  # Keywords (für Sidecar sinnvoll)
    }

    for k, v in meta.items():
        nk = _norm_xmp_key(str(k))
        if nk in XMP_TO_RIFF and str(v).strip():
            out[XMP_TO_RIFF[nk]] = str(v).strip()

    return out


def _harmonize_in(tags: Dict[str, str], ext: str) -> Dict[str, str]:
    t = dict(tags)

    if ext == "avi":
        out: Dict[str, str] = {}
        for k, v in t.items():
            lk = k.lower()
            if lk in _RIFF_IN_TO_CANON:
                ck = _RIFF_IN_TO_CANON[lk]
                out[ck] = _first_year(v) if ck == "production_year" else v
            else:
                out[lk] = v  # generische Namen durchreichen
        return out

    if _is_mp4_family(ext):
        out = dict(t)
        if "production_year" not in out:
            if "creation_time" in out:
                out["production_year"] = _first_year(out["creation_time"])
            elif "date" in out:
                out["production_year"] = _first_year(out["date"])
        if "encoder" not in out and "software" in out:
            out["encoder"] = out["software"]
        return out

    if _is_mkv_family(ext) or _is_mpeg_family(ext):
        out = dict(t)
        if "production_year" not in out and "date" in out:
            out["production_year"] = _first_year(out["date"])
        return out

    return dict(t)


# --- Helper: PFVID aus Kommentar rausfiltern ---------------------------------
def _strip_pfvid_from_text(s: str) -> str:
    if not s:
        return s
    idx = s.rfind(_PFVID_MARKER)
    return s[:idx].rstrip() if idx >= 0 else s


def read_editable_metadata(
    file: str | Path, to_json: Optional[str | Path] = None
) -> Dict[str, str]:
    """
    Liest *editierbare* Metadaten in kanonischen Keys.
    AVI: ergänzt ffprobe via ExifTool-Backmap (IARL/ICRD/…).
    """
    path = Path(file)
    ext = _ext(path)

    raw = _ffprobe_read_format_tags(path)
    if ext == "avi" and _has_exiftool():
        raw = _merge_preferring(raw, _exiftool_read_tags(path))

    norm = _harmonize_in(raw, ext)
    norm = _prefer_keys_for_defin(norm)
    editable = _filter_editable(norm)

    if to_json:
        jpath = Path(to_json)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(editable, f, indent=2, ensure_ascii=False)

        # PFVID-Fallbacks aus Kommentar extrahieren und reinmappen (nur editable Keys)
    try:
        pfvid = _extract_pfvid_from_raw_tags(raw, ext)
        if pfvid:
            # nur Keys übernehmen, die in defin.* existieren und editable sind
            ed_keys = _editable_keys()
            for k, v in pfvid.items():
                if k in ed_keys and str(v).strip() != "":
                    editable.setdefault(k, str(v))
    except Exception:
        pass

    # Kommentarwert bereinigen (PFVID-Blob entfernen), damit Vergleiche stabil sind
    try:
        if "comment" in editable:
            editable["comment"] = _strip_pfvid_from_text(editable.get("comment", ""))
    except Exception:
        pass

    return editable


# --------------------- Schreiben: ffmpeg (+ ExifTool/XMP) ---------------------

# Kanonisch → RIFF/INFO
_CANON_TO_RIFF_OUT = {
    # Achtung: IARL ist unzuverlässig – wir versuchen es, aber verlassen uns nicht darauf
    "location": "iarl",
    "production_year": "date",  # ffmpeg mappt auf ICRD
    "encoded_by": "ienc",
    "copyright": "icop",
    "director": "idir",
    "publisher": "ipub",
    "website": "iweb",
    "genre": "genre",  # ffmpeg kümmert sich um IGEN/IGNR
    # title/artist/album/comment: Standard-Keys funktionieren in ffmpeg
}


def _harmonize_out(editable: Mapping[str, str], ext: str) -> Dict[str, str]:
    """
    Mappt kanonische Keys auf container-spezifische Format-Tags für ffmpeg.
    - AVI/RIFF: auf I*** RIFF-INFO Keys
    - MP4/MOV: 'encoded_by' → 'encoder', 'production_year' → 'date'
    - MKV/WEBM/MPEG: 'production_year' → 'date'
    - 'language' NICHT als Container-Tag schreiben (per-stream Thema) → via XMP fallback.
    """
    data = {k: v for k, v in editable.items() if str(v).strip() != ""}

    # Sprache nie als reines Format-Metadatum schreiben (ffmpeg erwartet per-stream).
    data.pop("language", None)

    if ext == "avi":
        out: Dict[str, str] = {}
        for k, v in data.items():
            if k in _CANON_TO_RIFF_OUT:
                out[_CANON_TO_RIFF_OUT[k]] = (
                    _first_year(v) if k == "production_year" else v
                )
            else:
                out[k] = v
        return out

    if _is_mp4_family(ext):
        out = dict(data)

        # production_year → 'date' (human, robust)
        if "production_year" in data:
            out.setdefault("date", _first_year(data["production_year"]))

        # 'encoded_by' → 'encoder' (ffmpeg mappt MOV häufig auf ©too/encoder)
        if "encoded_by" in data and "encoder" not in out:
            out["encoder"] = data["encoded_by"]

        # iTunes TV-Atoms + generische Duplikate (breite Player-Kompat.)
        # show/season/episode/network/rating
        if "show" in data:
            out.setdefault("show", data["show"])  # generisch
            out.setdefault("tvsh", data["show"])  # iTunes
        if "season_number" in data:
            sn = (
                re.sub(r"[^\d]", "", str(data["season_number"])).strip()
                or data["season_number"]
            )
            out.setdefault("season_number", sn)  # generisch
            out.setdefault("tvsn", sn)  # iTunes
        if "episode_id" in data:
            out.setdefault("episode_id", data["episode_id"])
            out.setdefault("tves", data["episode_id"])  # iTunes
        if "network" in data:
            out.setdefault("network", data["network"])
            out.setdefault("tvnn", data["network"])  # iTunes
        if "rating" in data:
            out.setdefault("rating", data["rating"])
            out.setdefault("rtng", data["rating"])  # iTunes

        # Beschreibung kurz/lang (desc/ldes) + comment
        if "description" in data:
            out.setdefault("description", data["description"])
            out.setdefault("desc", data["description"])  # kurz
            out.setdefault("ldes", data["description"])  # lang
            # viele Player zeigen eher 'comment' an
            out.setdefault("comment", data["description"])

        # album_artist → aART (ffmpeg akzeptiert -metadata album_artist=)
        if "album_artist" in data:
            out.setdefault("album_artist", data["album_artist"])
            out.setdefault("aART", data["album_artist"])

        # track/track_total → "1/10", disc/disc_total → "1/2"
        tr = _compose_fraction_pair(data.get("track"), data.get("track_total"))
        if tr:
            out.pop("track", None)
            out.pop("track_total", None)
            out["track"] = tr
            out.setdefault("trkn", tr)  # iTunes konventionell

        dc = _compose_fraction_pair(data.get("disc"), data.get("disc_total"))
        if dc:
            out.pop("disc", None)
            out.pop("disc_total", None)
            out["disc"] = dc
            out.setdefault("disk", dc)

        return out

    if _is_mkv_family(ext) or _is_mpeg_family(ext):
        out = dict(data)
        if "production_year" in data:
            out.setdefault("date", _first_year(data["production_year"]))
        return out

    return dict(data)


# Kanonisch → ExifTool (AVI/RIFF) (Best Effort)
_CANON_TO_EXIF_AVI = {
    "location": "RIFF:ArchivalLocation",
    "production_year": "RIFF:DateCreated",
    "title": "RIFF:Title",
    "artist": "RIFF:Artist",
    "album": "RIFF:Album",
    "comment": "RIFF:Comment",
    "copyright": "RIFF:Copyright",
    "director": "RIFF:Director",
    "publisher": "RIFF:Publisher",
    "genre": "RIFF:Genre",
    "website": "RIFF:URL",
    "encoded_by": "RIFF:EncodedBy",
}

# XMP-Sidecar Mapping – enthält zusätzlich Titel/Artist/Album/Comment/Location
_CANON_TO_XMP_SIDECAR = {
    "title": "XMP-dc:Title",
    "artist": "XMP-dc:Creator",
    "album": "XMP-xmpDM:Album",
    "comment": "XMP-dc:Description",
    "location": "XMP-dc:Coverage",
    "show": "XMP:Show",
    "season_number": "XMP:Season",
    "episode_id": "XMP:Episode",
    "network": "XMP:Network",
    "language": "XMP-dc:Language",
    "keywords": "XMP-dc:Subject",
    "rating": "XMP:Rating",
    "album_artist": "XMP:AlbumArtist",
    "track": "XMP:TrackNumber",
    "track_total": "XMP:TrackCount",
    "disc": "XMP:DiscNumber",
    "disc_total": "XMP:DiscCount",
    "description": "XMP-dc:Description",
    "website": "XMP:URL",
    "date": "XMP:Date",
    "composer": "XMP:Composer",
    "publisher": "XMP:Publisher",
}


def verify_metadata_written(
    file: str | Path, desired: Mapping[str, str]
) -> tuple[bool, set[str]]:
    """
    Prüft Soll→Ist stabil:
      - language: per-Stream (a/s) auf ISO-639-2/T.
      - production_year: vergleicht nur das Jahr (YYYY).
      - comment: wird ohne PFVID-Blob verglichen.
    """
    path = Path(file)
    after = read_editable_metadata(path)

    missing: set[str] = set()

    for k, v in desired.items():
        if v is None:
            continue
        want = str(v).strip()

        if k == "language":
            iso3 = _normalize_language_code(want) or ""
            if iso3 and not _streams_languages_match(path, iso3):
                missing.add("language")
            continue

        if k == "production_year":
            want_norm = _first_year(want)
            got_norm = _first_year(after.get("production_year", ""))

            # Nur hart vergleichen, wenn wir auf beiden Seiten eine 4-stellige Jahreszahl haben
            if re.fullmatch(r"(19|20)\d{2}", want_norm) and re.fullmatch(
                r"(19|20)\d{2}", got_norm
            ):
                if want_norm != got_norm:
                    missing.add("production_year")
            # Wenn die Sollseite kein valides Jahr hat, werten wir das nicht als Fehler.
            continue

        if k == "comment":
            got = _strip_pfvid_from_text(after.get("comment", "")).strip()
            if want != got:
                missing.add("comment")
            continue

        got = str(after.get(k, "")).strip()
        if want != got:
            missing.add(k)

        # Default: exakter Vergleich (getrimmt)
        got = str(after.get(k, "")).strip()
        if want != got:
            missing.add(k)

    return (len(missing) == 0, missing)


def write_editable_metadata(
    src_file: str | Path, out_file: str | Path, data: Mapping[str, str]
) -> Path:
    """
    Schreibt *editable* Metadaten transaktional nach `out_file`.

    Änderungen:
      - KEIN generelles XMP-Sidecar mehr (ALLOW_XMP_SIDECAR steuert Ausnahmen).
      - 'language' wird als Stream-Tag im Container gesetzt (Audio & Subs),
        nicht als Format-Tag und nicht als Sidecar.
    """
    src = Path(src_file)
    dst = Path(out_file)
    dst.parent.mkdir(parents=True, exist_ok=True)

    out_ext = _ext(dst) or _ext(src)
    data_norm = _coerce_aliases_to_primary(dict(data))
    editable_only = _filter_editable(data_norm)

    # Sprache separat merken, da sie *nicht* als Format-Tag geschrieben wird.
    stream_lang_label = editable_only.pop("language", None)
    stream_lang_iso3 = (
        _normalize_language_code(stream_lang_label) if stream_lang_label else None
    )

    # 1) Container-spezifisch vorbereiten (ohne language)
    write_tags = _harmonize_out(editable_only, out_ext)

    # Welche kanonischen Keys gelten als "nativ unterstützt"?
    native = _native_supported_keys(out_ext)

    # Fallback-Map: alles, was *nicht* nativ ist → in PFVID-Kommentar sichern
    pfvid_fallback: Dict[str, str] = {}
    for k, v in editable_only.items():
        if k not in native and k != "language":
            s = str(v).strip()
            if s != "":
                pfvid_fallback[k] = s

    # PFVID in Kommentar integrieren (nie Sidecar)
    if pfvid_fallback:
        existing_comm = str(write_tags.get("comment", "")).rstrip()
        pfvid_blob = _encode_pfvid_blob(pfvid_fallback)
        if existing_comm:
            write_tags["comment"] = f"{existing_comm}\n{pfvid_blob}"
        else:
            write_tags["comment"] = pfvid_blob

    # ---------- Schreibversuche (2 Stufen): copy→drop ----------
    desired_verify = _merge_preferring(
        editable_only, {"language": stream_lang_label} if stream_lang_label else {}
    )

    def _ffmpeg_write_once(src_in: Path, dst_out: Path, copy_meta: bool) -> Path:
        tmp = dst_out.with_name(
            dst_out.stem
            + (".__meta_copy__" if copy_meta else ".__meta_drop__")
            + dst_out.suffix
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src_in),
            "-map",
            "0",
            "-c",
            "copy",
        ]
        cmd += ["-map_metadata", "0"] if copy_meta else ["-map_metadata", "-1"]

        if _is_mp4_family(_ext(dst_out)):
            cmd += ["-movflags", "use_metadata_tags"]

        for k, val in write_tags.items():
            cmd += ["-metadata", f"{k}={val}"]

        if stream_lang_iso3:
            # setze Sprache nur, wenn vorhanden – sonst führt :a/:s ohne Treffer zu Fehlern
            cmd += ["-metadata:s:a", f"language={stream_lang_iso3}"]
            cmd += ["-metadata:s:s", f"language={stream_lang_iso3}"]

        cmd += [str(tmp)]

        # 1) Wenn Progress-Wrapper existiert: benutzen – aber tolerant auswerten
        if pw and hasattr(pw, "run_ffmpeg_with_progress"):
            ok = pw.run_ffmpeg_with_progress(
                src_in.name,
                cmd,
                "Schreibe Metadaten …",
                "Metadaten geschrieben.",
                output_file=tmp,
                BATCH_MODE=True,
            )
            # Wrapper kann bei "nur Metadaten" fälschlich False liefern – Dateiexistenz entscheidet
            if (ok is False) and (not tmp.exists() or tmp.stat().st_size == 0):
                # einmal direkt laufen lassen, um sauberes stderr zu bekommen
                r = _run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
                    err_tail = (r.stderr or "").splitlines()[
                        -1:
                    ]  # letzte Zeile hilft oft
                    raise RuntimeError(
                        "FFMPEG_WRITE_FAILED: "
                        + (" ".join(err_tail) if err_tail else "unknown")
                    )
            return tmp

        # 2) Ohne Wrapper: normal laufen lassen und stderr prüfen
        r = _run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            err_tail = (r.stderr or "").splitlines()[-1:]
            raise RuntimeError(
                "FFMPEG_WRITE_FAILED: "
                + (" ".join(err_tail) if err_tail else "unknown")
            )
        return tmp

    def _maybe_embed_mp4_cover(src_path: Path, tmp_path: Path) -> Path:
        if not (_is_mp4_family(out_ext) and vt):
            return tmp_path
        try:
            has_cover, cov_idx = vt.mp4_has_cover(src_path)
        except Exception:
            has_cover, cov_idx = (False, None)
        if has_cover and cov_idx is not None:
            try:
                cover_jpg = vt.extract_mp4_cover(src_path, cov_idx)
                final_tmp = tmp_path.with_name(
                    tmp_path.stem + ".__final__" + tmp_path.suffix
                )
                vt.embed_mp4(tmp_path, cover_jpg, final_tmp)
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
                return final_tmp
            except Exception:
                return tmp_path
        return tmp_path

    # Versuch 1: Metadaten kopieren und überlagern
    tmp1 = _ffmpeg_write_once(src, dst, copy_meta=True)
    tmp1 = _maybe_embed_mp4_cover(src, tmp1)
    ok1, miss1 = verify_metadata_written(tmp1, desired_verify)

    tmp_final = tmp1
    if not ok1:
        # Versuch 2: Alle alten Metadaten verwerfen und nur unsere setzen
        try:
            tmp2 = _ffmpeg_write_once(src, dst, copy_meta=False)
            tmp2 = _maybe_embed_mp4_cover(src, tmp2)
            ok2, miss2 = verify_metadata_written(tmp2, desired_verify)
            if ok2:
                try:
                    tmp_final.unlink(missing_ok=True)  # py<3.12: wrap in try/except
                except Exception:
                    pass
                tmp_final = tmp2
            else:
                # Aufräumen ...
                try:
                    tmp2.unlink()
                except Exception:
                    pass
                try:
                    tmp1.unlink()
                except Exception:
                    pass

                # --- NEU: ExifTool-Fallback ohne Sidecar ---
                if _has_exiftool():
                    # Wir schreiben direkt in die Zieldatei (dst), nicht in tmp
                    # (bis hierher war noch nichts committed)
                    if _exiftool_write_generic(dst, out_ext, editable_only):
                        ok3, miss3 = verify_metadata_written(dst, desired_verify)
                        if ok3:
                            return dst
                        raise RuntimeError(
                            "DIRECT_METADATA_WRITE_FAILED_EXIFTOOL: "
                            + ", ".join(sorted(miss3))
                        )
                # Kein ExifTool oder auch das verfehlt:
                raise RuntimeError(
                    "DIRECT_METADATA_WRITE_FAILED: " + ", ".join(sorted(miss1))
                )

        except Exception:
            # Aufräumen und mit details aus miss1 raus
            try:
                tmp1.unlink()
            except Exception:
                pass
            raise RuntimeError(
                "DIRECT_METADATA_WRITE_FAILED: " + ", ".join(sorted(miss1))
            )

    # Commit
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    tmp_final.rename(dst)
    return dst
