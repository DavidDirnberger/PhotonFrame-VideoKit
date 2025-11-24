#!/usr/bin/env python3
# definitions.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, TypedDict, Union

__version__ = "0.0.1-alpha"

LOG_FILE = "LOG_DIR/PhotonFabric_VideoKit.log"

# Projektbasis (VM_BASE vom Wrapper, sonst ein Verzeichnis über src/)
PROJECT_ROOT = Path(
    os.environ.get("VM_BASE", Path(__file__).resolve().parents[1])
).resolve()

# --- Gemeinsame Strukturen ----------------------------------------------------


class LocalizedText(TypedDict):
    de: str
    en: str


# VIDEO_CODECS
class CodecInfo(TypedDict):
    name: str
    description: LocalizedText


# CONVERT_FORMAT_DESCRIPTIONS
class ConvertFormatSpec(TypedDict, total=False):
    name: str
    description: LocalizedText
    codecs: List[str]


# PRESETS – ohne NotRequired (Komposition: Pflicht + Optional)
class _PresetReq(TypedDict):
    name: str
    description: LocalizedText
    speed: str
    faststart: bool


class _PresetOpt(TypedDict, total=False):
    quality: Optional[int]
    scale: Optional[str]
    max_fps: Optional[int]
    lossless: bool
    audio_bitrate: Optional[str]  # z. B. "128k", "320k", "pcm"
    audio_channels: Optional[int]  # z. B. 2
    force_yuv420p: bool  # erzwingt yuv420p (Kompatibilität)
    prefer_10bit: bool  # z. B. für cinema/studio/ultra


class PresetSpec(_PresetReq, _PresetOpt):
    pass


# RESOLUTIONS
class ResolutionSpec(TypedDict):
    name: str
    description: LocalizedText
    scale: Optional[str]
    aspect: Optional[str]


# FRAMERATES
class FrameRateSpec(TypedDict):
    name: str
    description: LocalizedText
    fps: Optional[str]


# EXTRACT_MODE
class ExtractModeSpec(TypedDict, total=False):
    name: str
    description: LocalizedText
    formats: List[str]


# SUBTITLE_FORMATS
class SubtitleFormatSpec(TypedDict):
    name: str
    description: LocalizedText
    extensions: List[str]
    features: List[str]


# META_TAGS
class MetaTagInfo(TypedDict):
    protected: bool
    name: LocalizedText
    description: LocalizedText


# VIRTUAL_META_INFO
class VirtualMetaInfo(TypedDict):
    name: LocalizedText
    description: LocalizedText


# ENHANCE_PRESETS (breit, alles optional – „custom“ funktioniert damit auch)
class EnhancePresetSpec(TypedDict, total=False):
    name: LocalizedText
    description: LocalizedText
    filter_chain: str
    stabilize: bool
    stab_method: str
    stab_smooth: int
    stab_rx: int
    stab_ry: int
    denoise: bool
    denoise_method: str
    denoise_intensity: int
    warmth: int
    tint: int
    brightness: int
    contrast: int
    saturation: int
    virtual: bool


# STABILIZATION / NOISE_REDUCTION
class StabilizationSpec(TypedDict, total=False):
    description: LocalizedText
    default: int
    defaultx: int
    defaulty: int


class NoiseReductionSpec(TypedDict):
    description: LocalizedText
    default: int


# MEME_FONTSIZE
class MemeFontSizeSpec(TypedDict):
    name: LocalizedText
    size: int


# RELEVANT_PARAM_GROUPS – rekursiver Union-Typ, passend zu deiner Struktur
SpecLeaf = List[str]
MethodMap = Dict[str, SpecLeaf]  # z. B. "vidstab": ["stab_smooth"]
GateMap = Dict[bool, Union[SpecLeaf, Dict[str, MethodMap]]]
RelevantParamGroups = Dict[str, Union[Set[str], GateMap]]


ANSI_RESET = "\033[0m"
CLEAR_LINE = "\033[2K"  # clear entire line
ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*m")  # strip colours for length calc
TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

# Kompatibilität: alter Name SCRIPT_DIR zeigt jetzt auf das Projekt-Root
SCRIPT_DIR = PROJECT_ROOT
TMP_DIR = PROJECT_ROOT / "tmp"

LANG_ISO3 = {
    "de": "deu",
    "ger": "deu",
    "deu": "deu",
    "german": "deu",
    "deutsch": "deu",
    "de-at": "deu",
    "de-de": "deu",
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "englisch": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    "fr": "fra",
    "fre": "fra",
    "fra": "fra",
    "french": "fra",
    "français": "fra",
    "fr-ca": "fra",
    "es": "spa",
    "spa": "spa",
    "spanish": "spa",
    "español": "spa",
    "it": "ita",
    "ita": "ita",
    "italian": "ita",
    "italiano": "ita",
    "pt": "por",
    "por": "por",
    "portuguese": "por",
    "português": "por",
    "ru": "rus",
    "rus": "rus",
    "russian": "rus",
    "pt-br": "por",
    "pt-pt": "por",
    "zh": "zho",
    "chi": "zho",
    "zho": "zho",
    "zh-cn": "zho",
    "zh-tw": "zho",
    "chinese": "zho",
    "ja": "jpn",
    "jpn": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "kor": "kor",
    "korean": "kor",
    "pl": "pol",
    "pol": "pol",
    "polish": "pol",
    "nl": "nld",
    "dut": "nld",
    "nld": "nld",
    "dutch": "nld",
    "tr": "tur",
    "tur": "tur",
    "turkish": "tur",
    "jp": "jpn",
    "kr": "kor",
    "ua": "ukr",
    "cz": "ces",
    "gr": "ell",
    "el": "ell",
    "fa": "fas",
    "farsi": "fas",
    "persian": "fas",
    "he": "heb",
    "hebrew": "heb",
    "sv": "swe",
    "no": "nor",
    "da": "dan",
    "fi": "fin",
    "cs": "ces",
    "ro": "ron",
    "ar": "ara",
    "hi": "hin",
    "id": "ind",
    "in": "ind",
    "ms": "msa",
    "bn": "ben",
    "vi": "vie",
    "th": "tha",
    "bg": "bul",
    "sk": "slk",
    "sl": "slv",
    "sr": "srp",
    "hr": "hrv",
    "lt": "lit",
    "lv": "lav",
    "et": "est",
    "zh-hans": "zho",
    "zh-hant": "zho",
}

LANG_DISPLAY = {
    "deu": "Deutsch",
    "eng": "English",
    "fra": "Français",
    "spa": "Español",
    "ita": "Italiano",
    "por": "Português",
    "rus": "Русский",
    "zho": "中文",
    "jpn": "日本語",
    "kor": "한국어",
    "pol": "Polski",
    "nld": "Nederlands",
    "tur": "Türkçe",
    "ces": "Čeština",
    "ell": "Ελληνικά",
    "ron": "Română",
    "ukr": "Українська",
    "fas": "فارسی",
    "heb": "עברית",
    "hin": "हिन्दी",
    "swe": "Svenska",
    "nor": "Norsk",
    "dan": "Dansk",
    "fin": "Suomi",
    "ind": "Bahasa Indonesia",
    "msa": "Bahasa Melayu",
    "ben": "বাংলা",
    "vie": "Tiếng Việt",
    "tha": "ไทย",
    "bul": "Български",
    "slk": "Slovenčina",
    "slv": "Slovenščina",
    "srp": "Српски",
    "hrv": "Hrvatski",
    "lit": "Lietuvių",
    "lav": "Latviešu",
    "est": "Eesti",
}


VIDEO_EXTENSIONS = [
    ".mp4",
    ".mkv",
    ".webm",
    ".avi",
    ".mov",
    ".flv",
    ".mpeg",
    ".mpg",
    ".gif",
]
AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".ogg"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
SUB_EXTENSIONS = {"srt", "ass", "vtt"}

FORMATS = ["mp4", "mkv", "avi", "mov", "webm", "mpeg", "mpg"]

EXT_TO_CONTAINER = {
    "mp4": "mp4",
    "m4v": "mp4",
    "mkv": "mkv",
    "matroska": "mkv",
    "webm": "webm",
    "mov": "mov",
    "qt": "mov",
    "avi": "avi",
    "flv": "flv",
    "mpg": "mpeg",
    "mpeg": "mpeg",
    "mpe": "mpeg",
    "vob": "mpeg",
}

# 4) Fallback-Policy je Codec-Familie (nur *gleiche Familie*, in sinnvoller Reihenfolge)
#    Diese Reihenfolge wird in deinen Try-&-Fallback-Routinen verwendet.
CODEC_FALLBACK_POLICY: Dict[str, List[str]] = {
    # H.264
    "h264": [
        "h264_videotoolbox",
        "h264_nvenc",
        "h264_qsv",
        "h264_vaapi",
        "h264_amf",
        "libx264",
        "libx264rgb",  # rgb-Variante ans Ende (Sonderfall)
    ],
    # HEVC
    "hevc": [
        "hevc_videotoolbox",
        "hevc_nvenc",
        "hevc_qsv",
        "hevc_vaapi",
        "hevc_amf",
        "libx265",
    ],
    # AV1
    "av1": [
        "av1_videotoolbox",
        "av1_nvenc",
        "av1_qsv",
        "av1_vaapi",
        "av1_amf",
        "libaom-av1",
        "libsvtav1",
        "rav1e",
    ],
    # VP9
    "vp9": ["libvpx-vp9", "vp9_qsv", "vp9_vaapi"],
    # VP8
    "vp8": ["vp8_qsv", "libvpx-vp8", "libvpx"],
    # MPEG-2
    "mpeg2video": ["mpeg2_qsv", "mpeg2_vaapi", "mpeg2video"],
    # MPEG-1
    "mpeg1video": ["mpeg1video"],
    # MPEG-4 Part 2
    "mpeg4": ["libxvid", "mpeg4"],
    # ProRes
    "prores": ["prores_ks", "prores_aw", "prores"],
    # DNxHD/R
    "dnxhd": ["dnxhd"],
    # JPEG2000
    "jpeg2000": ["jpeg2000", "libopenjpeg"],
    # MJPEG
    "mjpeg": ["mjpeg"],
    # FFV1
    "ffv1": ["ffv1"],
    # HuffYUV
    "huffyuv": ["huffyuv", "ffvhuff"],
    # UTVideo
    "utvideo": ["utvideo"],
    # Theora
    "theora": ["libtheora"],
    # QuickTime Animation
    "qtrle": ["qtrle"],
    # Hap
    "hap": ["hap"],
    # Raw
    "rawvideo": ["rawvideo"],
    # PNG (Video)
    "png": ["png"],
    # MagicYUV
    "magicyuv": ["magicyuv"],
    # CineForm
    "cineform": ["cfhd"],
    # FLV/Legacy
    "h263": ["h263"],
    "vp6": ["vp6"],
    "screenvideo": ["flashsv2", "flashsv"],
}

# Erzwinge konstante Bildrate (CFR) pro Codec
CODEC_ENFORCE_CFR: Dict[str, bool] = {
    # MJPEG braucht stabile Timestamps, sonst zeigen Player Standbilder
    "mjpeg": True,
}

# Erzwinge Pixelformat (container-spezifisch, "*" = Default)
CODEC_FORCED_PIX_FMT: Dict[str, Dict[str, str]] = {
    # MJPEG erwartet JFIF (full-range) Formate → yuvj422p ist am robustesten
    "mjpeg": {
        "*": "yuvj422p",
        # wenn du magst, für AVI/MOV explizit noch mal wiederholen:
        # "avi": "yuvj422p",
        # "mov": "yuvj422p",
    },
    # Sinnvolle Defaults für J2K in toleranten Containern (Playback-glatt)
    "jpeg2000": {
        "mkv": "yuv422p10le",
        "matroska": "yuv422p10le",
        "mov": "yuv422p10le",
    },
}

# Erzwinge Color-Range (Container-Tags); für MJPEG nur "pc" (full)
CODEC_FORCE_COLOR_RANGE: Dict[str, str] = {
    "mjpeg": "pc",
}

# Fallback-Priorität, falls der gewünschte Codec nicht verfügbar ist:
FALLBACK_BY_CONTAINER: Dict[str, Tuple[str, ...]] = {
    # sinnvolle Standard-Fallbacks pro Container
    "mp4": ("h264", "hevc", "av1", "mpeg4"),
    "m4v": ("h264", "hevc", "av1", "mpeg4"),
    "mkv": ("h264", "hevc", "av1", "vp9", "mpeg4", "utvideo", "ffv1", "qtrle"),
    "matroska": ("h264", "hevc", "av1", "vp9", "mpeg4", "utvideo", "ffv1"),
    "webm": ("vp9", "av1", "vp8"),
    "mov": ("qtrle", "prores", "dnxhd", "hevc", "ffv1", "h264"),
    "avi": ("mpeg4", "mjpeg", "huffyuv", "utvideo", "rawvideo", "magicyuv"),
    "flv": ("h263", "screenvideo"),  # VP6 meist decode-only
    "mpeg": ("mpeg2video", "mpeg1video"),
    "ts": ("h264", "hevc", "mpeg2video"),  # Transport-Streams: H.264/HEVC, sonst MPEG-2
}


VIDEO_CODECS: Dict[str, CodecInfo] = {
    "copy": {
        "name": "Stream Copy",
        "description": {
            "de": "Versucht Bitstream-Copy ohne Neukodierung (sofern Container und Streams kompatibel sind).",
            "en": "Attempts bitstream copy without re-encoding (as long as container and streams are compatible).",
        },
    },
    "h264": {
        "name": "H.264 / MPEG-4 AVC",
        "description": {
            "de": "Weit verbreitet mit guter Effizienz und breiter Hardware-Unterstützung. Mathematisch verlustfrei möglich.",
            "en": "Widely used with good efficiency and broad hardware support. Mathematically lossless is possible.",
        },
    },
    "hevc": {
        "name": "H.265 / HEVC",
        "description": {
            "de": (
                "Effizienter als H.264 (typisch 25–50 % geringere Bitrate bei ähnlicher Qualität). "
                "Benötigt neuere Geräte/Player und passende Lizenzierung. Mathematisch verlustfrei möglich."
            ),
            "en": (
                "More efficient than H.264 (typically 25–50% lower bitrate at similar quality). "
                "Requires newer devices/players and suitable licensing. Mathematically lossless is possible."
            ),
        },
    },
    "av1": {
        "name": "AOMedia Video 1 (AV1)",
        "description": {
            "de": (
                "Royalty-free und sehr effizient; Unterstützung wächst schnell in Browsern und Hardware. "
                "Mathematisch verlustfrei möglich."
            ),
            "en": (
                "Royalty-free and very efficient; support is rapidly growing in browsers and hardware. "
                "Mathematically lossless is possible."
            ),
        },
    },
    "vp9": {
        "name": "Google VP9",
        "description": {
            "de": (
                "Offener Web-Codec; häufig 20–50 % kleiner als H.264 bei ähnlicher Qualität. "
                "Typisch in WebM. Mathematisch verlustfrei möglich."
            ),
            "en": (
                "Open web codec; often 20–50% smaller than H.264 at similar quality. "
                "Typically used in WebM. Mathematically lossless is possible."
            ),
        },
    },
    "vp8": {
        "name": "Google VP8",
        "description": {
            "de": "Älterer offener Web-Codec (8-bit 4:2:0), sehr kompatibel. Typisch in WebM.",
            "en": "Older open web codec (8-bit 4:2:0) with high compatibility. Typically used in WebM.",
        },
    },
    "mpeg4": {
        "name": "MPEG-4 Part 2 (ASP)",
        "description": {
            "de": (
                "Älter als H.264, geringere Effizienz, aber einfach zu dekodieren. "
                "Historisch Xvid/DivX-Klasse, heute eher für Kompatibilität."
            ),
            "en": (
                "Older than H.264 with lower efficiency but easy to decode. "
                "Historically the Xvid/DivX class; today mostly used for compatibility."
            ),
        },
    },
    "mpeg2video": {
        "name": "MPEG-2 Video",
        "description": {
            "de": (
                "Klassischer Broadcast/DVD-Codec; robust, aber höhere Bitraten. "
                "Gängig in MPEG-TS/PS und Archiv-Workflows."
            ),
            "en": (
                "Classic broadcast/DVD codec; robust but requires higher bitrates. "
                "Common in MPEG-TS/PS and archival workflows."
            ),
        },
    },
    "mpeg1video": {
        "name": "MPEG-1 Video",
        "description": {
            "de": (
                "Älterer Standard (VCD/Frühzeit von MPEG), robust bei .mpg/.mpeg (MPEG-PS). "
                "Verwandt mit dem MPEG-PS-Container."
            ),
            "en": (
                "Older standard (VCD/early MPEG), robust in .mpg/.mpeg (MPEG-PS). "
                "Closely related to the MPEG-PS container."
            ),
        },
    },
    "prores": {
        "name": "Apple ProRes",
        "description": {
            "de": "Intraframe-Mezzanine für Schnitt/Grading (10-bit 4:2:2/4:4:4:4 abhängig vom Profil); Alpha möglich.",
            "en": "Intraframe mezzanine for editing/grading (10-bit 4:2:2/4:4:4:4 depending on profile); alpha supported.",
        },
    },
    "dnxhd": {
        "name": "Avid DNxHD / DNxHR",
        "description": {
            "de": "Intraframe-Mezzanine für Schnitt/Proxies.",
            "en": "Intraframe mezzanine for editing/proxies.",
        },
    },
    "jpeg2000": {
        "name": "JPEG 2000",
        "description": {
            "de": "Intraframe, verbreitet u. a. in DCP und Archivierung. Alpha möglich; mathematisch verlustfrei möglich.",
            "en": "Intraframe codec used e.g. in DCP and archiving. Alpha possible; mathematically lossless is possible.",
        },
    },
    "mjpeg": {
        "name": "Motion JPEG",
        "description": {
            "de": "Intraframe, robust und sehr kompatibel. Größere Dateien; sinnvoll für Capture/Intermediates.",
            "en": "Intraframe, robust and highly compatible. Produces larger files; useful for capture and intermediates.",
        },
    },
    "ffv1": {
        "name": "FFV1",
        "description": {
            "de": (
                "Verlustfreier Intraframe-Codec mit hoher Effizienz; beliebt für Archivierung. "
                "Unterstützt u. a. 10-bit und optionale Prüfsummen. Mathematisch verlustfrei; Alpha möglich."
            ),
            "en": (
                "Lossless intraframe codec with high efficiency; popular for preservation. "
                "Supports 10-bit and optional checksums. Mathematically lossless; alpha is possible."
            ),
        },
    },
    "huffyuv": {
        "name": "HuffYUV",
        "description": {
            "de": "Verlustfrei und sehr schnell; historisch weit verbreitet. Mathematisch verlustfrei und Alpha möglich.",
            "en": "Lossless and very fast; historically widespread. Mathematically lossless with possible alpha support.",
        },
    },
    "utvideo": {
        "name": "Ut Video",
        "description": {
            "de": (
                "Verlustfrei, schnell und plattformübergreifend. Beliebt als Intermediate; "
                "mathematisch verlustfrei möglich, unterstützt auch RGBA."
            ),
            "en": (
                "Lossless, fast and cross-platform. Popular as an intermediate; "
                "mathematically lossless is possible and RGBA is supported."
            ),
        },
    },
    "theora": {
        "name": "Xiph Theora",
        "description": {
            "de": "Freier älterer Videocodec, heute selten im Einsatz.",
            "en": "Free, older video codec that is rarely used today.",
        },
    },
    "qtrle": {
        "name": "QuickTime Animation (RLE)",
        "description": {
            "de": (
                "Intraframe mit Alpha-Unterstützung; gut für UI-Captures und Grafiken. "
                "Typisch im MOV-Container. Mathematisch verlustfrei möglich."
            ),
            "en": (
                "Intraframe codec with alpha support; good for UI captures and graphics. "
                "Typically used in the MOV container. Mathematically lossless is possible."
            ),
        },
    },
    "hap": {
        "name": "HAP",
        "description": {
            "de": (
                "GPU-freundlicher Intraframe-Codec für Echtzeit-Playback mit Alphaunterstützung. "
                "Weit verbreitet in Live-/Installations-Setups; meist als MOV."
            ),
            "en": (
                "GPU-friendly intraframe codec for real-time playback with alpha support. "
                "Widely used in live/installation setups; typically stored in MOV."
            ),
        },
    },
    "rawvideo": {
        "name": "Uncompressed (rawvideo)",
        "description": {
            "de": (
                "Unkomprimiert und maximal kompatibel. Sehr große Dateien; praktisch keine Encoder-Optionen. "
                "Verlustfrei per Definition und Alpha möglich (z. B. RGBA)."
            ),
            "en": (
                "Uncompressed and maximally compatible. Very large files and virtually no encoder options. "
                "Lossless by definition and alpha possible (e.g. RGBA)."
            ),
        },
    },
    "png": {
        "name": "PNG Video",
        "description": {
            "de": (
                "PNG-basiertes Intraframe-Video; verlustfrei und mit Alpha möglich. "
                "Geeignet für Grafiken/Compositing, aber deutlich größere Dateien. "
                "Mathematisch verlustfrei; Alpha möglich."
            ),
            "en": (
                "PNG-based intraframe video; lossless and supports alpha. "
                "Suitable for graphics/compositing, but produces much larger files. "
                "Mathematically lossless; alpha is possible."
            ),
        },
    },
    "magicyuv": {
        "name": "MagicYUV",
        "description": {
            "de": "Sehr schneller, mathematisch verlustfreier Intraframe-Codec. Alpha (RGBA) möglich.",
            "en": "Very fast, mathematically lossless intraframe codec. Alpha (RGBA) is supported.",
        },
    },
    "cineform": {
        "name": "GoPro CineForm (CFHD)",
        "description": {
            "de": "Intraframe-Codec für performantes Editing; 10-/12-bit YUV/RGB je nach Build. Alpha möglich (RGBA).",
            "en": "Intraframe codec for performant editing; 10/12-bit YUV/RGB depending on build. Alpha possible (RGBA).",
        },
    },
}


# ---- Kandidatenlisten (Priorität von links nach rechts) --------------------
# Hinweis: Reihenfolge bevorzugt macOS VideoToolbox > NVIDIA NVENC > Intel QSV > VAAPI > AMD AMF > Software.
CODEC_ENCODER_CANDIDATES: Dict[str, List[str]] = {
    "h264": [
        "h264_videotoolbox",
        "h264_nvenc",
        "h264_qsv",
        "h264_vaapi",
        "h264_amf",
        "libx264",
    ],
    "hevc": [
        "hevc_videotoolbox",
        "hevc_nvenc",
        "hevc_qsv",
        "hevc_vaapi",
        "hevc_amf",
        "libx265",
    ],
    "av1": [
        "av1_videotoolbox",
        "av1_nvenc",
        "av1_qsv",
        "av1_vaapi",
        "av1_amf",
        "libaom-av1",
        "libsvtav1",
        "rav1e",
    ],
    "vp9": ["libvpx-vp9", "vp9_qsv", "vp9_vaapi"],
    "vp8": ["vp8_qsv", "libvpx-vp8", "libvpx"],  # libvpx-v8/libvpx – beide abgedeckt
    "mpeg2video": ["mpeg2_qsv", "mpeg2_vaapi", "mpeg2video"],
    "mpeg4": ["libxvid", "mpeg4"],
    "prores": ["prores_ks", "prores_aw", "prores"],
    "dnxhd": ["dnxhd"],  # DNxHR via dnxhd-Encoder per Profilen
    "jpeg2000": ["jpeg2000", "libopenjpeg"],
    "mjpeg": ["mjpeg"],
    "ffv1": ["ffv1"],
    "huffyuv": ["huffyuv", "ffvhuff"],
    "utvideo": ["utvideo"],
    "theora": ["libtheora"],
    "qtrle": ["qtrle"],
    "hap": ["hap"],
    "rawvideo": ["rawvideo"],
    "png": ["png"],
    "magicyuv": ["magicyuv"],
    "cineform": ["cfhd"],
    "h263": ["h263"],
    "vp6": ["vp6"],  # meist decode-only vorhanden; Fallbacks greifen sowieso
    "screenvideo": ["flashsv2", "flashsv"],
    "mpeg1video": ["mpeg1video"],
}


# 5) Container-spezifische Overrides:
#    Wenn vorhanden, überschreibt das die Kandidatenliste für (Container, Codec-Key).
#    Sinnvoll, um Browser-/Player-Kompatibilität oder "Best practice" je Container zu erzwingen.
CONTAINER_CODEC_OVERRIDES: Dict[tuple[str, str], List[str]] = {
    # WEBM – nur VP8/9/AV1 erlaubt; Prioritäten SW→HW oder umgekehrt je Praxis:
    ("webm", "vp9"): ["libvpx-vp9", "vp9_qsv", "vp9_vaapi"],
    ("webm", "av1"): [
        "libaom-av1",
        "libsvtav1",
        "av1_qsv",
        "av1_vaapi",
        "av1_nvenc",
        "av1_videotoolbox",
    ],
    ("webm", "vp8"): ["libvpx-vp8", "libvpx", "vp8_qsv"],
    # MP4 – H.264/HEVC/AV1/MPEG-4:
    ("mp4", "h264"): [
        "h264_videotoolbox",
        "h264_nvenc",
        "h264_qsv",
        "h264_vaapi",
        "h264_amf",
        "libx264",
    ],
    ("mp4", "hevc"): [
        "hevc_videotoolbox",
        "hevc_nvenc",
        "hevc_qsv",
        "hevc_vaapi",
        "hevc_amf",
        "libx265",
    ],
    ("mp4", "av1"): [
        "av1_nvenc",
        "av1_qsv",
        "av1_videotoolbox",
        "av1_vaapi",
        "libaom-av1",
        "libsvtav1",
    ],
    ("mp4", "mpeg4"): ["mpeg4", "libxvid"],
    # MKV – tolerant: für "h264" z. B. SW ganz vorne (bessere Qualität/Features)
    ("mkv", "h264"): [
        "libx264",
        "h264_nvenc",
        "h264_qsv",
        "h264_vaapi",
        "h264_amf",
        "h264_videotoolbox",
    ],
    ("mkv", "hevc"): [
        "libx265",
        "hevc_nvenc",
        "hevc_qsv",
        "hevc_vaapi",
        "hevc_amf",
        "hevc_videotoolbox",
    ],
    ("mkv", "av1"): [
        "libaom-av1",
        "libsvtav1",
        "av1_qsv",
        "av1_vaapi",
        "av1_nvenc",
        "av1_videotoolbox",
    ],
    ("mkv", "vp9"): ["libvpx-vp9", "vp9_qsv", "vp9_vaapi"],
    ("mkv", "jpeg2000"): ["jpeg2000", "libopenjpeg"],
    # MOV – ProRes / DNx sind typisch
    ("mov", "prores"): ["prores_ks", "prores_aw", "prores"],
    ("mov", "dnxhd"): ["dnxhd"],
    ("mov", "jpeg2000"): ["libopenjpeg", "jpeg2000"],
    # AVI – klassische/kompatible Encoder
    ("avi", "mpeg4"): ["mpeg4", "libxvid"],  # FourCC XVID setzt dein Quirk
    ("avi", "mjpeg"): ["mjpeg"],
    ("avi", "huffyuv"): ["huffyuv", "ffvhuff"],
    ("avi", "utvideo"): ["utvideo"],
    ("avi", "rawvideo"): ["rawvideo"],
    # MPEG-PS/TS
    ("mpeg", "mpeg2video"): ["mpeg2_qsv", "mpeg2_vaapi", "mpeg2video"],
    ("mpeg", "mpeg1video"): ["mpeg1video"],
}

# HW_SUFFIXES = ("_nvenc", "_qsv", "_vaapi", "_videotoolbox", "_amf")
HW_SUFFIXES: tuple[str, ...] = ("_nvenc", "_qsv", "_vaapi", "_amf", "_videotoolbox")

CONVERT_FORMAT_DESCRIPTIONS: Dict[str, ConvertFormatSpec] = {
    "keep": {
        "name": "Original",
        "description": {
            "de": "Behält den Quellcontainer je Datei.",
            "en": "Keep the source container for each file.",
        },
    },
    "mp4": {
        "name": "🎥 MP4",
        "description": {
            "de": (
                "Unterstützt mehrere Audio- und Untertitelspuren, einige Metadaten, Kapitel und Thumbnail. "
                "Weit verbreitet, hohe Kompatibilität, ideal für Web und Mobile."
            ),
            "en": (
                "Supports multiple audio and subtitle tracks, some metadata, chapters and a thumbnail. "
                "Widely used, highly compatible, ideal for web and mobile."
            ),
        },
        "codecs": ["h264", "hevc", "av1", "mpeg4"],
    },
    "mkv": {
        "name": "🎞  MKV",
        "description": {
            "de": (
                "Unterstützt mehrere Audio- und Untertitelspuren, alle Metadaten, Kapitel, Thumbnail, "
                "Alpha-Kanal und die meisten Codecs; ideal für die Archivierung von multimedialen Videos."
            ),
            "en": (
                "Supports multiple audio and subtitle tracks, all metadata, chapters, thumbnail, "
                "alpha channel and most codecs; ideal for archiving rich multimedia videos."
            ),
        },
        "codecs": [
            "h264",
            "hevc",
            "av1",
            "vp9",
            "vp8",
            "mpeg2video",
            "mpeg4",
            "prores",
            "dnxhd",
            "jpeg2000",
            "mjpeg",
            "ffv1",
            "huffyuv",
            "utvideo",
            "theora",
            "png",
            "magicyuv",
            "cineform",
            "rawvideo",
        ],
    },
    "avi": {
        "name": "📼 AVI",
        "description": {
            "de": (
                "Unterstützt wenige Metadaten und Alpha-Kanal. Älteres Format mit größeren Datenvolumen "
                "und weniger effizienter Komprimierung."
            ),
            "en": (
                "Supports only a few metadata fields and alpha channel. Older format with larger file sizes "
                "and less efficient compression."
            ),
        },
        "codecs": [
            "mpeg4",
            "cineform",
            "mjpeg",
            "huffyuv",
            "utvideo",
            "hap",
            "rawvideo",
        ],
    },
    "mov": {
        "name": "🍏 MOV",
        "description": {
            "de": (
                "Unterstützt mehrere Audio- und Untertitelspuren, einige Metadaten, Kapitel und Alpha-Kanal. "
                "Apple-QuickTime-Format, hohe Qualität, am besten geeignet für macOS/iOS."
            ),
            "en": (
                "Supports multiple audio and subtitle tracks, some metadata, chapters and alpha channel. "
                "Apple QuickTime format with high quality, best suited for macOS/iOS."
            ),
        },
        "codecs": [
            "prores",
            "dnxhd",
            "h264",
            "hevc",
            "jpeg2000",
            "qtrle",
            "hap",
            "cineform",
            "mjpeg",
            "png",
        ],
    },
    "webm": {
        "name": "🌐 WEBM",
        "description": {
            "de": "Unterstützt alle Metadaten. Für die Verwendung im Internet optimiertes offenes Format.",
            "en": "Supports all metadata. Open format optimized for use on the web.",
        },
        "codecs": ["vp9", "vp8", "av1"],
    },
    "mpeg": {
        "name": "💿 MPEG",
        "description": {
            "de": "Früher Standard (MPEG-1/2), der in älteren Systemen und für DVDs verwendet wird.",
            "en": "Former standard (MPEG-1/2) used on older systems and for DVDs.",
        },
        "codecs": ["mpeg1video", "mpeg2video"],
    },
}


CONVERT_PRESET: Final[Dict[str, PresetSpec]] = {
    "messenger360p": {
        "name": "📱 Messenger 360p",
        "description": {
            "de": (
                "Sehr schnell, niedrige Qualität, 640px Breite (360p), "
                "kleine Datei für Chat-Apps (H.264/MP4 empfohlen)."
            ),
            "en": (
                "Very fast, low quality, 640px width (360p), "
                "small files for chat apps (H.264/MP4 recommended)."
            ),
        },
        "speed": "ultrafast",
        "quality": 30,
        "scale": "640:-2",
        "max_fps": 30,
        "faststart": True,
        "audio_bitrate": "128k",
        "audio_channels": 2,
        "force_yuv420p": True,
    },
    "messenger720p": {
        "name": "📱 Messenger 720p",
        "description": {
            "de": (
                "Schnelles Encoding, gute Dateigröße, 1280px Breite (720p); "
                "ideal für Messenger."
            ),
            "en": (
                "Fast encoding, good file size, 1280px width (720p); "
                "ideal for messengers."
            ),
        },
        "speed": "veryfast",
        "quality": 28,
        "scale": "1280:-2",
        "max_fps": 30,
        "faststart": True,
        "audio_bitrate": "160k",
        "audio_channels": 2,
        "force_yuv420p": True,
    },
    "web": {
        "name": "🌐 Web",
        "description": {
            "de": (
                "Schnelles Encoding, reduzierte Größe, maximale Kompatibilität "
                "und ordentliche Qualität für Online-Nutzung."
            ),
            "en": (
                "Fast encoding, reduced file size, maximum compatibility "
                "and solid quality for online use."
            ),
        },
        "speed": "veryfast",
        "quality": 26,
        "scale": None,
        "max_fps": None,
        "faststart": True,
        "audio_bitrate": "160k",
        "audio_channels": 2,
        "force_yuv420p": True,
    },
    "casual": {
        "name": "💻 Casual",
        "description": {
            "de": (
                "Ausgewogene Qualität, Geschwindigkeit und maximale Kompatibilität "
                "(für Alltags-Videos)."
            ),
            "en": (
                "Balanced quality, speed and maximum compatibility "
                "(for everyday videos)."
            ),
        },
        "speed": "medium",
        "quality": 22,
        "scale": None,
        "max_fps": None,
        "faststart": True,
        "audio_bitrate": "192k",
        "audio_channels": 2,
        "force_yuv420p": True,
    },
    "cinema": {
        "name": "🎬 Cinema",
        "description": {
            "de": (
                "Hohe Qualität, langsameres Encoding und hohe Kompatibilität "
                "(filmähnlicher Output)."
            ),
            "en": (
                "High quality, slower encoding and high compatibility "
                "(film-like output)."
            ),
        },
        "speed": "slow",
        "quality": 20,
        "scale": None,
        "max_fps": None,
        "faststart": False,
        "audio_bitrate": "256k",
        "audio_channels": 2,
        "prefer_10bit": True,
    },
    "studio": {
        "name": "🎛️ Studio",
        "description": {
            "de": (
                "Sehr hohe Qualität, ggf. auf Kosten der Kompatibilität, "
                "für Master/Archiv in praxisnaher Größe."
            ),
            "en": (
                "Very high quality, potentially at the expense of compatibility, "
                "for masters/archives in practical file sizes."
            ),
        },
        "speed": "slow",
        "quality": 18,
        "scale": None,
        "max_fps": None,
        "faststart": False,
        "audio_bitrate": "320k",
        "audio_channels": 2,
        "prefer_10bit": True,
    },
    "ultra": {
        "name": "🧪 Ultra",
        "description": {
            "de": (
                "Maximale visuelle Qualität, ohne Rücksicht auf Kompatibilität, "
                "ohne echtes Lossless."
            ),
            "en": (
                "Maximum visual quality, disregarding compatibility, "
                "but not truly lossless."
            ),
        },
        "speed": "slow",
        "quality": 14,
        "scale": None,
        "max_fps": None,
        "faststart": False,
        "audio_bitrate": "320k",
        "audio_channels": 2,
        "prefer_10bit": True,
    },
    "lossless": {
        "name": "🧊 Lossless",
        "description": {
            "de": (
                "Mathematisch verlustfrei (wo der Codec es unterstützt), sonst near-lossless, "
                "so gut es die Format-+Codec-Kombination zulässt. Sehr große Dateien, langsames Encoding."
            ),
            "en": (
                "Mathematically lossless (where supported by the codec), otherwise near-lossless "
                "as far as the format+codec combination allows. Very large files, slow encoding."
            ),
        },
        "lossless": True,
        "speed": "slow",
        "quality": None,
        "scale": None,
        "max_fps": None,
        "faststart": False,
        "audio_bitrate": "pcm",
        "audio_channels": 2,
    },
}


# Einheitliche Definition (Reihenfolge bleibt erhalten)
RESOLUTIONS: Dict[str, ResolutionSpec] = {
    "original": {
        "name": "Original",
        "description": {
            "de": "Original-Auflösung beibehalten, in Abhängigkeit von dem gewählten preset",
            "en": "Keep original resolution, depending on the selected preset",
        },
        "scale": None,
        "aspect": None,
    },
    "240p": {
        "name": "240p",
        "description": {"de": "426:240 (16:9)", "en": "426:240 (16:9)"},
        "scale": "426:240",
        "aspect": "16:9",
    },
    "360p": {
        "name": "360p",
        "description": {"de": "640:360 (16:9)", "en": "640:360 (16:9)"},
        "scale": "640:360",
        "aspect": "16:9",
    },
    "480p": {
        "name": "480p",
        "description": {"de": "854:480 (16:9)", "en": "854:480 (16:9)"},
        "scale": "854:480",
        "aspect": "16:9",
    },
    "720p": {
        "name": "720p",
        "description": {"de": "1280:720 (16:9)", "en": "1280:720 (16:9)"},
        "scale": "1280:720",
        "aspect": "16:9",
    },
    "1080p": {
        "name": "1080p (HD)",
        "description": {"de": "1920:1080 (16:9)", "en": "1920:1080 (16:9)"},
        "scale": "1920:1080",
        "aspect": "16:9",
    },
    "1440p": {
        "name": "1440p (QHD)",
        "description": {"de": "2560:1440 (16:9)", "en": "2560:1440 (16:9)"},
        "scale": "2560:1440",
        "aspect": "16:9",
    },
    "QHD+": {
        "name": "QHD+",
        "description": {
            "de": "3200:1800 (~16:9) [Laptop-Displays]",
            "en": "3200:1800 (~16:9) [laptop displays]",
        },
        "scale": "3200:1800",
        "aspect": "~16:9",
    },
    "4K": {
        "name": "4K",
        "description": {
            "de": "3840:2160 (16:9) 2160p Ultra High Definition",
            "en": "3840:2160 (16:9) 2160p Ultra High Definition",
        },
        "scale": "3840:2160",
        "aspect": "16:9",
    },
    "4K-DCI": {
        "name": "4K-DCI",
        "description": {
            "de": "4096:2160 (17:9) Digital Cinema Initiatives",
            "en": "4096:2160 (17:9) Digital Cinema Initiatives",
        },
        "scale": "4096:2160",
        "aspect": "17:9",
    },
    "8K": {
        "name": "8K",
        "description": {
            "de": "7680:4320 (16:9) 4320p Full Ultra High Definition",
            "en": "7680:4320 (16:9) 4320p Full Ultra High Definition",
        },
        "scale": "7680:4320",
        "aspect": "16:9",
    },
    "8K-DCI": {
        "name": "8K-DCI",
        "description": {
            "de": "8192:4320 (17:9) Digital Cinema Initiatives",
            "en": "8192:4320 (17:9) Digital Cinema Initiatives",
        },
        "scale": "8192:4320",
        "aspect": "17:9",
    },
    "custom": {
        "name": "Custom",
        "description": {
            "de": "Benutzerdefinierte Auflösung (Breite:Höhe).",
            "en": "Custom resolution (width:height).",
        },
        "scale": None,
        "aspect": None,
    },
}


FRAMERATES: Dict[str, FrameRateSpec] = {
    "original": {
        "name": "Original",
        "description": {
            "de": "Originale Bildrate beibehalten; vermeidet Ruckler durch Umrechnung und erhält die Bewegungscharakteristik der Quelle.",
            "en": "Keep the original frame rate; avoids cadence issues and preserves the source motion characteristics.",
        },
        "fps": None,
    },
    "15": {
        "name": "15 fps",
        "description": {
            "de": "Sehr geringe Bewegungsauflösung; geeignet für Überwachung, Vorschauen und extrem geringe Bandbreiten.",
            "en": "Very low motion resolution; suitable for surveillance, previews, and ultra-low bandwidth scenarios.",
        },
        "fps": "15",
    },
    "23.976": {
        "name": "23.976 fps",
        "description": {
            "de": "Klassische Film-Kadenz in NTSC-Umgebungen; vermittelt einen filmischen, ruhigen Bildeindruck.",
            "en": "Classic film cadence in NTSC contexts; delivers a cinematic, natural motion feel.",
        },
        "fps": "23.976",
    },
    "24": {
        "name": "24 fps",
        "description": {
            "de": "Kino-Standard; wirkt natürlich und filmisch, ideal für erzählerische Inhalte.",
            "en": "Cinema standard; natural, cinematic look ideal for narrative content.",
        },
        "fps": "24",
    },
    "25": {
        "name": "25 fps",
        "description": {
            "de": "PAL-/Europa-Standard in der TV-Produktion; guter Kompromiss aus Schärfe und Bewegungsdarstellung.",
            "en": "PAL/Europe TV standard; a solid balance between sharpness and motion portrayal.",
        },
        "fps": "25",
    },
    "29.97": {
        "name": "29.97 fps",
        "description": {
            "de": "NTSC-TV-Kadenz; etwas flüssiger als der Film-Look, verbreitet in TV- und Web-Workflows.",
            "en": "NTSC TV cadence; a touch smoother than the cinema look, common in TV and web workflows.",
        },
        "fps": "29.97",
    },
    "30": {
        "name": "30 fps",
        "description": {
            "de": "Gängig für Web und Mobile; bietet flüssige Wiedergabe bei hoher Kompatibilität.",
            "en": "Common for web and mobile; provides smooth playback with high compatibility.",
        },
        "fps": "30",
    },
    "50": {
        "name": "50 fps",
        "description": {
            "de": "Hohe Bewegungsauflösung im PAL-Bereich; ideal für Sport, Live-Produktion und schnelle Action.",
            "en": "High motion resolution in PAL regions; ideal for sports, live production, and fast action.",
        },
        "fps": "50",
    },
    "59.94": {
        "name": "59.94 fps",
        "description": {
            "de": "Hohe Bewegungsauflösung im NTSC-Bereich; beliebt für Sport, Gaming und Broadcast-Workflows.",
            "en": "High motion resolution in NTSC regions; popular for sports, gaming, and broadcast workflows.",
        },
        "fps": "59.94",
    },
    "60": {
        "name": "60 fps",
        "description": {
            "de": "Sehr flüssige Wiedergabe; häufig genutzt für YouTube, Gameplay und UI-Aufnahmen.",
            "en": "Very smooth playback; often used for YouTube, gameplay, and UI captures.",
        },
        "fps": "60",
    },
    "120": {
        "name": "120 fps",
        "description": {
            "de": "Extrem flüssige Darstellung; nützlich für Zeitlupen und Displays mit hoher Bildwiederholrate.",
            "en": "Extremely fluid motion; useful for slow-motion and high-refresh-rate displays.",
        },
        "fps": "120",
    },
    "custom": {
        "name": "Custom",
        "description": {
            "de": "Benutzerdefinierte Bildrate; flexibel für Spezialfälle und zum Angleichen an Projektspezifikationen.",
            "en": "User-defined frame rate; flexible for special cases and project-specific matching.",
        },
        "fps": None,
    },
}


# Sprach-Normalisierung (ein paar gängige Aliase)
LANG_ALIASES = {
    "de": "de",
    "ger": "de",
    "deu": "de",
    "german": "de",
    "deutsch": "de",
    "de-at": "de",
    "de-de": "de",
    "en": "en",
    "eng": "en",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "español": "es",
    "it": "it",
    "ita": "it",
    "italian": "it",
    "ja": "ja",
    "jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
}
BLACKLIST_LANG = {"", "und", "undetermined", "unknown", "xx", "zxx"}


EXTRACT_SUBTITLE_OPTIONS = {
    "german": {"de": "🇩🇪 Deutsch", "en": "🇩🇪 German"},
    "english": {"de": "🇬🇧 Englisch", "en": "🇬🇧 English"},
    "german+english": {
        "de": "🇩🇪 Deutsch + 🇬🇧 Englisch",
        "en": "🇩🇪 German + 🇬🇧 English",
    },
    "all": {"de": "Alle vorhandene Untertitel", "en": "All available subtitles"},
    "custom": {"de": "Benutzereingabe", "en": "User input"},
}

EXTRACT_MODE: Dict[str, ExtractModeSpec] = {
    "audio": {
        "name": "🔊 Audio",
        "description": {
            "de": "Alle Audiospuren extrahieren (MP3).",
            "en": "Extract all audio tracks (MP3).",
        },
        "formats": ["mp3"],
    },
    "subtitles": {
        "name": "💬 Subtitles",
        "description": {
            "de": "Untertitel extrahieren (alle oder gefiltert).",
            "en": "Extract subtitles (all or filtered).",
        },
        "formats": ["srt", "ass", "vtt"],
    },
    "frame": {
        "name": "🖼️ Frame",
        "description": {
            "de": "Ein einzelnes Frame extrahieren.",
            "en": "Extract a single frame.",
        },
        "formats": ["png", "jpg", "webp", "bmp"],
    },
    "video_only": {
        "name": "🎞️🔇 Video-only",
        "description": {
            "de": "Nur Video – Audio und Untertitel entfernen.",
            "en": "Video only — remove audio & subtitles.",
        },
    },
}

FRAME_TIMES_PRESET = {
    "1s": {
        "name": {"de": "1 Sekunde", "en": "1 second"},
        "description": {"de": "Bei der ersten Sekunde", "en": "at the first second"},
    },
    "60s": {
        "name": {"de": "1 Minute", "en": "1 minute"},
        "description": {"de": "Bei der ersten Minute", "en": "at the first minute"},
    },
    "1p": {
        "name": {"de": "1%", "en": "1%"},
        "description": {
            "de": "Bei einem Prozent des Videos",
            "en": "at one percent of the video",
        },
    },
    "10p": {
        "name": {"de": "10%", "en": "10%"},
        "description": {
            "de": "Bei 10 Prozent des Videos",
            "en": "at 10 percent of the video",
        },
    },
    "33p": {
        "name": {"de": "1/3", "en": "1/3"},
        "description": {
            "de": "Bei einem drittel des Videos",
            "en": "at one-third of the video",
        },
    },
    "50p": {
        "name": {"de": "50%", "en": "50%"},
        "description": {
            "de": "Bei der Hälfte des Videos",
            "en": "at the half of the video",
        },
    },
    "66p": {
        "name": {"de": "2/3", "en": "2/3"},
        "description": {
            "de": "Bei zwei drittel des Videos",
            "en": "at two-third of the video",
        },
    },
    "90p": {
        "name": {"de": "90%", "en": "90%"},
        "description": {
            "de": "Bei neunzig Prozent des Videos",
            "en": "at ninety percent of the video",
        },
    },
    "custom": {
        "name": {"de": "Individuelle Eingabe", "en": "individual input"},
        "description": {"de": "eigenen Zeitpunkt eingeben", "en": "enter custom time"},
    },
}


SUBTITLE_FORMATS: Dict[str, SubtitleFormatSpec] = {
    "srt": {
        "name": "📝 SRT",
        "description": {
            "de": "Einfaches Textformat mit Zeitstempeln; extrem weit kompatibel.",
            "en": "Simple text format with timestamps; extremely widely compatible.",
        },
        "extensions": ["srt"],
        "features": ["Plain text", "Broad compatibility", "Easy to edit"],
    },
    "ass": {
        "name": "🎨 ASS/SSA",
        "description": {
            "de": "Erweitertes Format mit Styling, Positionierung und Karaoke-Effekten.",
            "en": "Advanced format with styling, positioning, and karaoke effects.",
        },
        "extensions": ["ass", "ssa"],
        "features": ["Rich styling", "Precise positioning", "Karaoke effects"],
    },
    "vtt": {
        "name": "🌐 WebVTT",
        "description": {
            "de": "Web-Standard für HTML5-Player; unterstützt Basis-Styling und Kapitel.",
            "en": "Web standard for HTML5 players; supports basic styling and chapters.",
        },
        "extensions": ["vtt"],
        "features": ["HTML5 friendly", "Chapters & cues", "UTF-8 by default"],
    },
}


META_TAGS: Dict[str, MetaTagInfo] = {
    # -------- PROTECTED (nicht setzen) --------
    "major_brand": {
        "protected": True,
        "name": {"de": "Hauptmarke", "en": "Major brand"},
        "description": {
            "de": "Container-Markenkennung (MP4/QuickTime).",
            "en": "Container brand identifier (MP4/QuickTime).",
        },
    },
    "minor_version": {
        "protected": True,
        "name": {"de": "Minor-Version", "en": "Minor version"},
        "description": {
            "de": "Container-Versionskennzahl.",
            "en": "Container version indicator.",
        },
    },
    "compatible_brands": {
        "protected": True,
        "name": {"de": "Kompatible Marken", "en": "Compatible brands"},
        "description": {
            "de": "Liste kompatibler MP4/QuickTime-Marken.",
            "en": "List of compatible MP4/QuickTime brands.",
        },
    },
    "encoder": {
        "protected": True,
        "name": {"de": "Encoder", "en": "Encoder"},
        "description": {
            "de": "Software/Encoder, der die Datei erzeugt hat.",
            "en": "Software/encoder that produced the file.",
        },
    },
    # NEW: Container-/Aufnahme-Zeitstempel
    "creation_time": {
        "protected": True,
        "name": {"de": "Erstellungszeit", "en": "Creation time"},
        "description": {
            "de": "Zeitstempel der Aufnahme/Dateierstellung aus dem Container (UTC, ISO-8601).",
            "en": "Recording/file creation timestamp from the container (UTC, ISO-8601).",
        },
    },
    # Häufig bei MP4/MOV: Encoding-Zeitpunkt (vom Muxer gesetzt)
    "encoded_date": {
        "protected": True,
        "name": {"de": "Kodierzeit", "en": "Encoded date"},
        "description": {
            "de": "Zeitpunkt der Kodierung/Muxing (falls vorhanden, UTC/ISO-8601).",
            "en": "Time of encoding/muxing (if present, UTC/ISO-8601).",
        },
    },
    # Häufig in Kamera-Files (falls vorhanden), sinnvoll als protected
    "make": {
        "protected": True,
        "name": {"de": "Gerätemarke", "en": "Device make"},
        "description": {
            "de": "Hersteller des Aufnahmegeräts (Kamera/Telefon).",
            "en": "Manufacturer of the recording device (camera/phone).",
        },
    },
    "model": {
        "protected": True,
        "name": {"de": "Gerätemodell", "en": "Device model"},
        "description": {
            "de": "Modellbezeichnung des Aufnahmegeräts.",
            "en": "Model name of the recording device.",
        },
    },
    # -------- EDITABLE --------
    "title": {
        "protected": False,
        "name": {"de": "Titel", "en": "Title"},
        "description": {
            "de": "Anzeigename des Inhalts/der Datei.",
            "en": "Display title of the content/file.",
        },
    },
    "show": {
        "protected": False,
        "name": {"de": "Serie/Show", "en": "Show"},
        "description": {
            "de": "Name der TV-Serie oder Show.",
            "en": "TV show or series name.",
        },
    },
    "season_number": {
        "protected": False,
        "name": {"de": "Staffel", "en": "Season"},
        "description": {
            "de": "Staffelnummer, z. B. 1.",
            "en": "Season number, e.g., 1.",
        },
    },
    "episode_id": {
        "protected": False,
        "name": {"de": "Episode", "en": "Episode"},
        "description": {
            "de": "Episodenkennung/-nummer.",
            "en": "Episode identifier/number.",
        },
    },
    "director": {
        "protected": False,
        "name": {"de": "Regie", "en": "Director"},
        "description": {
            "de": "Regisseur/Regisseurin.",
            "en": "Director of the content.",
        },
    },
    "production_year": {
        "protected": False,
        "name": {"de": "Produktionsjahr", "en": "Production year"},
        "description": {
            "de": "Jahr der Produktion oder Fertigstellung.",
            "en": "Year of production or completion.",
        },
    },
    "network": {
        "protected": False,
        "name": {"de": "Sender/Netzwerk", "en": "Network"},
        "description": {
            "de": "Ausstrahlender Sender/Streaming-Netzwerk.",
            "en": "Broadcast network/streaming service.",
        },
    },
    "artist": {
        "protected": False,
        "name": {"de": "Künstler", "en": "Artist"},
        "description": {
            "de": "Urheber/Performer/Creator.",
            "en": "Creator or performer.",
        },
    },
    "album": {
        "protected": False,
        "name": {"de": "Album", "en": "Album"},
        "description": {
            "de": "Zugehöriges Album/Verbundtitel.",
            "en": "Associated album/collection.",
        },
    },
    "date": {
        "protected": False,
        "name": {"de": "Datum", "en": "Date"},
        "description": {
            "de": "Veröffentlichungsdatum (frei formatiert).",
            "en": "Release date (free-form text).",
        },
    },
    "track": {
        "protected": False,
        "name": {"de": "Tracknummer", "en": "Track number"},
        "description": {
            "de": "Position im Album/Playlist.",
            "en": "Position in album/playlist.",
        },
    },
    "composer": {
        "protected": False,
        "name": {"de": "Komponist", "en": "Composer"},
        "description": {
            "de": "Komponist/in des Inhalts.",
            "en": "Composer of the content.",
        },
    },
    "publisher": {
        "protected": False,
        "name": {"de": "Publisher", "en": "Publisher"},
        "description": {
            "de": "Verlag/Distributor/Publisher.",
            "en": "Publisher or distributor.",
        },
    },
    "genre": {
        "protected": False,
        "name": {"de": "Genre", "en": "Genre"},
        "description": {
            "de": "Genre, z. B. Drama, Sci-Fi.",
            "en": "Genre, e.g., Drama, Sci-Fi.",
        },
    },
    "language": {
        "protected": False,
        "name": {"de": "Sprache", "en": "Language"},
        "description": {
            "de": "Inhaltssprache (frei/Code).",
            "en": "Content language (free/code).",
        },
    },
    "comment": {
        "protected": False,
        "name": {"de": "Kommentar", "en": "Comment"},
        "description": {
            "de": "Freitext-Kommentar.",
            "en": "Free-text comment.",
        },
    },
    "keywords": {
        "protected": False,
        "name": {"de": "Schlüsselwörter", "en": "Keywords"},
        "description": {
            "de": "Suchschlüsselwörter (kommagetrennt).",
            "en": "Search keywords (comma-separated).",
        },
    },
    # -------- NEW EDITABLE (nützlich in der Praxis) --------
    "description": {
        "protected": False,
        "name": {"de": "Beschreibung", "en": "Description"},
        "description": {
            "de": "Kurzbeschreibung/Plot/Synopsis.",
            "en": "Short description/plot/synopsis.",
        },
    },
    "copyright": {
        "protected": False,
        "name": {"de": "Urheberrecht", "en": "Copyright"},
        "description": {
            "de": "Copyright-/Lizenzhinweise.",
            "en": "Copyright/licensing notes.",
        },
    },
    "rating": {
        "protected": False,
        "name": {"de": "Altersfreigabe", "en": "Rating"},
        "description": {
            "de": "Bewertung/FSK/MPAA/TV-Rating (z. B. FSK 12, PG-13).",
            "en": "Content rating (e.g., FSK 12, PG-13).",
        },
    },
    "album_artist": {
        "protected": False,
        "name": {"de": "Album-Künstler", "en": "Album artist"},
        "description": {
            "de": "Übergeordneter Künstler/Interpret für Sammlungen.",
            "en": "Top-level artist for album/collections.",
        },
    },
    "track_total": {
        "protected": False,
        "name": {"de": "Tracks gesamt", "en": "Track total"},
        "description": {
            "de": "Gesamtanzahl der Tracks (z. B. 10).",
            "en": "Total number of tracks (e.g., 10).",
        },
    },
    "disc": {
        "protected": False,
        "name": {"de": "Disc", "en": "Disc"},
        "description": {
            "de": "Datenträger/Disc-Nummer (z. B. 1).",
            "en": "Disc number (e.g., 1).",
        },
    },
    "disc_total": {
        "protected": False,
        "name": {"de": "Discs gesamt", "en": "Disc total"},
        "description": {
            "de": "Gesamtanzahl der Discs (z. B. 2).",
            "en": "Total number of discs (e.g., 2).",
        },
    },
    "website": {
        "protected": False,
        "name": {"de": "Webseite", "en": "Website"},
        "description": {
            "de": "Offizielle Website/URL zum Inhalt.",
            "en": "Official website/URL for the content.",
        },
    },
    "encoded_by": {
        "protected": False,
        "name": {"de": "Kodiert von", "en": "Encoded by"},
        "description": {
            "de": "Freitext-Hinweis, wer/was kodiert hat.",
            "en": "Free-form note who/what encoded the file.",
        },
    },
    "location": {
        "protected": False,
        "name": {"de": "Ort", "en": "Location"},
        "description": {
            "de": "Aufnahmeort (frei; ggf. als Ort, Land).",
            "en": "Recording location (free form).",
        },
    },
}


# Lokalisiertes Label/Desc für virtuelle Tags (nur Anzeige in --list-tagnames)
VIRTUAL_META_INFO: Dict[str, VirtualMetaInfo] = {
    "container": {
        "name": {"de": "Container", "en": "Container"},
        "description": {
            "de": "Containerformat (Dateiendung)",
            "en": "Container format (file extension)",
        },
    },
    "duration": {
        "name": {"de": "Dauer", "en": "Duration"},
        "description": {
            "de": "Gesamtlaufzeit (HH:MM:SS)",
            "en": "Total runtime (HH:MM:SS)",
        },
    },
    "resolution": {
        "name": {"de": "Auflösung", "en": "Resolution"},
        "description": {
            "de": "Breite×Höhe des Videostreams",
            "en": "Video stream width×height",
        },
    },
    "codec": {
        "name": {"de": "Codec", "en": "Codec"},
        "description": {
            "de": "Videocodec des ersten Streams",
            "en": "Codec of the first video stream",
        },
    },
    "fps": {
        "name": {"de": "FPS", "en": "FPS"},
        "description": {
            "de": "Bildrate des ersten Streams",
            "en": "Frame rate of the first video stream",
        },
    },
    "pixel_format": {
        "name": {"de": "Pixelformat", "en": "Pixel format"},
        "description": {
            "de": "Pixelformat des ersten Video-Streams (ffprobe: pix_fmt).",
            "en": "Pixel format of the first video stream (ffprobe: pix_fmt).",
        },
    },
    # Datei / Container
    "filename": {
        "name": {"de": "Dateiname", "en": "Filename"},
        "description": {
            "de": "Dateiname ohne Endung.",
            "en": "File name without extension.",
        },
    },
    "file_size": {
        "name": {"de": "Dateigröße", "en": "File size"},
        "description": {
            "de": "Dateigröße in menschenlesbarem Format.",
            "en": "File size in human-readable form.",
        },
    },
    "container_long": {
        "name": {"de": "Container (Langname)", "en": "Container (long name)"},
        "description": {
            "de": "Langer Containername laut ffprobe.",
            "en": "Long container name reported by ffprobe.",
        },
    },
    "thumbnail": {
        "name": {"de": "Thumbnail vorhanden", "en": "Has thumbnail"},
        "description": {
            "de": "Ob ein eingebettetes Vorschaubild vorhanden ist.",
            "en": "Whether an embedded thumbnail exists.",
        },
    },
    # Video-Geometrie
    "display_resolution": {
        "name": {"de": "Display-Auflösung", "en": "Display resolution"},
        "description": {
            "de": "Angezeigte Breite×Höhe (unter Berücksichtigung von SAR/DAR).",
            "en": "Displayed width×height (respecting SAR/DAR).",
        },
    },
    "sar": {
        "name": {"de": "SAR", "en": "SAR"},
        "description": {
            "de": "Sample Aspect Ratio (z. B. 1:1, 64:45).",
            "en": "Sample Aspect Ratio (e.g., 1:1, 64:45).",
        },
    },
    "dar": {
        "name": {"de": "DAR", "en": "DAR"},
        "description": {
            "de": "Display Aspect Ratio (z. B. 16:9).",
            "en": "Display Aspect Ratio (e.g., 16:9).",
        },
    },
    # Alpha/Transparenz
    "alpha_channel": {
        "name": {"de": "Alpha-Kanal", "en": "Alpha channel"},
        "description": {
            "de": "Ob das Pixelformat Alpha grundsätzlich unterstützt.",
            "en": "Whether the pixel format supports alpha.",
        },
    },
    "transparency": {
        "name": {"de": "Transparenz (Inhalt)", "en": "Transparency (content)"},
        "description": {
            "de": "Ob innerhalb der ersten Frames echte Transparenz vorkommt.",
            "en": "Whether actual transparency occurs within the first frames.",
        },
    },
    # Audio-Quick-Summary (vom Default-/Hauptstream)
    "audio_codec": {
        "name": {"de": "Audio-Codec", "en": "Audio codec"},
        "description": {
            "de": "Codec des Standard-Audiostreams.",
            "en": "Codec of the default audio stream.",
        },
    },
    "audio_bitrate": {
        "name": {"de": "Audio-Bitrate", "en": "Audio bitrate"},
        "description": {
            "de": "Bitrate des Standard-Audiostreams.",
            "en": "Bit rate of the default audio stream.",
        },
    },
    "audio_channels": {
        "name": {"de": "Audio-Kanäle", "en": "Audio channels"},
        "description": {
            "de": "Kanalanzahl/-Layout des Standard-Audiostreams.",
            "en": "Channel count/layout of the default audio stream.",
        },
    },
    "audio_sample_rate": {
        "name": {"de": "Sample-Rate", "en": "Sample rate"},
        "description": {
            "de": "Sample-Rate des Standard-Audiostreams.",
            "en": "Sample rate of the default audio stream.",
        },
    },
    "audio_language": {
        "name": {"de": "Sprache (Audio)", "en": "Language (audio)"},
        "description": {
            "de": "Sprachcode/-name des Standard-Audiostreams.",
            "en": "Language code/name of the default audio stream.",
        },
    },
    # Stream-Zähler
    "audio_streams": {
        "name": {"de": "Audio-Streams", "en": "Audio streams"},
        "description": {
            "de": "Anzahl der Audiostreams im Container.",
            "en": "Number of audio streams in the container.",
        },
    },
    "subtitle_streams": {
        "name": {"de": "Untertitel-Streams", "en": "Subtitle streams"},
        "description": {
            "de": "Anzahl der Untertitelstreams im Container.",
            "en": "Number of subtitle streams in the container.",
        },
    },
    "primaries": {
        "name": {"de": "Primaries", "en": "Primaries"},
        "description": {
            "de": "Farbauflösung Primaries (ffprobe: color_primaries).",
            "en": "Color primaries (ffprobe: color_primaries).",
        },
    },
    "trc": {
        "name": {"de": "Transfer/TrC", "en": "Transfer/TrC"},
        "description": {
            "de": "Transferkennlinie/TrC (ffprobe: color_transfer).",
            "en": "Transfer characteristic/TrC (ffprobe: color_transfer).",
        },
    },
    "matrix": {
        "name": {"de": "Matrix", "en": "Matrix"},
        "description": {
            "de": "Farbraum-Matrix (ffprobe: color_space).",
            "en": "Color matrix (ffprobe: color_space).",
        },
    },
    "range": {
        "name": {"de": "Range", "en": "Range"},
        "description": {
            "de": "Signalbereich (ffprobe: color_range; tv/pc).",
            "en": "Signal range (ffprobe: color_range; tv/pc).",
        },
    },
    "chapters": {
        "name": {"de": "Kapitel", "en": "Chapters"},
        "description": {
            "de": "Anzahl der Kapitel (aus dem Container).",
            "en": "Number of chapters (from container).",
        },
    },
}

PROTECTED_META_KEYS = {k for k, s in META_TAGS.items() if s["protected"]}
EDITABLE_META_KEYS = [k for k, s in META_TAGS.items() if not s["protected"]]


ENHANCE_PRESETS: Dict[str, EnhancePresetSpec] = {
    "soft": {
        "name": {"de": "Sanft", "en": "Soft"},
        "description": {
            "de": "Sehr leichte Korrektur: Minimaler Brightness-/Contrast-Boost, sanfte Stabilisierung & leichtes Denoising.",
            "en": "Very light correction: minimal brightness/contrast boost, gentle stabilization & light denoise.",
        },
        "stabilize": True,
        "stab_method": "vidstab",
        "stab_smooth": 15,  # 0–100 Skala
        "denoise": True,
        "denoise_method": "hqdn3d",
        "denoise_intensity": 10,  # 0–100 Skala
        "warmth": 50,
        "tint": 50,
        "brightness": 52,
        "contrast": 52,
        "saturation": 52,
    },
    "realistic": {
        "name": {"de": "Realistisch", "en": "Realistic"},
        "description": {
            "de": "Realistische Optimierung: normalize + Standard-Stabilisierung & mittleres Denoising.",
            "en": "Realistic optimization: normalize + standard stabilization & medium denoise.",
        },
        "filter_chain": "normalize",
        "stabilize": True,
        "stab_method": "vidstab",
        "stab_smooth": 25,
        "denoise": True,
        "denoise_method": "nlmeans",
        "denoise_intensity": 20,
        "warmth": 50,
        "tint": 50,
        "brightness": 50,
        "contrast": 50,
        "saturation": 51,
    },
    "max": {
        "name": {"de": "Maximal verbessert", "en": "Maximum enhanced"},
        "description": {
            "de": "Maximale Verbesserung: normalize + Histogramm-Equalizer, starke Stabilisierung & Denoising.",
            "en": "Maximum improvement: normalize + histogram equalizer, aggressive stabilization & denoise.",
        },
        "filter_chain": "normalize,histeq=strength=0.5",
        "stabilize": True,
        "stab_method": "vidstab",
        "stab_smooth": 60,
        "denoise": True,
        "denoise_method": "nlmeans",
        "denoise_intensity": 40,
        "warmth": 52,
        "tint": 49,
        "brightness": 51,
        "contrast": 51,
        "saturation": 56,
    },
    "color_levels": {
        "name": {"de": "Color Levels Boost", "en": "Color Levels Boost"},
        "description": {
            "de": "Leicht erhöhter Dynamikumfang durch angehobene Levels.",
            "en": "Slight boost of dynamic range via raised levels.",
        },
        "stabilize": False,
        "denoise": False,
        "warmth": 50,
        "tint": 50,
        "brightness": 55,
        "contrast": 55,
        "saturation": 50,
    },
    "cinematic": {
        "name": {"de": "Cinematic Curves", "en": "Cinematic Curves"},
        "description": {
            "de": "Cinematic Look: sanfte Kurvenanpassung über Slider nachgeahmt.",
            "en": "Cinematic look: gentle curve adjustment emulated via sliders.",
        },
        "stabilize": False,
        "denoise": False,
        "warmth": 48,
        "tint": 52,
        "brightness": 52,
        "contrast": 48,
        "saturation": 50,
    },
    "hist_eq": {
        "name": {"de": "Histogram Equalizer", "en": "Histogram Equalizer"},
        "description": {
            "de": "Automatische Histogramm-Equalization nachgeahmt über Slider.",
            "en": "Histogram equalization emulated via sliders.",
        },
        "stabilize": False,
        "denoise": False,
        "warmth": 50,
        "tint": 50,
        "brightness": 54,
        "contrast": 56,
        "saturation": 50,
    },
    "vibrance": {
        "name": {"de": "Vibrance Boost", "en": "Vibrance Boost"},
        "description": {
            "de": "Selektiver Sättigungs-Boost über den Saturation-Slider.",
            "en": "Selective saturation boost via the saturation slider.",
        },
        "stabilize": False,
        "denoise": False,
        "warmth": 50,
        "tint": 50,
        "brightness": 50,
        "contrast": 50,
        "saturation": 65,
    },
    "stabilize_only": {
        "name": {"de": "Nur Stabilisierung", "en": "Stabilization Only"},
        "description": {
            "de": "Nur Stabilisierung, keine weiteren Filter.",
            "en": "Stabilization only, no other filters.",
        },
        "stabilize": True,
        "stab_method": "vidstab",
        "stab_smooth": 50,
        "denoise": False,
        "warmth": 50,
        "tint": 50,
        "brightness": 50,
        "contrast": 50,
        "saturation": 50,
    },
    "denoise_only": {
        "name": {"de": "Nur Denoise", "en": "Denoise Only"},
        "description": {
            "de": "Nur Denoising, keine weiteren Filter.",
            "en": "Denoising only, no other filters.",
        },
        "stabilize": False,
        "denoise": True,
        "denoise_method": "nlmeans",
        "denoise_intensity": 30,
        "warmth": 50,
        "tint": 50,
        "brightness": 50,
        "contrast": 50,
        "saturation": 50,
    },
    "custom": {  # ← neuer Key!
        "virtual": True,  # Marker: kein echter Filter-Satz
        "name": {"de": "Manuelle Konfiguration", "en": "Custom settings"},
        "description": {
            "de": "Konfiguriere alle Werte nach eigenen Ermessen",
            "en": "Lets you adjust every value/filter manually.",
        },
    },
}

# keys: trigger-param (Flag oder Auswahl), value: dict von 'value'→zugehörige Sub-Parameter
# Welche Color-Keys als "Default=50 -> ausblenden" gelten:
COLOR_PARAMS_DEFAULT_SKIP = {"warmth", "tint", "brightness", "contrast", "saturation"}

RELEVANT_PARAM_GROUPS: RelevantParamGroups = {
    "general": {
        "preset",
        "format",
        "duration",
        "total_duration",
        "resize_mode",
        "quality",
        "precision",
        "output",
    },
    "video": {
        "video_device",
        "video_codec",
        "encoder",
        "resolution",
        "scale",
        "framerate",
        "offset",
        "deint",
    },
    "audio": {"audio_source", "audio_codec"},
    "container": {"container", "faststart", "lossless"},
    "time": {"start_time", "end_time"},
    # Stabilisierung: Methode immer zeigen; je nach Methode weitere Keys
    "stabilize": {
        True: {
            # WICHTIG: 'stab_method' selbst anzeigen
            "stab_method": {
                "vidstab": ["stab_smooth"],  # z. B. "Glättungsstärke"
                "deshake": ["stab_rx", "stab_ry"],  # horizont./vertikal
            }
        },
        False: {},
    },
    # Denoise: Methode + Intensität immer zeigen, wenn aktiv
    "denoise": {True: ["denoise_method", "denoise_intensity"], False: []},
    # NEU: Color – alle Regler, aber die print-Funktion blendet 50% weg
    "color": {"warmth", "tint", "brightness", "contrast", "saturation"},
}


STABILIZATION: Dict[str, StabilizationSpec] = {
    "vidstab": {
        "default": 20,
        "description": {
            "de": "Fortschrittlicher Stabilisator mit viel Kontrolle.",
            "en": "Advanced stabilizer with plenty of control.",
        },
    },
    "deshake": {
        "defaultx": 15,
        "defaulty": 10,
        "description": {
            "de": "Sehr einfach, schnell, für leichte Korrektur.",
            "en": "Very simple, fast, for minor correction.",
        },
    },
}


NOISE_REDUCTION: Dict[str, NoiseReductionSpec] = {
    "hqdn3d": {
        "default": 15,
        "description": {
            "de": "Sehr schnell, gut für leichte/mittlere Korrektur.",
            "en": "Very fast, good for light/medium correction.",
        },
    },
    "nlmeans": {
        "default": 8,
        "description": {
            "de": "Sehr effektiv, für stark verrauschte Videos, aber langsam.",
            "en": "Very effective for videos with a lot of noise, but slow.",
        },
    },
}


# ---------------------------------
# ---- gif font size presets ------

MEME_FONTSIZE: Dict[str, MemeFontSizeSpec] = {
    "thiny": {
        "name": {"de": "Winzig", "en": "Thiny"},
        "size": 15,
    },
    "small": {
        "name": {"de": "Klein", "en": "Small"},
        "size": 28,
    },
    "medium": {
        "name": {"de": "Mittel", "en": "Medium"},
        "size": 43,
    },
    "grande": {
        "name": {"de": "Mittelgroß", "en": "grande"},
        "size": 61,
    },
    "large": {
        "name": {"de": "Groß", "en": "Large"},
        "size": 82,
    },
    "huge": {
        "name": {"de": "Riesig", "en": "Huge"},
        "size": 105,
    },
}

# ---------------------------------------------
# Supported AI-Models (unchanged labels)
# ---------------------------------------------
SUPPORTED_MODELS = {
    "realesr-general-x4v3": "Universal 4× (Foto, Comic, KI-Art) – DN regelbar",
    "RealESRGAN_x4plus_anime_6B": "Anime / Cartoon 4×",
    "RealESRGAN_x4plus": "Foto / Real-World 4×",
    "RealESRGAN_x2plus": "Foto / Real-World 2×",
    "realesr-animevideov3": "Anime Video (Torch/NCNN) 4×/3×/2×",
    "realcugan-se": "RealCUGAN (models-se) – Anime, konservativ/variantenreich",
    "realcugan-pro": "RealCUGAN (models-pro) – Anime, stärker (pro)",
    "realcugan-nose": "RealCUGAN (models-nose) – Anime, feine Kanten (no-se)",
}

# ---------------------------------------------
# MODEL_META with explicit backends & priorities
# ---------------------------------------------
# Kombiniertes MODEL_META – original Felder bleiben erhalten,
# zusätzlich pro Eintrag: "caps" aus MODEL_CAPS.

MODEL_META: Dict[str, Dict[str, Any]] = {
    "realesr-general-x4v3": {
        "desc": {
            "de": "4× Allround (Foto/Video, Streams, Screencasts) – Standardmodell für gemischtes Realmaterial; vergleichsweise ressourcenschonendes 'tiny'-Modell mit regelbarer Entrauschung. Gut geeignet für längere Videos und Mittelklasse-GPUs als Default.",
            "en": "4× all-round model (photo/video, streams, screencasts) – default model for mixed real-world material; comparatively resource-friendly “tiny” model with adjustable denoising. Well suited as default for longer videos and mid-range GPUs.",
        },
        "supports_dn": True,
        "weight": "realesr-general-x4v3.pth",
        "available_backends": ["pytorch"],
        "preferred_backends": ["pytorch"],
        "supports_tta": False,
        "tta_backends": [],
        "caps": {
            "denoise": {"type": "strength", "range": [0.0, 1.0]},
            "face_enhance": {"native": False, "via": ["gfpgan", "codeformer"]},
            "scale": {
                "pytorch": {"mode": "arbitrary_outscale", "model_scale": 4},
                "ncnn": None,
            },
            "noise_levels": None,
        },
    },
    "RealESRGAN_x4plus": {
        "desc": {
            "de": "4× klassisches Foto/Video-Modell (large) – sehr detailstark für hochwertige Real-Videos; deutlich ressourcenintensiver als realesr-general-x4v3.",
            "en": "4× classic photo/video model (large) – very high detail for high-quality real videos; significantly more resource-intensive than realesr-general-x4v3.",
        },
        "supports_dn": False,
        "weight": "RealESRGAN_x4plus.pth",
        "ncnn_name": "realesrgan-x4plus",
        "available_backends": ["pytorch", "ncnn"],
        "preferred_backends": ["pytorch", "ncnn"],
        "supports_tta": True,
        "tta_backends": ["ncnn"],
        "caps": {
            "denoise": {"type": "none"},
            "face_enhance": {"native": False, "via": ["gfpgan", "codeformer"]},
            "scale": {
                "pytorch": {"mode": "arbitrary_outscale", "model_scale": 4},
                "ncnn": {"mode": "discrete", "values": [2, 3, 4], "tta": True},
            },
            "noise_levels": None,
        },
    },
    "RealESRGAN_x2plus": {
        "desc": {
            "de": "2× Foto/Video – moderates Upscaling mit klarer Schärfeverbesserung bei deutlich geringerem Ressourcenbedarf als bei den 4×-Modelle.",
            "en": "2× photo/video – moderate upscaling with clear sharpness improvement and noticeably lower resource usage than the 4× models.",
        },
        "supports_dn": False,
        "weight": "RealESRGAN_x2plus.pth",
        "available_backends": ["pytorch"],
        "preferred_backends": ["pytorch"],
        "supports_tta": False,
        "tta_backends": [],
        "caps": {
            "denoise": {"type": "none"},
            "face_enhance": {"native": False, "via": ["gfpgan", "codeformer"]},
            "scale": {"pytorch": {"mode": "fixed", "model_scale": 2}, "ncnn": None},
            "noise_levels": None,
        },
    },
    "RealESRGAN_x4plus_anime_6B": {
        "desc": {
            "de": "4× Anime/Line-Art (large) – sehr detailreiches Modell für hochwertige Animes; liefert extrem scharfe Kanten, ist aber stark VRAM- und rechenintensiv.",
            "en": "4× anime/line art (large) – very detailed model for high-quality anime; produces extremely sharp edges but is very demanding in terms of VRAM and compute.",
        },
        "supports_dn": False,
        "weight": "RealESRGAN_x4plus_anime_6B.pth",
        "ncnn_name": "realesrgan-x4plus-anime",
        "available_backends": ["pytorch", "ncnn"],
        "preferred_backends": ["pytorch", "ncnn"],
        "supports_tta": True,
        "tta_backends": ["ncnn"],
        "caps": {
            "denoise": {"type": "none"},
            "face_enhance": {"native": False, "via": ["gfpgan", "codeformer"]},
            "scale": {
                "pytorch": {"mode": "arbitrary_outscale", "model_scale": 4},
                "ncnn": {"mode": "discrete", "values": [2, 3, 4], "tta": True},
            },
            "noise_levels": None,
        },
    },
    "realesr-animevideov3": {
        "desc": {
            "de": "2×/3×/4× Anime-Video v3 (tiny) – sehr stabil auf komprimiertem Material und vergleichsweise ressourcenschonend, ideal für längere Videos.",
            "en": "2×/3×/4× anime-video v3 (tiny) – very stable on compressed material and comparatively resource-friendly, ideal for longer videos.",
        },
        "supports_dn": False,
        "weight": "realesr-animevideov3.pth",
        "ncnn_name": "realesr-animevideov3",
        "ncnn_files": ["realesr-animevideov3.param", "realesr-animevideov3.bin"],
        "available_backends": ["pytorch", "ncnn"],
        "preferred_backends": ["pytorch", "ncnn"],
        "supports_tta": True,
        "tta_backends": ["ncnn"],
        "caps": {
            "denoise": {"type": "none"},
            "face_enhance": {"native": False, "via": ["gfpgan", "codeformer"]},
            "scale": {
                "pytorch": {
                    "mode": "discrete",
                    "values": [2, 3, 4],
                    "outscale": "optional-float",
                },
                "ncnn": {"mode": "discrete", "values": [2, 3, 4], "tta": True},
            },
            "noise_levels": None,
        },
    },
    "realcugan-se": {
        "desc": {
            "de": "2×/3×/4× Anime/Line-Art (SE) – konservatives, stabiles RealCUGAN-Modell mit wählbaren Rauschprofilen; gut für allgemeine Anime-Upscales bei mittlerem Ressourcenbedarf.",
            "en": "2×/3×/4× anime/line art (SE) – conservative, stable RealCUGAN model with selectable noise profiles; good for general anime upscales with medium resource usage.",
        },
        "backend": "ncnn",
        "available_backends": ["ncnn"],
        "preferred_backends": ["ncnn"],
        "supports_dn": False,
        "supports_tta": True,
        "tta_backends": ["ncnn"],
        "ncnn_name": "realcugan-ncnn-vulkan",
        "ncnn_model_dir": "models-se",
        "caps": {
            "denoise": {"type": "levels", "values": [-1, 0, 1, 2, 3]},
            "face_enhance": {"native": False},
            # WICHTIG: SE hat 2x/3x/4x
            "scale": {
                "pytorch": None,
                "ncnn": {"mode": "discrete", "values": [2, 3, 4], "tta": True},
            },
            # Union (wie bisher) + präzise je Scale:
            "noise_levels": [-1, 0, 1, 2, 3],
            "noise_levels_by_scale": {
                "2": [-1, 0, 1, 2, 3],
                "3": [-1, 0, 3],
                "4": [-1, 0, 3],
            },
        },
    },
    "realcugan-pro": {
        "desc": {
            "de": "2×/3× Anime (PRO) – aggressiveres RealCUGAN-Modell mit starker Detail- und Kantenbetonung für hochwertige Anime-Quellen; mittlerer bis hoher Ressourcenbedarf, kann Artefakte schwacher Quellen verstärken.",
            "en": "2×/3× anime (PRO) – more aggressive RealCUGAN model with strong detail and edge enhancement for high-quality anime sources; medium to high resource usage, can amplify artifacts in weak sources.",
        },
        "backend": "ncnn",
        "available_backends": ["ncnn"],
        "preferred_backends": ["ncnn"],
        "supports_dn": False,
        "supports_tta": True,
        "tta_backends": ["ncnn"],
        "ncnn_name": "realcugan-ncnn-vulkan",
        "ncnn_model_dir": "models-pro",
        "caps": {
            "denoise": {"type": "levels", "values": [-1, 0, 3]},
            "face_enhance": {"native": False},
            # WICHTIG: PRO hat nur 2x/3x – KEIN 4x
            "scale": {
                "pytorch": None,
                "ncnn": {"mode": "discrete", "values": [2, 3], "tta": True},
            },
            "noise_levels": [-1, 0, 3],
            "noise_levels_by_scale": {
                "2": [-1, 0, 3],
                "3": [-1, 0, 3],
            },
        },
    },
    "realcugan-nose": {
        "desc": {
            "de": "2× Anime (NOSE) – alternative RealCUGAN-Variante mit feinen Kanten für leichtes Upscaling; nur 2×-Scale und damit meist etwas ressourcenschonender als 3×/4×-Modelle.",
            "en": "2× anime (NOSE) – alternative RealCUGAN variant with fine edges for light upscaling; supports only 2× scale and is therefore usually somewhat more resource-friendly than 3×/4× models.",
        },
        "backend": "ncnn",
        "available_backends": ["ncnn"],
        "preferred_backends": ["ncnn"],
        "supports_dn": False,
        "supports_tta": True,
        "tta_backends": ["ncnn"],
        "ncnn_name": "realcugan-ncnn-vulkan",
        "ncnn_model_dir": "models-nose",
        "caps": {
            "denoise": {"type": "levels", "values": [0]},
            "face_enhance": {"native": False},
            # WICHTIG: NOSE hat nur 2x
            "scale": {
                "pytorch": None,
                "ncnn": {"mode": "discrete", "values": [2], "tta": True},
            },
            "noise_levels": [0],
            "noise_levels_by_scale": {"2": [0]},
        },
    },
}

# ---------------------------------------------
# Fallback-Download-Quellen (Regex → URL) – deckt übliche Real-ESRGAN .pth ab
#  (RealCUGAN benötigt keine .pth, kommt als NCNN-Paket mit param/bin)
# ---------------------------------------------
MODEL_FALLBACK_URLS: List[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^RealESRGAN_x4plus\.pth$"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    (
        re.compile(r"^RealESRGAN_x2plus\.pth$"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    (
        re.compile(r"^RealESRGAN_x4plus_anime_6B\.pth$"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    (
        re.compile(r"^realesr-general-x4v3\.pth$"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
    # Für PyTorch-Fälle von animevideov3 existiert ein .pth in Releases, NCNN nutzt param/bin
    (
        re.compile(r"^realesr-animevideov3\.pth$"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    ),
]
