#!/usr/bin/env python3
import re
import os
from pathlib import Path

ANSI_RESET = "\033[0m"
CLEAR_LINE = "\033[2K"           # clear entire line
ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*m")  # strip colours for length calc
TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = Path(SCRIPT_DIR) / 'tmp'
SCREENCAST_SCRIPT = os.path.join(SCRIPT_DIR, "screenrecord", "screenrecord.sh")
SCREENCAST_INFO = os.path.join(SCRIPT_DIR, "screenrecord", "screenrecord.info")

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"]
AUDIO_EXTENSIONS = {'.mp3', '.aac', '.wav', '.flac', '.ogg'}
SUB_EXTENSIONS = {'.srt', '.ass', '.vtt'}

FORMATS = ["mp4", "mkv", "avi", "mov", "webm", "flv", "mpeg"]
FORMAT_DESCRIPTIONS = {
    "mp4": "🎥 MP4  – widley supported, ideal for web and mobile (uses H.264/H.265).",
    "mkv": "🎞  MKV  – supporting multiple audio and subtitle tracks, great for archiving.",
    "avi": "📼 AVI  – Older format with larger file sizes and less efficient compression.",
    "mov": "🍏 MOV  – Apple QuickTime format, high quality, best for macOS/iOS.",
    "webm": "🌐 WEBM – Open format optimized for web use (uses VP8/VP9 codecs).",
    "flv": "📺 FLV  – Flash video format, outdated but still used in legacy content.",
    "mpeg": "💿 MPEG – Early standard (MPEG-1/2), used in older systems and DVDs."
}
PRESETS = {
    "messenger": "-preset ultrafast -crf 32 -vf scale=640:-2",
    "web": "-preset veryfast -crf 28",
    "casual": "-preset medium -crf 24",
    "cinema": "-preset slow -crf 20"
}
PRESET_DESCRIPTIONS = {
    "messenger": "📱 Messenger – Super fast, low quality, small file (for chat apps)",
    "web":       "🌐 Web – Fast encoding, reduced size, decent quality (for online use)",
    "casual":    "🎞  Casual – Balanced quality and speed (for everyday viewing)",
    "cinema":    "🎬 Cinema – High quality, slower encoding (for film-like output)"
}
RESOLUTIONS = ["8K-DCI", "8K", "4K-DCI", "4K", "QHD+", "1440p", "1080p", "720p", "480p", "360p", "240p","original"]
RESOLUTION_DESCRIPTIONS = {
    "240p": "240p => 426:240 (16:9)",
    "360p": "360p => 640:360 (16:9)",
    "480p": "480p => 854:480 (16:9)",
    "720p": "720p => 1280:720 (16:9)",
    "1080p": "1080p (HD) => 1920:1080 (16:9)",
    "1440p": "1440p (QHD) => 2560:1440 (16:9)",
    "QHD+": "QHD+ => 3200:1800 (~16:9) [Laptop Displays]",
    "4K": "4K-UHD => 3840:2160 (16:9) 2160p Ultra High Definition",
    "4K-DCI": "4K-DCI => 4096:2160 (17:9) Digital Cinema Initiatives",
    "8K": "8K-UHD => 7680:4320 (16:9) 4320p Full Ultra High Definition",
    "8K-DCI": "8K-DCI => 8192:4320 (17:9) Digital Cinema Initiatives"
}
RESOLUTION_SCALES = {
    "240p": "426:240",
    "360p": "640:360",
    "480p": "854:480",
    "720p": "1280:720",
    "1080p": "1920:1080",
    "1440p": "2560:1440",
    "QHD+": "3200:1800",
    "4K": "3840:2160",
    "4K-DCI": "4096:2160",
    "8K": "7680:4320",
    "8K-DCI": "8192:4320"
}

FRAMERATES = ["original", "15", "23.976", "24", "25", "29.97", "30", "50", "59.94", "60", "120"]
FRAMERATE_DESCRIPTIONS = {
    "15": "15 fps - Low quality / surveillance",
    "23.976": "23.976 fps - NTSC film",
    "24": "24 fps - Cinema standard",
    "25": "25 fps - PAL (Europe)",
    "29.97": "29.97 fps - NTSC (TV)",
    "30": "30 fps - Web & mobile",
    "50": "50 fps - PAL HD",
    "59.94": "59.94 fps - NTSC HD",
    "60": "60 fps - YouTube / Gaming",
    "120": "120 fps - High FPS / Slow motion",
    "original": "Use original framerate"
}

CODECS = ["libx264", "libx265", "vp9", "mpeg4", "copy"]
CODEC_DESCRIPTIONS = {
    "libx264": "H.264 (widely supported)",
    "libx265": "H.265 (smaller size, higher CPU)",
    "vp9": "VP9 (open, good quality)",
    "mpeg4": "MPEG-4 (older devices)",
    "copy": "Copy original video codec"
}

PROTECTED_KEYS = {"major_brand", "minor_version", "compatible_brands", "encoder", "duration"}
TAG_DESCRIPTIONS = {
            "title": "Title of the file",
            "show": "TV show or series name",
            "season_number": "Season number",
            "episode_id": "Episode number",
            "director": "Director of the content",
            "production_year": "Year of production",
            "network": "Broadcasting network",            
            "artist": "Creator or performer",
            "album": "Associated album",
            "date": "Release date",
            "track": "Track number in album",
            "composer": "Composer of the content",
            "publisher": "Publisher or distributor",
            "genre": "Genre such as Drama or Sci-Fi",
            "language": "Language of the content",            
            "comment": "User comment",
            "keywords": "Search keywords",
}
