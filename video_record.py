#!/usr/bin/env python3
import subprocess, threading, signal, time, re
from pathlib import Path
from datetime import datetime
import userInteraction as ui


"""
Record function v3 – robust busy‑device detection
=================================================
* Tests **both** selected video **and** audio devices via short FFmpeg probes before starting.
* Falls back automatisch auf freie Alternativ‑Knoten derselben Karte (z. B. /dev/video3, plughw:2,0 → pulse‑Quelle) oder bietet erneute Auswahl.
* ENTER‑Stop: SIGINT → SIGTERM after timeout, buffered stderr.
"""

# --------------------------------------------------
# Device discovery helpers
# --------------------------------------------------

def _v4l2_devices():
    res = subprocess.run(["v4l2-ctl", "--list-devices"], text=True, capture_output=True)
    devices, label = [], None
    for ln in res.stdout.splitlines():
        if ln and not ln.startswith("\t"):
            label = ln.strip()
        elif label and "/dev/video" in ln:
            node = ln.strip(); devices.append((node, f"{label} ({node})"))
    return devices


def _alsa_devices():
    res = subprocess.run(["arecord", "-l"], text=True, capture_output=True)
    pat = r"Karte (\d+): ([^ ]+) .*?Gerät (\d+): (.*?)\["
    return [(f"plughw:{c},{d}", f"ALSA {cn.strip()} – {dn.strip()} (plughw:{c},{d})")
            for c, cn, d, dn in re.findall(pat, res.stdout)]


def _pulse_sources():
    try:
        res = subprocess.run(["pactl", "list", "short", "sources"], text=True, capture_output=True)
    except FileNotFoundError:
        return []
    out = []
    for ln in res.stdout.splitlines():
        cols = ln.split("\t")
        if len(cols) >= 2 and "monitor" not in cols[1]:
            name, desc = cols[1], cols[-1]
            out.append((name, f"Pulse {desc} ({name})"))
    return out

# --------------------------------------------------
# Quick FFmpeg probes
# --------------------------------------------------

def _probe_ffmpeg(fmt, device):
    cmd = ["ffmpeg", "-v", "error", "-f", fmt, "-i", device, "-t", "0.1", "-c", "copy", "-f", "null", "-"]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

# --------------------------------------------------
# Minimal UI fallbacks
# --------------------------------------------------
try:
    select_from_list = ui.select_from_list
except AttributeError:
    def select_from_list(prompt, opts, default=None):
        print(f"\n{prompt}")
        for i, (_, lbl) in enumerate(opts, 1):
            print(f"  {i}) {lbl}")
        while True:
            ch = input(f"Select [1-{len(opts)}] or ENTER for default ({default}): ").strip()
            if not ch and default:
                return default
            if ch.isdigit() and 1 <= int(ch) <= len(opts):
                return opts[int(ch)-1][0]
            print("Invalid selection – try again.")

try:
    get_input = ui.get_input
except AttributeError:
    def get_input(prompt, default=None):
        inp = input(f"{prompt} [ENTER = {default}]: ") if default else input(f"{prompt}: ")
        return inp.strip() or default

