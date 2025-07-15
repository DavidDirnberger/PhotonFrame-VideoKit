#!/usr/bin/env python3
from datetime import datetime, timedelta
from pathlib import Path
import csv, time
import re
import userInteraction as ui
import json, os, subprocess, shlex, statistics, time
from typing import List, Dict, Tuple
import definitions as defin


SCRIPT_DIR   = Path(__file__).resolve().parent
EVAL_DIR     = SCRIPT_DIR / "duration_estimation"
EVAL_DIR.mkdir(exist_ok=True)                         # legt Ordner bei Bedarf an

CALIBRATION_FILE = EVAL_DIR / "ffmpeg_speed_profile.json"
RUN_LOG          = EVAL_DIR / "runs.csv"              # optionales CSV-Log


def total_sec(paths: list[Path], pause_len: float = 0.0, pause_count: int = 0) -> float:
    """Sum of durations for every file in paths  (+ pause_len * pause_count)."""
    total = 0.0
    for p in paths:
        total += ui.get_duration(p) or 0
    total += pause_len * pause_count
    return total


def s2sec(t):
    if not t: return 0
    p = [float(x) for x in t.split(':')]
    return p[0] if len(p) == 1 else p[0]*60 + p[1] if len(p) == 2 else p[0]*3600 + p[1]*60 + p[2]


def parse_time(t: str) -> timedelta:
    """Parst eine Zeitangabe wie HH:MM:SS, MM:SS oder SS zu timedelta."""
    parts = t.strip().split(":")
    parts = [int(p) for p in parts if p.isdigit()]
    if len(parts) == 3:
        return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
    elif len(parts) == 2:
        return timedelta(minutes=parts[0], seconds=parts[1])
    elif len(parts) == 1:
        return timedelta(seconds=parts[0])
    else:
        raise ValueError("Ungültiges Zeitformat")

def seconds_to_time(t:timedelta) -> str:
    return f"{int(t.total_seconds())//3600:02}:{(int(t.total_seconds())%3600)//60:02}:{int(t.total_seconds())%60:02}"


def add_timecodes(start: str, duration: str) -> str:
    """
    Add two timecodes in HH:MM:SS, MM:SS, or SS format and return result as HH:MM:SS.
    """
    try:
        t1 = parse_time(start)
        t2 = parse_time(duration)
        total = t1 + t2
        hours, remainder = divmod(int(total.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    except Exception as e:
        ui.print_error(f"❌ Timecalculation failed: {e}")
        return "00:00:00"
    