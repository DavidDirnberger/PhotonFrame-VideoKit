
# -----------------------------------------------------------------------------
# 1) ffprobe-Hilfsfunktion ----------------------------------------------------
# -----------------------------------------------------------------------------
def probe_video(path: str) -> Tuple[float, int, float, str]:
    """
    returns: duration_sec, pixels_per_frame, fps, codec_name
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,width,height,avg_frame_rate,codec_name",
        "-of", "json", path
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    s = data["streams"][0]
    dur  = float(s["duration"])
    px   = int(s["width"]) * int(s["height"])
    num, den = map(int, s["avg_frame_rate"].split("/"))
    fps  = num / den if den else num
    codec = s["codec_name"].lower()
    return dur, px, fps, codec

# -----------------------------------------------------------
# 2) Baseline-Geschwindigkeiten ------------------------------
# -----------------------------------------------------------
# 1080p, H.264, CPU-Encode (oder dein NVENC-Profil etc.)
BASE_FPS = {
    "convert"    : 120,
    "interpolate":  30,
    "trim"       : 500,
    "compress"   : 100,
    "gif"        :  25,
    "extract"    : 400,
    "merge"      : 900,
    "crop_pad"   :  80,
}


def load_profile() -> Dict[str, float]:
    if CALIBRATION_FILE.exists():
        return {**BASE_FPS, **json.loads(CALIBRATION_FILE.read_text())}
    return BASE_FPS.copy()


def save_profile(p: Dict[str, float]):
    CALIBRATION_FILE.write_text(json.dumps(p, indent=2))


# Codec-/Format-Faktoren (1 = H.264 1080p)
CODEC_FACTOR = {
    "h264"   : 1.0,
    "hevc"   : 1.4,
    "av1"    : 2.0,
    "vp9"    : 1.7,
    "mpeg4"  : 0.8,
    "prores" : 0.8,
    "gif"    : 2.0,
    # unbekannt → 1.2 – eher pessimistisch
}

REF_PX = 1920 * 1080     # 1080p

def _codec_factor(c: str) -> float:
    return CODEC_FACTOR.get(c, 1.2)


# -----------------------------------------------------------
# 3) Hauptfunktion  – liefert jetzt nur noch total_sec -------
# -----------------------------------------------------------
def estimate_ffmpeg_duration_old(operation: str,
                             files: list[str],
                             methodname: str,
                             speed_profile: dict[str, float] | None = None
) -> float:
    """
    Rückgabe: Gesamt-ETA in Sekunden (float)
    """
    if speed_profile is None:
        speed_profile = load_profile()


    base_fps = speed_profile.get(operation, BASE_FPS[methodname])

    total = 0.0
    for f in files:
        dur, px, _, codec = probe_video(f)
        cf          = _codec_factor(codec)
        res_factor  = (px / REF_PX) ** 0.5
        adj_fps     = base_fps / (cf * res_factor)
        total      += dur / adj_fps

    return total        # <-- nur noch eine Zahl



def estimate_ffmpeg_duration(
    ffmpeg_cmd: list[str],
    files: list[str],
    base_operation: str,
    speed_profile: dict[str, float] | None = None
) -> float:
    """
    Schätzt die Gesamtprozessdauer basierend auf ffmpeg-Parametern, Videoeigenschaften und Erfahrungswerten.
    """

    if speed_profile is None:
        speed_profile = load_profile()

    cmd_str = " ".join(ffmpeg_cmd).lower()

    # Preset-Faktor -----------------------------------------------------------
    preset_factor = 1.0
    for preset in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]:
        if preset in cmd_str:
            preset_factor = {
                "ultrafast": 0.5,
                "superfast": 0.6,
                "veryfast": 0.7,
                "faster": 0.8,
                "fast": 0.9,
                "medium": 1.0,
                "slow": 1.2,
                "slower": 1.5,
                "veryslow": 2.0
            }[preset]
            break

    # CRF-Faktor -------------------------------------------------------------
    crf_factor = 1.0
    match = re.search(r"-crf\s+(\d+)", cmd_str)
    if match:
        crf = int(match.group(1))
        crf_factor = max(0.8, min(1.2, (23 / crf)))

    # Ziel-Auflösung ----------------------------------------------------------
    scale_factor = 1.0
    match = re.search(r"scale=([0-9]+):([0-9\-]+)", cmd_str)
    if match:
        w, h = match.groups()
        try:
            w = int(w)
            h = int(h) if h != "-2" else int(w) * 9 // 16
            px = w * h
            scale_factor = (px / REF_PX) ** 0.5
        except:
            pass

    # Interpolation -----------------------------------------------------------
    interpolation_factor = 2.0 if 'minterpolate' in cmd_str else 1.0

    # Formatwechsel -----------------------------------------------------------
    try:
        infile = Path(ffmpeg_cmd[ffmpeg_cmd.index('-i') + 1])
        outfile = Path(ffmpeg_cmd[-1])
        quellformat = infile.suffix.lower()
        zielformat = outfile.suffix.lower()
        format_factor = 1.2 if quellformat != zielformat else 1.0
    except Exception:
        format_factor = 1.0

    # Codec-Wechsel -----------------------------------------------------------
    reencode_factor = 1.5 if '-c:v' in ffmpeg_cmd else 1.0

    base_fps = speed_profile.get(base_operation, BASE_FPS[base_operation])

    total = 0.0
    for f in files:
        dur, px, _, codec = probe_video(f)
        cf = _codec_factor(codec)
        size_factor = (px / REF_PX) ** 0.5

        adj_fps = base_fps / (
            cf * size_factor * scale_factor * crf_factor * preset_factor *
            format_factor * reencode_factor * interpolation_factor
        )
        total += dur / adj_fps

    return total



# -----------------------------------------------------------
# 4) Kalibrierung -------------------------------------------
# -----------------------------------------------------------

def update_profile(operation: str,
                   estimated_sec: float,
                   real_sec: float,
                   alpha: float = 0.25,
                   meta: dict | None = None) -> None:
    """
    Passt Baseline-FPS an und loggt den Run in runs.csv.
    meta = optionale Zusatzinfos (z.B. {'file': 'in.mkv', 'pixels': 2073600, 'codec': 'h264'})
    """
    if real_sec <= 0:
        return

    # 1) Profil adaptieren ----------------------------------------------------
    prof      = load_profile()
    base_old  = prof.get(operation, BASE_FPS[operation])
    factor    = estimated_sec / real_sec
    base_new  = (1 - alpha) * base_old * factor + alpha * base_old
    prof[operation] = round(base_new, 2)
    save_profile(prof)

    # 2) Lauf protokollieren --------------------------------------------------
    RUN_LOG.touch(exist_ok=True)
    with RUN_LOG.open("a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:                           # Kopfzeile nur einmal
            writer.writerow(["ts", "operation", "est_sec", "real_sec", "meta"])
        writer.writerow([time.time(), operation, round(estimated_sec, 2),
                         round(real_sec, 2), json.dumps(meta or {}, ensure_ascii=False)])
        

# ---------------------------------------------------------------------------
# 1) Create a tiny 5-second 1080p test clip (if it doesn't exist) ------------
# ---------------------------------------------------------------------------
def _make_testclip(path: Path, dur: int = 5):
    """
    Generate a 5-second colour-bars clip (1920×1080, 30 fps, H.264).
    """
    if path.exists():
        return
    cmd = (
        f"ffmpeg -y -f lavfi -i testsrc=size=1920x1080:rate=30 "
        f"-t {dur} -pix_fmt yuv420p -c:v libx264 -crf 17 {path}"
    )
    subprocess.run(shlex.split(cmd), check=True)

# ---------------------------------------------------------------------------
# 2) One minimal FFmpeg command per operation --------------------------------
# ---------------------------------------------------------------------------
def build_dummy_cmd(op: str, infile: Path, outdir: Path) -> str:
    """
    Return a short FFmpeg command that exercises the given operation.
    """
    out = outdir / f"{op}.mp4"
    match op:
        case "convert":
            return f"ffmpeg -y -i {infile} -c:v libx265 {out}"
        case "interpolate":
            return f"ffmpeg -y -i {infile} -vf minterpolate=fps=60 {out}"
        case "trim":
            return f"ffmpeg -y -ss 1 -t 2 -i {infile} -c copy {out}"
        case "compress":
            return f"ffmpeg -y -i {infile} -c:v libx264 -crf 28 {out}"
        case "gif":
            out = outdir / "gif.gif"
            return f"ffmpeg -y -i {infile} -vf fps=15,scale=480:-1 -gifflags -offsetting {out}"
        case "extract":
            out = outdir / "frame_%03d.png"
            return f"ffmpeg -y -i {infile} -vf fps=10 {out}"
        case "merge":
            # simple left/right split-screen merge
            return (
                f"ffmpeg -y -i {infile} -i {infile} "
                f"-filter_complex "
                f"[0:v]crop=iw/2:ih:0:0[left]; "
                f"[1:v]crop=iw/2:ih:iw/2:0[right]; "
                f"[left][right]hstack\" -c:v libx264 {out}"
            )
        case "crop_pad":
            return f"ffmpeg -y -i {infile} -vf crop=1280:720:320:180,pad=1920:1080:0:0:black {out}"
    raise ValueError(f"Unknown operation: {op}")

# ---------------------------------------------------------------------------
# 3) Warm-up driver ----------------------------------------------------------
# ---------------------------------------------------------------------------
def warmup_calibration():
    operations = [
        "convert", "interpolate", "trim", "compress",
        "gif", "extract", "merge", "crop_pad",
    ]
    tmp_dir = defin.TMP_DIR
    src_clip = tmp_dir / "testsrc.mp4"
    _make_testclip(src_clip)

    print(f"[Warm-Up] Test clip created at: {src_clip}")
    print(f"[Warm-Up] Temporary output directory: {tmp_dir}\n")

    for op in operations:
        command = build_dummy_cmd(op, src_clip, tmp_dir)
        files   = [src_clip]

        est_sec = estimate_ffmpeg_duration(op, files)

        t0 = time.time()
        subprocess.run(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        real_sec = time.time() - t0

        update_profile(
            op,
            est_sec,
            real_sec,
            meta={"dummy": True, "clip": str(src_clip)},
        )

        print(f"{op:<12}  ETA {est_sec:>5.2f}s   real {real_sec:>5.2f}s")

    print("\n[Warm-Up] Done. Baseline profile updated – future estimates will be tighter.")
