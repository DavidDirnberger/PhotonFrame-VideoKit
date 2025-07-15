from pathlib import Path

import userInteraction as ui                   # project‑local helper (select files, progress bar …)
import definitions as defin                # project‑wide constants (e.g. VIDEO_EXTENSIONS)



# =============================================================================
# Low‑level FFmpeg command builder
# =============================================================================


def _build_filter_cmd(
    *,
    input_file: Path,
    output_file: Path,
    crop_noise: int = 16,
    deint_mode: str = "send_field",
    # hqdn3d params
    denoise_spatial_luma: int = 4,
    denoise_spatial_chroma: int = 3,
    denoise_temporal_luma: int = 6,
    denoise_temporal_chroma: int = 6,
    # chroma shift
    cbh: int = -2,
    crh: int = 2,
    # unsharp mask
    unsharp_luma_msize_x: int = 5,
    unsharp_luma_msize_y: int = 5,
    unsharp_luma_amount: float = 0.8,
    unsharp_chroma_msize_x: int = 3,
    unsharp_chroma_msize_y: int = 3,
    unsharp_chroma_amount: float = 0.4,
    # encoder
    crf: float = 21.0,
    preset: str = "medium",
    threads: int = 0,
    extra_ffmpeg_args: list[str] | None = None,
) -> list[str]:
    """Assemble the ffmpeg command list with all filters."""

    vf_chain = [
        f"crop=in_w:in_h-{crop_noise}:0:0",
        f"bwdif=mode={deint_mode}",
        f"hqdn3d={denoise_spatial_luma}:{denoise_spatial_chroma}:{denoise_temporal_luma}:{denoise_temporal_chroma}",
        f"chromashift=cbh={cbh}:crh={crh}",
        (
            f"unsharp={unsharp_luma_msize_x}:{unsharp_luma_msize_y}:{unsharp_luma_amount}:"
            f"{unsharp_chroma_msize_x}:{unsharp_chroma_msize_y}:{unsharp_chroma_amount}"
        ),
    ]

    cmd: list[str | Path] = [
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", input_file,
        "-vf", ",".join(vf_chain),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-threads", str(threads),
        "-pix_fmt", "yuv420p",
        "-x264-params", "colormatrix=bt709:range=tv",
        "-c:a", "copy",
    ]

    if extra_ffmpeg_args:
        cmd.extend(extra_ffmpeg_args)


    cmd.append(output_file)
    return [str(c) for c in cmd if c is not None]