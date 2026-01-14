# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by common open-source practice (semantic versioning with
human-readable summaries).

## [0.2.1-alpha] – 2025-12-26

### Changed

- **aienhance:** Updated the worker selection mechanism to improve how the number
  of workers is determined and applied (more consistent scaling across different
  inputs and environments).

### Fixed

- **aienhance (emulated TTA):** Fixed a bug affecting longer videos (around
  ~1 minute and above) where incorrect memory management could lead to an
  early/incorrect abort.
- **Non-AI methods:** Fixed an intermittent issue in the FFmpeg wrapper where the
  final output name could sometimes still display a stale (not yet updated)
  filename in certain rename/overwrite flows.

---

## [0.0.1-alpha] – 2024-11-22

### Added

- **Initial public alpha release of _PhotonFrame – VideoKit_.**
- Core CLI entry point:
  - `video` launcher script for all subcommands.
  - Shared interactive vs. CLI/batch behaviour across commands.

- **Conversion & compression:**
  - `convert`: container/codec conversion with presets
    (e.g. `messenger360p`, `web`, `cinema`, `studio`, `ultra`, `lossless`).
  - `compress`: filesize reduction via a simple visual quality percentage
    (`--quality` → CRF mapping), with stream/thumbnail/metadata preservation.

- **Editing & formatting tools:**
  - `trim`: fast (lossless/GOP) and precise (re-encode) cutting.
  - `scale`: resolution scaling with aspect-ratio aware presets and custom sizes.
  - `croppad`: axis-wise crop/pad to target resolutions, alpha-aware.
  - `merge`: concatenation of clips plus merging of audio/subtitle tracks,
    including offsets, pauses and simple target-resolution strategies.

- **Frame & sequence tools:**
  - `interpolate`: frame-rate upsampling/normalisation using ffmpeg’s
    `minterpolate`, with quality profiles.
  - `img2vid`: creation of videos from numbered image sequences or folders.
  - `extract`: extraction of audio, subtitles, frames, thumbnails and
    other streams (including percent/time-based frame positions).

- **Enhancement & AI:**
  - `enhance`: classic ffmpeg-based enhancement presets
    (stabilisation, denoise, color/levels tweaks).
  - `aienhance` / `ai-enhance`: AI upscaling/enhancement based on
    **Real-ESRGAN** and **RealCUGAN**, including:
    - model selection (`--aimodel`, e.g. `realesr-general-x4v3`, anime models, etc.)
    - scaling factors (`--scale`)
    - optional TTA and blending controls.

- **GIF & meme creation:**
  - `gif`: animated GIF creation from video/GIF sources with:
    - top/bottom meme text (`--text-top`, `--text-bottom`)
    - font-size presets (`thiny`, `small`, `medium`, `grande`, `large`, `huge`)
    - support for higher-quality GIF pipelines and optional auto-open behaviour.

- **Metadata subsystem:**
  - `metadata`: detailed metadata inspection and editing:
    - `--list-tags`, `--list-tags-json`, `--list-tagnames`
    - generic `--tag key` / `--tag key=value` interface
    - convenience flags for common tags (e.g. `--title`, `--artist`,
      `--production_year`, …)
    - thumbnail management (`--set-thumbnail`, `--delete-thumbnail`, `--show-thumbnail`)
    - internal `metadata_support` layer with schema of editable/protected tags.
  - Alpha/pixel-format aware reporting (color primaries, transfer, matrix,
    limited/full range, alpha presence).

- **Installer & environment setup:**
  - `install.sh`:
    - language prompt (EN/DE) and minimal i18n layer for installer messages.
    - OS/GPU detection and backend choice (Torch vs. NCNN, CUDA/MPS/CPU).
    - Robust, retry-heavy download ladder for Conda/PyTorch/NCNN/model assets.
    - Local Miniconda bootstrap with strict `conda-forge`/`pytorch` channels.
    - Environment setup for:
      - core Python stack (NumPy, Pillow, OpenCV headless, argcomplete, etc.)
      - ffmpeg (Conda/system fallback, vidstab support where possible)
      - ExifTool installation (Conda-first, then system package managers).
    - Automatic install of:
      - Real-ESRGAN (Python) + weights
      - BasicSR, facexlib, GFPGAN (+ optional CodeFormer)
      - NCNN binaries (Real-ESRGAN-NCNN, RealCUGAN-NCNN) and associated models.
    - Creation of activation hooks for:
      - face model paths (GFPGAN, CodeFormer)
      - default ESRGAN model path/name (if weights found).
    - Generator for `video` runner in `~/.local/bin` and argcomplete integration
      for Bash/Zsh.

- **Configuration & logging:**
  - Automatic creation/merge of `~/.config/videoManager/config.ini` with:
    - language, OS, GPU/AI backend settings
    - install/data/cache/state directories
    - default output & temp paths
    - logging path and log-level.
  - Project-local `config.ini` and helpers for environment-driven defaults.
  - Basic logging infrastructure (`loghandler.py`, `mem_guard.py`, etc.) for
    CLI operations and AI jobs.

- **Documentation & infofiles:**
  - `README.md` with:
    - project overview
    - command summary
    - installation and usage basics
    - licensing notes and third-party overview.
  - Per-command English infofiles under `infofiles/PhotonFrame.*.en.info`
    as single source of truth for flags and behaviour:
    - `PhotonFrame.en.info` (global overview)
    - `PhotonFrame.convert.en.info`
    - `PhotonFrame.compress.en.info`
    - `PhotonFrame.trim.en.info`
    - `PhotonFrame.scale.en.info`
    - `PhotonFrame.croppad.en.info`
    - `PhotonFrame.merge.en.info`
    - `PhotonFrame.interpolate.en.info`
    - `PhotonFrame.img2vid.en.info`
    - `PhotonFrame.extract.en.info`
    - `PhotonFrame.enhance.en.info`
    - `PhotonFrame.aienhance.en.info`
    - `PhotonFrame.gif.en.info`
    - `PhotonFrame.metadata.en.info`.

- **Licensing:**
  - Repository licensed under the **MIT License** (`LICENSE` added).
  - `THIRD_PARTY_LICENSES.md` generated by the installer with a summary of:
    - Real-ESRGAN, BasicSR, facexlib, GFPGAN, CodeFormer
    - Microsoft Core Fonts – Impact
    - and other external dependencies, each under its own upstream license.

### Known Limitations

- Alpha quality:
  - Command-line interfaces and presets may still change in upcoming versions.
  - Error handling and edge-case behaviour (exotic containers/codecs) are still
    being refined.
- Windows support:
  - The Bash-based installer does **not** support Windows directly; only Linux
    (and partially macOS) have been tested.

---

### [0.3.0-alpha] – 2026-01-14

### Added

   - metadata: Extended media probing output with additional HDR and bitrate details:

   - HDR type (e.g., HDR10 / HLG / Dolby Vision where detectable)

   - Video bitrate (reported as a dedicated field to make quality/compatibility checks easier)

   - convert: Added explicit audio configuration controls to better target playback compatibility and consistent library standards:

   - Audio codec selection (choose the desired output codec instead of relying on implicit defaults)

   - Audio channel layout / channel count selection (e.g., force stereo, preserve multichannel, etc., depending on your chosen workflow)

   - convert: Implemented HDR → SDR mapping to enable creating SDR-compatible outputs from HDR sources without manual filter construction, improving playback compatibility on non-HDR displays/clients.

   - convert: Introduced an explicit pixel format selector, allowing you to control output subsampling/format (e.g., yuv420p for broad device compatibility) instead of inheriting encoder defaults.

   - convert: Added an explicit video bitrate option to allow deterministic output targeting (useful for bandwidth-constrained devices, Jellyfin streaming profiles, and consistent storage planning).


---

[0.3.0-alpha]: https://github.com/<your-account>/PhotonFrame-VideoKit/releases/tag/v0.3.0-alpha
[0.2.1-alpha]: https://github.com/<your-account>/PhotonFrame-VideoKit/releases/tag/v0.2.1-alpha
[0.0.1-alpha]: https://github.com/<your-account>/PhotonFrame-VideoKit/releases/tag/v0.0.1-alpha
