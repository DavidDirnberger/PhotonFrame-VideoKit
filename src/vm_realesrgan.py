#!/usr/bin/env python3
# vm_realesrgan_api_runner.py (v3.1)
# Drop-in kompatibel zum Original-CLI, robustes imread + Auto-Tile-Fallback, kein Drittcode-Patch nötig.

import argparse
import glob
import inspect
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, MutableMapping, Sequence, Union, cast

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from numpy.typing import NDArray
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from ai_weights import resolve_gfpgan_weight

ArrayU8 = NDArray[np.uint8]
ArrayU16 = NDArray[np.uint16]
ImageArray = Union[ArrayU8, ArrayU16]


def guard_versions_and_caps():
    def get_ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return "unknown"

    re_v = get_ver("realesrgan")
    bs_v = get_ver("basicsr")
    tc_v = get_ver("torch")

    # Prüfe optionale Parameter des RealESRGANer-Konstruktors
    from realesrgan import RealESRGANer

    sig = inspect.signature(RealESRGANer.__init__)
    has_dni = "dni_weight" in sig.parameters
    has_dn_strength = "denoise_strength" in sig.parameters

    print(
        f"[caps] realesrgan={re_v} basicsr={bs_v} torch={tc_v} "
        f"dni_weight={has_dni} denoise_strength={has_dn_strength}"
    )

    # Beispiel: harte Schranke, falls du willst
    # if re_v not in ("0.3.0", "0.3.1"): print("[warn] ungetestete realesrgan-Version:", re_v, file=sys.stderr)


def robust_imread(path: str, unchanged: bool = False) -> ImageArray | None:
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    try:
        img = cv2.imread(os.fspath(path), flag)
        if isinstance(img, np.ndarray):
            # dtype sicherstellen (Pylance: MatLike → ImageArray)
            if img.dtype != np.uint8 and img.dtype != np.uint16:
                img = img.astype(np.uint8, copy=False)
            img = np.ascontiguousarray(img)
            return cast(ImageArray, img)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        dec = cv2.imdecode(arr, flag)
        if isinstance(dec, np.ndarray):
            if dec.dtype != np.uint8 and dec.dtype != np.uint16:
                dec = dec.astype(np.uint8, copy=False)
            dec = np.ascontiguousarray(dec)
            return cast(ImageArray, dec)
        return None
    except Exception:
        return None


def write_image(path: str, arr: ImageArray) -> None:
    out = np.ascontiguousarray(arr)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8, copy=False)
    ok = cv2.imwrite(path, out)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def pick_model(model_name: str):
    mn = model_name.split(".")[0]
    if mn == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif mn == "RealESRNet_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif mn == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif mn == "RealESRGAN_x2plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif mn == "realesr-animevideov3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ]
    elif mn == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]
    else:
        raise ValueError(f"Unknown model_name: {mn}")
    return mn, model, netscale, urls


def resolve_model_path(
    model_name: str, cli_model_path: str | None, urls: list[str]
) -> str | list[str]:
    if cli_model_path:
        return cli_model_path
    model_path = os.path.join("weights", model_name + ".pth")
    if os.path.isfile(model_path):
        return model_path
    ROOT_DIR = os.environ.get("VM_BASE") or str(Path(__file__).resolve().parents[1])
    for url in urls:
        model_path = load_file_from_url(
            url=url,
            model_dir=os.path.join(ROOT_DIR, "weights"),
            progress=True,
            file_name=None,
        )
    return model_path  # string (wie Original nach letzter URL)


def build_upsampler(
    netscale: int,
    model,
    model_path: str | list[str],
    tile: int,
    tile_pad: int,
    pre_pad: int,
    use_fp32: bool,
    gpu_id: int | None,
    denoise_strength: float | None,
    dni_weight: Sequence[float] | None,
):
    half: bool = bool((not use_fp32) and torch.cuda.is_available())

    # Wichtig: Dict bewusst "Any", damit Pylance nicht einschränkt
    kwargs: MutableMapping[str, Any] = {
        "scale": int(netscale),
        "model_path": model_path,
        "model": model,
        "tile": int(tile),
        "tile_pad": int(tile_pad),
        "pre_pad": int(pre_pad),
        "half": half,
        "gpu_id": (
            gpu_id if (gpu_id is None or isinstance(gpu_id, int)) else int(gpu_id)
        ),
    }

    if dni_weight is not None:
        kwargs["dni_weight"] = [float(x) for x in dni_weight]

    if (
        denoise_strength is not None
        and "denoise_strength" in inspect.signature(RealESRGANer.__init__).parameters
    ):
        kwargs["denoise_strength"] = float(denoise_strength)

    return RealESRGANer(**kwargs)


def auto_tile_try_list(initial_tile: int) -> list[int]:
    if initial_tile and initial_tile > 0:
        return [initial_tile, 640, 512, 320, 256, 128, 64]
    return [0, 640, 512, 320, 256, 128, 64]


def try_enhance(
    upsampler, img: ImageArray, outscale: float, out_path: str, tiles: list[int]
) -> tuple[bool, int, str | None]:
    last_err: str | None = None
    for t in tiles:
        try:
            if hasattr(upsampler, "tile"):
                setattr(upsampler, "tile", int(t))
            with torch.inference_mode():
                output, _ = upsampler.enhance(img, outscale=outscale)
            # typing-happy write
            write_image(out_path, output)
            return True, t, None
        except RuntimeError as e:
            msg = str(e)
            oom = (
                "CUDA out of memory" in msg
                or "cublasStatusAllocFailed" in msg
                or "CUDA error" in msg
            )
            last_err = msg
            if not oom:
                return False, t, msg
        except Exception as e:
            return False, t, str(e)
    return False, (tiles[-1] if tiles else 0), last_err


def main():
    guard_versions_and_caps()
    parser = argparse.ArgumentParser(
        description="Robuster Real-ESRGAN Runner (drop-in kompatibel)."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="inputs", help="Input image or folder"
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="RealESRGAN_x4plus",
        help=(
            "Model names: RealESRGAN_x4plus | RealESRNet_x4plus | "
            "RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | "
            "realesr-animevideov3 | realesr-general-x4v3"
        ),
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="Output folder"
    )
    parser.add_argument(
        "-dn",
        "--denoise_strength",
        type=float,
        default=0.5,
        help="Only used for realesr-general-x4v3 (DNI mix).",
    )
    parser.add_argument(
        "-s", "--outscale", type=float, default=4, help="Final upsampling scale"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Optional explicit model path"
    )
    parser.add_argument(
        "--suffix", type=str, default="out", help="Suffix of the restored image"
    )
    parser.add_argument(
        "-t", "--tile", type=int, default=0, help="Tile size, 0 for no tiling"
    )
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
    parser.add_argument("--pre_pad", type=int, default=0, help="Pre padding at borders")
    parser.add_argument(
        "--face_enhance",
        action="store_true",
        help="Use GFPGAN to enhance faces (if available)",
    )
    parser.add_argument(
        "--fp32", action="store_true", help="Force FP32 (default: half on CUDA)"
    )
    parser.add_argument(
        "--alpha_upsampler",
        type=str,
        default="realesrgan",
        help="(kept for compatibility)",
    )
    parser.add_argument("--ext", type=str, default="auto", help="auto | jpg | png")
    parser.add_argument(
        "-g", "--gpu-id", type=int, default=None, help="gpu device index (e.g. 0)"
    )

    args = parser.parse_args()

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    mn, model, netscale, urls = pick_model(args.model_name)
    model_path = resolve_model_path(mn, args.model_path, urls)

    # DNI-Mix nur für realesr-general-x4v3, und nur wenn beide Files vorhanden
    dni_weight = None
    if mn == "realesr-general-x4v3" and args.denoise_strength != 1:
        if isinstance(model_path, str):
            # versuche WDN neben explizitem Pfad
            wdn_model_path = model_path.replace(
                "realesr-general-x4v3", "realesr-general-wdn-x4v3"
            )
            if os.path.isfile(wdn_model_path):
                model_path = [model_path, wdn_model_path]
                dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
            else:
                print(
                    f"[warn] WDN weights not found: {wdn_model_path} → continue without DNI",
                    file=sys.stderr,
                )
        else:
            paths = list(model_path)
            x = [p for p in paths if "realesr-general-x4v3" in p]
            w = [p for p in paths if "realesr-general-wdn-x4v3" in p]
            if x and w and os.path.isfile(x[-1]) and os.path.isfile(w[-1]):
                model_path = [x[-1], w[-1]]
                dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
            else:
                print(
                    "[warn] Could not resolve both x4v3 + wdn-x4v3 weights → continue without DNI",
                    file=sys.stderr,
                )

    upsampler = build_upsampler(
        netscale=netscale,
        model=model,
        model_path=model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        use_fp32=args.fp32,
        gpu_id=args.gpu_id,
        denoise_strength=(
            args.denoise_strength if mn == "realesr-general-x4v3" else None
        ),
        dni_weight=dni_weight,
    )

    face_enhancer = None
    if args.face_enhance:
        try:
            from gfpgan import GFPGANer

            vm_base = (
                Path(os.environ.get("VM_BASE", "")).resolve()
                if os.environ.get("VM_BASE")
                else None
            )
            local_face = resolve_gfpgan_weight(vm_base)

            # letzter Fallback nur wenn GAR NICHTS lokal gefunden wurde
            fallback_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
            model_path = str(local_face) if local_face else fallback_url

            face_enhancer = GFPGANer(
                model_path=model_path,
                upscale=args.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
            if local_face:
                print(f"[gfpgan] using local model: {local_face}")
            else:
                print("[gfpgan] no local model found → using fallback URL (once)")
        except Exception as e:
            print(
                f"[warn] GFPGAN not available ({e}) – continue without faces",
                file=sys.stderr,
            )
            face_enhancer = None

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        paths = []
        for pat in exts:
            paths += glob.glob(os.path.join(args.input, pat))
        paths = sorted(paths)

    tile_plan = auto_tile_try_list(args.tile)
    t0 = time.time()
    ok = fail = 0

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print("Testing", idx, imgname)

        img = robust_imread(path, unchanged=True)
        if img is None:
            print(f"[skip] cannot read {imgname}")
            fail += 1
            continue

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        try:
            if face_enhancer:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
                _ext = (
                    "png"
                    if img_mode == "RGBA"
                    else (extension[1:] if args.ext == "auto" else args.ext)
                )
                save_path = os.path.join(
                    args.output,
                    (
                        f"{imgname}_{args.suffix}.{_ext}"
                        if args.suffix
                        else f"{imgname}.{_ext}"
                    ),
                )
                out_arr = np.ascontiguousarray(output)
                if out_arr.dtype != np.uint8 and out_arr.dtype != np.uint16:
                    out_arr = out_arr.astype(np.uint8, copy=False)
                write_image(save_path, cast(ImageArray, out_arr))
                ok += 1
            else:
                _ext = extension[1:] if args.ext == "auto" else args.ext
                if img_mode == "RGBA":
                    _ext = "png"
                save_path = os.path.join(
                    args.output,
                    (
                        f"{imgname}_{args.suffix}.{_ext}"
                        if args.suffix
                        else f"{imgname}.{_ext}"
                    ),
                )
                success, used_tile, err = try_enhance(
                    upsampler, img, args.outscale, save_path, tile_plan
                )
                if success:
                    ok += 1
                else:
                    print("Error", err)
                    print(
                        "If you encounter CUDA out of memory, try a smaller --tile (auto-fallback tried)."
                    )
                    fail += 1
        except KeyboardInterrupt:
            raise
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tile with a smaller number."
            )
            fail += 1
        except Exception as e:
            print("Error", e)
            fail += 1

    dt = time.time() - t0
    print(f"[done] ok={ok} fail={fail} dt={dt:.1f}s")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
