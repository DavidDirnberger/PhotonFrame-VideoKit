#!/usr/bin/env python3
# enhance.py
from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import consoleOutput as co
import definitions as defin
import helpers as he
import process_wrappers as pw
import userInteraction as ui
import VideoEncodersCodecs as vc
from cli_parser import dict_with_defaults_from_args
from ffmpeg_perf import autotune_final_cmd

# local modules
from i18n import _, tr

# ────────────────────────────────────────────────────────────────
# Globale Steuerung
# ────────────────────────────────────────────────────────────────
_preset_name: Optional[str] = None
# Begrenzung der Vidstab-Analyse; >0 = begrenzt, <=0 = komplette Länge
STAB_ANALYZE_SECONDS = int(os.environ.get("VM_STAB_SECONDS", "15"))


# ────────────────────────────────────────────────────────────────
#  Hilfsfunktion: Colorbalance lokal (statt video_filters)
# ────────────────────────────────────────────────────────────────
def _build_colorbalance_filter_local(warmth: int, tint: int) -> str:
    """
    Erzeugt einen einfachen colorbalance-String für FFmpeg.
    warmth/tint: 0–100, 50 = neutral.
    """
    if warmth == 50 and tint == 50:
        return "colorbalance"
    # Werte von -1..+1
    w = max(-1.0, min(1.0, (warmth - 50) / 50.0))
    t = max(-1.0, min(1.0, (tint - 50) / 50.0))
    # Wärme: Rot hoch, Blau runter (Midtones)
    rm = +0.35 * w
    bm = -0.35 * w
    # Tint: Grün vs. Magenta (Grün hoch, Rot/Blau leicht runter)
    gm = +0.30 * t

    # Begrenzen
    def clip(x: float) -> float:
        return max(-1.0, min(1.0, x))

    rm = clip(rm)
    bm = clip(bm)
    gm = clip(gm)

    return f"colorbalance=rm={rm:.3f}:gm={gm:.3f}:bm={bm:.3f}"


# ────────────────────────────────────────────────────────────────
#  VEC-BRIDGE: Plan + Helper zum Bauen der Encode-Args
# ────────────────────────────────────────────────────────────────
class _VecPlan(NamedTuple):
    preset_name: str
    target_container: str
    codec_key: str
    input_path: Path
    force_pix_fmt: bool = False
    force_key_at_start: bool = False


def _build_vec_encode_cmd(
    input_path: Path,
    vf_chain: Optional[str],
    *,
    preset_name: str = "casual",
    container: Optional[str] = None,
    desired_codec_key: Optional[str] = None,
) -> List[str]:
    """
    Baut ein FFmpeg-Command (ohne Output-Datei!) mit Hilfe von vec.
    Filter (vf_chain) kommen aus enhance. Container/Codec werden nicht extra abgefragt.
    """
    if vc is None:
        raise RuntimeError(
            "Das Modul 'vec' konnte nicht importiert werden. Stelle sicher, dass es im PYTHONPATH liegt."
        )

    # 1) Container ermitteln (keep-Logik)
    detected = (
        vc.detect_container_from_path(input_path)
        if hasattr(vc, "detect_container_from_path")
        else None
    )
    container = container or detected or "mp4"

    # 2) passenden Codec-Key je Container wählen
    default_key = vc.suggest_codec_for_container(container)
    codec_key = vc.pick_crf_codec_for_container(
        container, desired_codec_key or default_key
    )

    # 3) Basis-Command (Input + Mapping + Audio)
    base_cmd: List[str] = ["ffmpeg", "-y", "-i", str(input_path)]
    base_cmd += vc.build_stream_mapping_args(container, input_path=input_path)
    base_cmd += vc.build_audio_args(
        container, preset_name=preset_name, input_path=input_path
    )

    # 4) Encoder + Preset + Fallbacks; Filter injizieren
    cmd_wo_out = vc.try_encode_with_fallbacks(
        base_cmd, codec_key, container, preset_name, vf_chain
    )

    # 5) Post-Processing (PixFmt, Color-Signaling, Container-Quirks, Reihenfolge)
    plan = _VecPlan(
        preset_name=preset_name,
        target_container=container,
        codec_key=codec_key,
        input_path=input_path,
    )
    cmd_wo_out = vc.postprocess_cmd_all_presets(cmd_wo_out, plan)

    return cmd_wo_out


def _percent_int_or_none_safe(v: Any) -> Optional[int]:
    """Wrapper auf deine bestehende Helper-Funktion, defensiv."""
    try:
        return he.percent_int_or_none(v)  # existiert schon bei dir
    except Exception:
        try:
            # Fallback: Zahl/Fraction/str -> int
            if v is None:
                return None
            s = str(v).strip().replace(",", ".").replace("%", "")
            if "/" in s:
                num, den = s.split("/", 1)
                return int(round(float(num) / float(den) * 100))
            return int(round(float(s)))
        except Exception:
            return None


def _build_enhance_param_view(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Erzeugt eine 'Sicht' auf params speziell für die Enhance-Ausgabe.
    - Stabilisierung: Methode immer zeigen, passende Unter-Parameter je nach Methode
    - Denoise: Methode (bei aktivem denoise) + Intensität
    - Color: nur anzeigen, wenn ≠ 50 %, dabei als 'NN%'
    - Sonstige Keys bleiben unberührt
    """
    p = dict(params)  # nicht mutierend

    # --- Leere/None-Parameter, die wir explizit NICHT sehen wollen ---
    # Falls keine Voreinstellung / kein Output gesetzt wurde, gar nicht anzeigen.
    for key in ("preset", "output"):
        if key in p and p[key] is None:
            p.pop(key, None)

    # --- Stabilisierung ---
    stab_on = bool(p.get("stabilize"))
    method = str(p.get("stab_method") or "")
    # Standard: Unterkeys ausblenden
    for k in ("stab_smooth", "stab_rx", "stab_ry"):
        # werden gleich ggf. wieder aktiviert
        if k in p:
            p.pop(k, None)
    if stab_on and method:
        # Methode erzwingen
        p["stab_method"] = method
        if method == "vidstab":
            if "stab_smooth" in params:
                p["stab_smooth"] = params.get("stab_smooth")
        elif method == "deshake":
            if "stab_rx" in params:
                p["stab_rx"] = params.get("stab_rx")
            if "stab_ry" in params:
                p["stab_ry"] = params.get("stab_ry")
    else:
        # komplett ausblenden
        p.pop("stab_method", None)

    # --- Denoise ---
    if bool(p.get("denoise")):
        # Methode immer zeigen, falls vorhanden
        if "denoise_method" in params:
            p["denoise_method"] = params.get("denoise_method")
        # Intensität (falls vorhanden) belassen
        if "denoise_intensity" in params:
            p["denoise_intensity"] = params.get("denoise_intensity")
    else:
        p.pop("denoise_method", None)
        p.pop("denoise_intensity", None)
        # Intensität darf stehenbleiben oder entfernt werden — hier neutral: belassen falls gesetzt

    # --- Color (nur ≠ 50%) ---
    for key in ("warmth", "tint", "brightness", "contrast", "saturation"):
        if key not in params:
            continue
        vint = _percent_int_or_none_safe(params.get(key))
        if vint is None or vint == 50:
            p.pop(key, None)
        else:
            p[key] = f"{vint}%"

    return p


def _print_selected_params_table_enhance(
    params: Dict[str, Any],
    labelcolor: str = "sky_blue",
    *,
    labels: Optional[Dict[str, Any]] = None,
    relevant_groups: Optional[Dict[str, Any]] = None,
    show_files: bool = True,
) -> None:
    """
    Enhance-spezifischer Wrapper um die generische Tabelle.
    Bereitet die Parameteransicht auf und ruft dann die generische Ausgabe.
    """
    prepared = _build_enhance_param_view(params)
    # Für Enhance: Relevanz-Gating standardmäßig deaktivieren,
    # damit alle in 'prepared' verbliebenen Keys angezeigt werden.
    if relevant_groups is None:
        rg: Dict[str, Any] = {}
    else:
        rg = relevant_groups
    # Optional: relevante Keys überschreiben, z. B. um Methode/Unter-Keys sicher einzuschließen
    # Wenn du RELEVANT_PARAM_GROUPS schon passend definiert hast, kannst du das einfach durchreichen.
    co.print_selected_params_table(
        prepared,
        labelcolor,
        labels=labels,
        relevant_groups=rg,
        show_files=show_files,
    )


# ────────────────────────────────────────────────────────────────
#  Enhance-Argumente (safe Defaults für TypedDict)
# ────────────────────────────────────────────────────────────────
def _stab_default_vidstab() -> int:
    try:
        return int(defin.STABILIZATION.get("vidstab", {}).get("default", 16))  # type: ignore[reportGeneralTypeIssues]
    except Exception:
        return 16


def _stab_defaultx() -> int:
    try:
        return int(defin.STABILIZATION.get("deshake", {}).get("defaultx", 50))  # type: ignore[reportGeneralTypeIssues]
    except Exception:
        return 50


def _stab_defaulty() -> int:
    try:
        return int(defin.STABILIZATION.get("deshake", {}).get("defaulty", 50))  # type: ignore[reportGeneralTypeIssues]
    except Exception:
        return 50


def _denoise_default_nlmeans() -> int:
    try:
        return int(defin.NOISE_REDUCTION.get("nlmeans", {}).get("default", 50))  # type: ignore[reportGeneralTypeIssues]
    except Exception:
        return 50


@dataclass
class EnhanceArgs:
    files: List[str] = field(default_factory=list)
    preset: Optional[str] = None
    stabilize: Optional[bool] = False
    stab_method: Optional[str] = "vidstab"
    stab_smooth: Optional[int] = field(default_factory=_stab_default_vidstab)
    stab_rx: Optional[int] = field(default_factory=_stab_defaultx)
    stab_ry: Optional[int] = field(default_factory=_stab_defaulty)
    denoise: Optional[bool] = False
    denoise_method: Optional[str] = "nlmeans"
    denoise_intensity: Optional[int] = field(default_factory=_denoise_default_nlmeans)
    warmth: Optional[int] = 50
    tint: Optional[int] = 50
    brightness: Optional[int] = 50
    contrast: Optional[int] = 50
    saturation: Optional[int] = 50
    output: Optional[str] = None


def _adjust_all_params(args: Any) -> Any:
    """
    Prüft, ob relevante Unterparameter für z.B. stabilize oder denoise gesetzt sind,
    und setzt dann das entsprechende Hauptflag auf True.
    Gibt ein angepasstes Namespace- oder Dataclass-Objekt zurück.
    """
    args = deepcopy(args)

    # Stabilize prüfen
    if (
        getattr(args, "stab_method", None) is not None
        or getattr(args, "stab_smooth", None) is not None
        or getattr(args, "stab_rx", None) is not None
        or getattr(args, "stab_ry", None) is not None
    ):
        setattr(args, "stabilize", True)

    # Denoise prüfen
    if (
        getattr(args, "denoise_method", None) is not None
        or getattr(args, "denoise_intensity", None) is not None
    ):
        setattr(args, "denoise", True)

    return args


def _filter_params_active(params: Mapping[str, Any]) -> bool:
    """
    True sobald irgendwo ein Filter aktiv ist:
     - Stabilisierung
     - Denoising
     - Color-Sliders ≠ 50
     - filter_chain explizit gesetzt / Preset
    """
    # 1) eigener filter_chain im preset
    fc = params.get("filter_chain")
    if fc:
        if isinstance(fc, str) and fc.strip():
            return True
        if isinstance(fc, (list, tuple)) and len(fc) > 0:
            return True

    # 2) Stabilisierung
    if params.get("stabilize"):
        return True

    # 3) Denoising
    if params.get("denoise"):
        return True

    # 4) Farb-Sliders
    for key in ("warmth", "tint", "brightness", "contrast", "saturation"):
        val = params.get(key)
        if isinstance(val, (int, float)) and val != 50:
            return True

    # 5) Preset
    global _preset_name
    preset = params.get("preset")
    if preset and preset != "custom":
        for k in defin.ENHANCE_PRESETS.keys():  # type: ignore[reportGeneralTypeIssues]
            if k == preset:
                _preset_name = preset
                return True

    return False


# ——— Helper: sichere Prozent-Koerzierung (akzeptiert "15", "15%", "0,8", "0.8")
def _coerce_percent_value(x: Any, default: float = 50.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        s = str(x).strip().replace(",", ".").rstrip("%")
        try:
            v = float(s)
        except Exception:
            return default
    return max(0.0, min(100.0, v))


# ——— Helper: Mapping 0–100 % → Sigma je Verfahren
def _sigma_from_percent(method: str, percent: float) -> float:
    p = max(0.0, min(100.0, percent))
    m = (method or "").lower()
    if m == "hqdn3d":
        # wie bisher ~0.08*%, aber mit sanfter Kappung (verhindert unplausible 10+)
        s = round(p * 0.08, 2)  # 15% → 1.2 (wie in deinem Log)
        return max(0.0, min(8.0, s))
    # nlmeans: typische Skala 1..30
    return max(1.0, min(30.0, round(1.0 + (p / 100.0) * 29.0, 2)))


# ——— Helper: deshake-Parameter auf 16er-Raster zwingen (min=16, max=64)
def _quantize_deshake(
    val: Any, *, min_val: int = 16, max_val: int = 64, step: int = 16
) -> int:
    try:
        n = int(val)
    except Exception:
        n = min_val
    n = max(min_val, min(max_val, n))
    q = int(round(n / step)) * step
    return max(min_val, min(max_val, q))


# ────────────────────────────────────────────────────────────────
#  Denoise-Filter mit Overloads (für Pylance)
# ────────────────────────────────────────────────────────────────
@overload
def _build_denoise_filter(
    params: Mapping[str, Any], for_filename: Literal[True]
) -> Tuple[List[str], Optional[str]]: ...
@overload
def _build_denoise_filter(
    params: Mapping[str, Any], for_filename: Literal[False] = ...
) -> List[str]: ...


def _build_denoise_filter(
    params: Mapping[str, Any], for_filename: bool = False
) -> Union[List[str], Tuple[List[str], Optional[str]]]:
    """
    Liefert den passenden Denoise-Filterstring (Liste) oder
    (Liste, Dateitag) wenn for_filename=True.
    - Robust gegen Eingaben wie "15%", "0,8", "0.8".
    - Konsistente 0–100 %-Skala für hqdn3d und nlmeans.
    """
    if not params.get("denoise"):
        return ([], None) if for_filename else []

    method = str(params.get("denoise_method") or "hqdn3d").lower()
    pct = _coerce_percent_value(params.get("denoise_intensity"), 50.0)

    if method == "hqdn3d":
        sigma = _sigma_from_percent("hqdn3d", pct)
        filterstr = f"hqdn3d={sigma}:{sigma}:{sigma}:{sigma}"
    else:
        sigma = _sigma_from_percent("nlmeans", pct)
        filterstr = f"nlmeans=s={sigma}"

    if for_filename:
        # Beibehaltung deines bisherigen Tag-Schemas (σ * 100 → int)
        sigma_tag = int(round(sigma * 100))
        filename_tag = f"denoise{method}{sigma_tag}"
        return [filterstr], filename_tag

    return [filterstr]


def _build_color_correction_filters(params: Mapping[str, Any]) -> List[str]:
    """
    Baut FFmpeg-Filter für Warm/Tint, Brightness/Contrast und Saturation.
    Alle Regler sind 0-100 %, wobei 50 neutral bedeutet.
    """
    required = ("warmth", "tint", "brightness", "contrast", "saturation")
    if not all(k in params for k in required):
        return []

    warmth = int(params.get("warmth", 50))
    tint = int(params.get("tint", 50))
    bright = int(params.get("brightness", 50))
    cont = int(params.get("contrast", 50))
    sat = int(params.get("saturation", 50))

    filters: List[str] = []

    # Warmth/Tint (lokaler Ersatz für video_filters.build_colorbalance_filter)
    if warmth != 50 or tint != 50:
        cb = _build_colorbalance_filter_local(warmth, tint)
        if cb != "colorbalance":
            filters.append(cb)

    # Brightness/Contrast
    if bright != 50 or cont != 50:
        b = (bright - 50) / 50.0
        c = 1.0 + (cont - 50) / 50.0
        filters.append(f"eq=brightness={b:.3f}:contrast={c:.3f}")

    # Saturation
    if sat != 50:
        s = 1.0 + (sat - 50) / 50.0
        filters.append(f"eq=saturation={s:.3f}")

    return filters


# ────────────────────────────────────────────────────────────────
#  collect_individual_params – mit Back-Option
# ────────────────────────────────────────────────────────────────
def _collect_individual_params(sample_path: Path) -> Optional[Dict[str, Any]]:
    """
    Fragt alle Filter interaktiv ab und liefert das Param-Dict zurück.
    Gibt None zurück, wenn der Benutzer irgendwo „Zurück / Back“ gewählt hat.
    """
    params: Dict[str, Any] = {}

    # 1) Stabilisierung
    yn = ui.ask_yes_no(
        "apply_stabilization", explanation="stabilization_explanation", default=True
    )
    if yn is None:
        return None
    if yn:
        stab_keys = list(defin.STABILIZATION)  # type: ignore[reportGeneralTypeIssues]
        stab_labels = [
            tr(defin.STABILIZATION[k].get("description", k)) for k in stab_keys
        ]  # type: ignore[reportGeneralTypeIssues]

        stab_method = ui.ask_user(_("select_stabilization"), stab_keys, stab_labels, 0)
        if stab_method is None:
            return None

        if stab_method == "vidstab":
            default_smooth = int(
                defin.STABILIZATION.get("vidstab", {}).get("default", 16)
            )  # type: ignore[reportGeneralTypeIssues]
            smooth = ui.read_percent("vidstab_smooth_prompt", default_smooth)
            params.update(
                {
                    "stabilize": True,
                    "stab_method": "vidstab",
                    "stab_smooth": int(round(smooth)),
                }
            )
        else:  # deshake
            defx = int(defin.STABILIZATION.get("deshake", {}).get("defaultx", 50))  # type: ignore[reportGeneralTypeIssues]
            defy = int(defin.STABILIZATION.get("deshake", {}).get("defaulty", 50))  # type: ignore[reportGeneralTypeIssues]
            rx_pct = ui.read_percent("deshake_rx_prompt", defx)
            ry_pct = ui.read_percent("deshake_ry_prompt", defy)

            # 0..100 % → 0..64 → auf 16er-Raster (mind. 16)
            rx_raw = int(round(_coerce_percent_value(rx_pct, defx) / 100.0 * 64))
            ry_raw = int(round(_coerce_percent_value(ry_pct, defy) / 100.0 * 64))
            rx = _quantize_deshake(rx_raw)
            ry = _quantize_deshake(ry_raw)

            params.update(
                {
                    "stabilize": True,
                    "stab_method": "deshake",
                    "stab_rx": rx,
                    "stab_ry": ry,
                }
            )
    else:
        params["stabilize"] = False

    # 2) Denoising
    yn = ui.ask_yes_no(
        "apply_noise_reduction", explanation="noise_explanation", default=True
    )
    if yn is None:
        return None
    if yn:
        noise_keys = list(defin.NOISE_REDUCTION)  # type: ignore[reportGeneralTypeIssues]
        noise_labels = [
            tr(defin.NOISE_REDUCTION[k].get("description", k)) for k in noise_keys
        ]  # type: ignore[reportGeneralTypeIssues]
        while True:
            denoise_method = ui.ask_user(
                _("select_denoising_filter"), noise_keys, noise_labels, 0
            )
            if denoise_method is None:
                return None

            default_int = int(
                defin.NOISE_REDUCTION.get(denoise_method, {}).get("default", 50)
            )  # type: ignore[reportGeneralTypeIssues]
            intensity_pct = _coerce_percent_value(
                ui.read_percent("noise_intensity_promt", default_int), default_int
            )

            if denoise_method == "hqdn3d":
                sigma = _sigma_from_percent("hqdn3d", intensity_pct)
                filt_chain = [f"hqdn3d={sigma}:{sigma}:{sigma}:{sigma}"]
            else:  # nlmeans
                sigma = _sigma_from_percent("nlmeans", intensity_pct)
                filt_chain = [f"nlmeans=s={sigma}"]

            pac = ui.preview_and_confirm(
                sample_path,
                ",".join(filt_chain),
                _("noise_reduction_method"),
                int(round(intensity_pct)),
            )
            if pac is None:
                return None
            if pac:
                params.update(
                    {
                        "denoise": True,
                        "denoise_method": denoise_method,
                        "denoise_intensity": int(round(intensity_pct)),
                    }
                )
                break
    else:
        params["denoise"] = False
        co.print_info(_("skip") + " " + _("noise_reduction_method"))

    # 3) Farb-/Helligkeits-Korrektur
    yn = ui.ask_yes_no(
        "apply_color_correction",
        explanation="color_correction_explanation",
        default=True,
    )
    if yn is None:
        return None
    if yn:
        while True:
            warmth_pct = ui.read_percent("color_warmth_promt", 50)
            tint_pct = ui.read_percent("color_tint_promt", 50)
            brightness_pct = ui.read_percent("color_brightness_promt", 50)
            contrast_pct = ui.read_percent("color_contrast_promt", 50)
            saturation_pct = ui.read_percent("color_saturation_promt", 50)

            params.update(
                {
                    "warmth": warmth_pct,
                    "tint": tint_pct,
                    "brightness": brightness_pct,
                    "contrast": contrast_pct,
                    "saturation": saturation_pct,
                }
            )

            color_filters = _build_color_correction_filters(params)
            if not color_filters:
                co.print_info(_("no_color_correction"))
                break

            color_params = he.format_color_params(params)
            pac = ui.preview_and_confirm(
                sample_path,
                ",".join(color_filters),
                _("color_correction_method"),
                color_params,
            )
            if pac is None:
                return None
            if pac:
                break
    else:
        co.print_info(_("skip") + " " + _("color_correction_method"))

    return params


# ────────────────────────────────────────────────────────────────
#  enhance – vec-gestützte Kommando-Erstellung, Filter bleiben hier
# ────────────────────────────────────────────────────────────────
def enhance(args: Any) -> None:
    """
    All-In-One Video Enhancement:
    Stabilisierung, Denoising, Farbkorrektur – interaktiv oder CLI/batch.
    """
    co.print_start(_("enhancing_method"))

    BATCH_MODE, files = he.prepare_inputs(args)

    # 1) PARAMETER ERMITTELN (Integrität bleibt!)
    if BATCH_MODE:
        pargs = _adjust_all_params(args)
        params: Dict[str, Any] = dict_with_defaults_from_args(EnhanceArgs, pargs)
    else:
        sample_path = Path(files[0])
        while True:
            params_opt = _interactive_params(sample_path)
            if params_opt is not None:
                params = params_opt
                break

    # Fallback auf Default-Preset
    if BATCH_MODE and params is None:  # type: ignore[reportUnboundVariable]
        params = dict(defin.ENHANCE_PRESETS.get("realistic", {}))  # type: ignore[reportGeneralTypeIssues]

    # Keine Filter → raus
    if not _filter_params_active(params):
        co.print_fail(_("no_filter_no_video"))
        co.print_finished(_("enhancing_method"))
        return

    # 2) FILTER-KETTE BAUEN (weiterhin hier)
    filter_chain = _build_full_filter_chain(params)
    filename_tags = _filename_tags_from(params)

    # 3) VERARBEITEN – vec baut Encode-Args (Codec/Container/Audio/PixFmt/Color/Quirks)
    global _preset_name
    for file in files:
        path = Path(file)
        tag = "_".join(filename_tags) if filename_tags else "enhanced"
        try:
            output = path.with_stem(f"{path.stem}_{tag}")
        except AttributeError:
            # Python < 3.9 fallback
            output = path.with_name(f"{path.stem}_{tag}{path.suffix}")

        # Anzeige der Param-Tabelle wie bisher
        if _preset_name is None:
            _print_selected_params_table_enhance(params)
        else:
            co.print_preset_params_table(_preset_name)

        if params.get("stabilize") and params.get("stab_method") == "vidstab":
            # Analyse-Pass
            with tempfile.NamedTemporaryFile(suffix=".trf", delete=False) as tf:
                trf_path = tf.name

            detect_cmd: List[str] = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vf",
                f"vidstabdetect=shakiness=5:accuracy=15:result={trf_path}",
            ]
            # Nur begrenzen, wenn STAB_ANALYZE_SECONDS > 0
            if STAB_ANALYZE_SECONDS > 0:
                detect_cmd += ["-t", str(STAB_ANALYZE_SECONDS)]
            detect_cmd += ["-f", "null", "-"]

            pw.run_ffmpeg_with_progress(
                path.name,
                detect_cmd,
                _("analyze_vidstab"),
                _("analyze_vidstab_done"),
                path.name,
                analysis_mode=True,
            )

            # Filter inkl. Transform
            vf_chain = ",".join(
                f"{f}:input={trf_path}" if f.startswith("vidstabtransform") else f
                for f in filter_chain
            )

            cmd_no_out = _build_vec_encode_cmd(path, vf_chain, preset_name="casual")
            cmd = autotune_final_cmd(path, cmd_no_out + [str(output)])

            # co.print_debug("ffmpeg cmd",cmd=cmd)
            # co.print_debug("output",output=output)
            # subprocess.run(cmd)
            pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("enhancing_file_progress"),
                _("enhancing_file_done"),
                output,
                BATCH_MODE=BATCH_MODE,
            )

            try:
                os.remove(trf_path)
            except OSError:
                pass

        else:
            # kein Analyse-Pass nötig
            vf_chain = ",".join(filter_chain) if filter_chain else None

            cmd_no_out = _build_vec_encode_cmd(path, vf_chain, preset_name="casual")
            cmd = autotune_final_cmd(path, cmd_no_out + [str(output)])

            # co.print_debug("ffmpeg cmd",cmd=cmd)
            # co.print_debug("output",output=output)
            # subprocess.run(cmd)
            pw.run_ffmpeg_with_progress(
                path.name,
                cmd,
                _("enhancing_file_progress"),
                _("enhancing_file_done"),
                output,
                BATCH_MODE=BATCH_MODE,
            )

    co.print_finished(_("enhancing_method"))


# ——————————————————————————————————————————————
# Helpers


def _interactive_params(sample_path: Path) -> Optional[Dict[str, Any]]:
    """Preset-Dialog + Preview-Loop, gibt params (oder None bei Back) zurück."""
    preset_keys = list(defin.ENHANCE_PRESETS)  # type: ignore[reportGeneralTypeIssues]
    preset_labels = [tr(defin.ENHANCE_PRESETS[k].get("name", k)) for k in preset_keys]  # type: ignore[reportGeneralTypeIssues]
    preset_descs = [
        tr(defin.ENHANCE_PRESETS[k].get("description", "")) for k in preset_keys
    ]  # type: ignore[reportGeneralTypeIssues]

    global _preset_name

    while True:
        _preset_name = None
        choice = ui.ask_user(
            _("enhance_presets_list"),
            preset_keys,
            preset_descs,
            default=1,
            display_labels=preset_labels,
        )
        if choice is None:
            return None
        if choice == "custom":
            return _collect_individual_params(sample_path)

        # Preset gewählt
        spec_map = defin.ENHANCE_PRESETS.get(choice, {})  # type: ignore[reportGeneralTypeIssues]
        params: Dict[str, Any] = {
            k: v for k, v in spec_map.items() if k not in ("name", "description")
        }

        _preset_name = choice

        # Preview-Loop (ohne trim/setpts, damit preview_and_confirm stabil extrahieren kann)
        while True:
            chain = _build_preview_chain(params)
            if not chain:
                co.print_fail(_("preset_no_active_filter"))
                break  # zurück in den äußeren Wahl-Loop

            if ui.preview_and_confirm(sample_path, ",".join(chain), "Preset", choice):
                return params  # akzeptiert → sofort zurück
            else:
                break  # abgelehnt → zurück in den äußeren Wahl-Loop


def _build_preview_chain(params: Mapping[str, Any]) -> List[str]:
    """
    Vorschau-Kette NUR mit Filtern, die die Timeline nicht kürzen/verschieben.
    Kein 'trim' / 'setpts' → verhindert fehlende Preview-Frames.
    """
    denoise_filters = cast(List[str], _build_denoise_filter(params))
    color_filters = _build_color_correction_filters(params)
    return list(denoise_filters) + list(color_filters)


def _build_full_filter_chain(params: Mapping[str, Any]) -> List[str]:
    """
    Komplette Filter-Kette:
      1) Stabilisierung
      2) Denoise
      3) Color Correction
    """
    chain: List[str] = []

    # 1) Stabilisierung
    if params.get("stabilize"):
        if params.get("stab_method") == "vidstab":
            smooth = int(params.get("stab_smooth", 16))
            chain.append(f"vidstabtransform=smoothing={smooth}")
        else:
            # Deshake: Werte defensiv auf 16er-Raster (16..64) zwingen
            rx = _quantize_deshake(params.get("stab_rx", 16))
            ry = _quantize_deshake(params.get("stab_ry", 16))
            # Farbformat vor deshake stabilisieren
            chain.append("format=yuv420p")
            chain.append(f"deshake=rx={rx}:ry={ry}")

    # 2) Denoising
    denoise_filters = _build_denoise_filter(params)
    chain.extend(denoise_filters)

    # 3) Farbkorrektur
    color_filters = _build_color_correction_filters(params)
    chain.extend(color_filters)

    return chain


def _filename_tags_from(params: Mapping[str, Any]) -> List[str]:
    """
    Erzeugt eine Liste von Tags für den Dateinamen, u.a. aus Denoise und Color-Slidern.
    """
    tags: List[str] = []

    # Denoise-Tag
    denoise_filters_and_tag = _build_denoise_filter(params, for_filename=True)
    denoise_tag = denoise_filters_and_tag[1]
    if denoise_tag:
        tags.append(denoise_tag)

    # Color-Tags (warm, tint, bri, con, sat)
    for short, key in [
        ("warm", "warmth"),
        ("tint", "tint"),
        ("bri", "brightness"),
        ("con", "contrast"),
        ("sat", "saturation"),
    ]:
        val = params.get(key, 50)
        if isinstance(val, (int, float)) and val != 50:
            tags.append(f"{short}{int(round(val))}")

    return tags
