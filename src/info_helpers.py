# ========================= info_helpers.py =========================
from __future__ import annotations

from typing import Any, Dict

# Struktur:
# {
#   "headline": {de, en},
#   "blocks": [
#       {"title": {de, en}, "bullets": [{de, en}, ...]},
#       ...
#   ]
# }

_PRESET_INFO: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------
    # Spezieller Text für das Preset "lossless"
    # ------------------------------------------------------------
    "lossless": {
        "headline": {
            "de": "Voreinstellung: Lossless – Hinweise zu Container/Codec-Kombinationen",
            "en": "Preset: Lossless – Notes on container/codec combinations",
        },
        "blocks": [
            {
                "title": {
                    "de": "Empfehlungen (strikt lossless)",
                    "en": "Recommendations (strict lossless)",
                },
                "bullets": [
                    {
                        "name": "MKV + FFV1",
                        "description": {
                            "de": "Echte, mathematisch verlustfreie Speicherung (bitgenau). Sehr robust für Archiv und Master-Dateien. Nachteile: sehr große Dateien; primär für Desktop-Wiedergabe und professionelle Workflows geeignet, viele TV-Apps/Smartphones unterstützen es nicht.",
                            "en": "Truly lossless (bit-exact) storage. Very robust for archiving and master files. Downsides: very large files; mainly for desktop/pro workflows, many TV apps/phones won’t support it.",
                        },
                    },
                    {
                        "name": "MKV + H.264",
                        "description": {
                            "de": "Streng verlustfrei möglich. Gute Wahl, wenn das Zielmaterial im H.264-Ökosystem bleiben soll. Für PC-Player unproblematisch; bei älteren oder sehr einfachen Geräten kann die Unterstützung variieren.",
                            "en": "Strictly lossless is possible. Good when you want to stay in the H.264 ecosystem. Desktop players handle it well; support on older or very simple devices may vary.",
                        },
                    },
                    {
                        "name": "MKV + HEVC",
                        "description": {
                            "de": "Streng verlustfrei möglich und häufig kleinere Dateien als H.264-Lossless. Moderne Abspielgeräte/Player kommen gut zurecht; sehr alte Geräte eher nicht.",
                            "en": "Strict lossless is possible, often with smaller files than H.264 lossless. Modern players/devices handle it well; very old ones may not.",
                        },
                    },
                    {
                        "name": "AVI + UTVideo/HuffYUV",
                        "description": {
                            "de": "Streng verlustfrei und sehr schnell – beliebt für Aufnahmen und Zwischenschritte. Dateien sind sehr groß. AVI ist ein älterer Container, funktioniert aber in vielen Schnittprogrammen zuverlässig.",
                            "en": "Strictly lossless and fast—popular for capture and intermediates. Files are very large. AVI is older but widely supported in editors.",
                        },
                    },
                ],
            },
            {
                "title": {
                    "de": "Lossless (nur falls MP4 zwingend ist)",
                    "en": "Lossless (only if MP4 is mandatory)",
                },
                "bullets": [
                    {
                        "name": "MP4 + H.264",
                        "description": {
                            "de": "Technisch verlustfrei möglich und insgesamt am kompatibelsten innerhalb der MP4-Welt. Geeignet, wenn Distribution über MP4 benötigt wird; als Langzeit-Archivformat hingegen weniger effizient.",
                            "en": "Technically lossless is possible and overall the most compatible choice within MP4. Suitable when you must distribute in MP4; less efficient as a long-term archival format.",
                        },
                    },
                    {
                        "name": "MP4 + HEVC",
                        "description": {
                            "de": "Technisch verlustfrei möglich und effizienter als H.264. Gut auf modernen Geräten/TVs; ältere Geräte können Einschränkungen haben.",
                            "en": "Technically lossless is possible and more efficient than H.264. Works well on modern devices/TVs; older devices may have limitations.",
                        },
                    },
                ],
            },
            {
                "title": {
                    "de": "Web/Browser (technisch möglich, aber mit Bedacht)",
                    "en": "Web/browsers (technically possible, use with care)",
                },
                "bullets": [
                    {
                        "name": "WebM + VP9",
                        "description": {
                            "de": "Lossless funktioniert, ist aber eher ein Spezialfall. Für reines Web-Delivery selten sinnvoll, da die Dateien groß werden; eher für spezielle Workflows innerhalb des Browsers gedacht.",
                            "en": "Lossless works but is niche. Rarely ideal for pure web delivery due to large files; more for specialized in-browser workflows.",
                        },
                    },
                    {
                        "name": "WebM + AV1",
                        "description": {
                            "de": "Lossless ist möglich. Unterstützung in Browsern und Hardware wächst, Performance-Anforderungen sind jedoch höher – Einsatz daher abwägen.",
                            "en": "Lossless is possible. Browser/hardware support is growing, but performance demands are higher—use judiciously.",
                        },
                    },
                ],
            },
            {
                "title": {
                    "de": "Mezzanine (near-lossless, schnittfreundlich)",
                    "en": "Mezzanine (near-lossless, edit-friendly)",
                },
                "bullets": [
                    {
                        "name": "MOV + ProRes",
                        "description": {
                            "de": "Sehr schnitt- und farbkorrekturfreundlich, allerdings nicht mathematisch verlustfrei. Weit verbreitet im Profi-Bereich. Dateien sind groß, dafür sehr flüssig im Schnitt und in der Vorschau.",
                            "en": "Very editor- and color-friendly, but not mathematically lossless. Widely used in professional workflows. Large files, but smooth to edit and preview.",
                        },
                    },
                    {
                        "name": "MOV + DNxHR",
                        "description": {
                            "de": "Für Schnitt/Color-Workflows gedacht; ebenfalls nicht streng verlustfrei. Gute Performance auf vielen Systemen, große aber handhabbare Dateien.",
                            "en": "Designed for editing/color workflows; also not strictly lossless. Good performance on many systems; large but manageable files.",
                        },
                    },
                    {
                        "name": "MOV/MKV + JPEG 2000",
                        "description": {
                            "de": "Hochwertiges Mezzanine für Archiv/Spezial-Pipelines. Visuell sehr stabil, jedoch relativ große Dateien und höhere Rechenlast. Geeignet, wenn Qualität und Langzeitstabilität Vorrang haben.",
                            "en": "High-quality mezzanine for archive/specialized pipelines. Visually very stable, but relatively large files and higher compute cost. Suitable when quality and long-term stability come first.",
                        },
                    },
                ],
            },
        ],
    },
    # ------------------------------------------------------------
    # Einheitliche Infos für alle NICHT-Lossless-Presets
    # (casual, ultra, studio, cinema, …)
    # ------------------------------------------------------------
    "default": {
        "headline": {
            "de": "Hinweise zu Format/Codec (allgemein, nicht-lossless)",
            "en": "Format/codec guidance (general, non-lossless)",
        },
        "blocks": [
            {
                "title": {
                    "de": "Empfohlene Kombis & Einsatzbereiche",
                    "en": "Recommended combos & use-cases",
                },
                "bullets": [
                    {
                        "name": "MP4 + H.264",
                        "description": {
                            "de": "Sehr hohe Kompatibilität mit praktisch allen Geräten und Plattformen. Ideal zum Teilen, für Social, Präsentationen und den Alltag.",
                            "en": "Extremely compatible with almost all devices and platforms. Ideal for sharing, social, presentations, and everyday use.",
                        },
                    },
                    {
                        "name": "MP4 + HEVC",
                        "description": {
                            "de": "Moderner und effizienter als H.264 (ähnliche Qualitätswahrnehmung bei kleinerer Datei). Läuft auf aktuellen Geräten/TVs gut; bei älteren Modellen vorher prüfen.",
                            "en": "More efficient than H.264 (similar perceived quality at smaller sizes). Works well on modern devices/TVs; check older models first.",
                        },
                    },
                    {
                        "name": "MP4/WebM + AV1",
                        "description": {
                            "de": "Sehr effizient und zukunftssicher. Die Unterstützung wächst schnell, ältere Geräte/Browser können jedoch noch Probleme haben.",
                            "en": "Very efficient and future-oriented. Support is growing fast, but older devices/browsers may still struggle.",
                        },
                    },
                    {
                        "name": "WebM + VP9",
                        "description": {
                            "de": "Bewährte Web-Option mit breiter Unterstützung in Desktop-Browsern und vielen Plattformen. Gute Wahl für reines Web-Delivery.",
                            "en": "Well-proven web option with wide support in desktop browsers and many platforms. A solid choice for web delivery.",
                        },
                    },
                    {
                        "name": "MKV + H.264/HEVC/AV1/VP9",
                        "description": {
                            "de": "Flexibler Container für PC-Player und Archivierung. Streaming-Sticks/TV-Apps unterstützen MKV uneinheitlich – am PC jedoch sehr zuverlässig, inklusive mehrspuriger Audio/Untertitel.",
                            "en": "Flexible container for PC players and archiving. TV sticks/apps vary in MKV support—very reliable on PCs, including multi-audio and subtitles.",
                        },
                    },
                    {
                        "name": "MOV + ProRes/DNxHR/JPEG 2000",
                        "description": {
                            "de": "Für Schnitt, Farbkorrektur und hochwertige Zwischenformate gedacht. Nicht für breite Distribution, dafür sehr stabil im Editing.",
                            "en": "Designed for editing, color work, and high-quality intermediates. Not for wide distribution; excellent stability in editing.",
                        },
                    },
                ],
            },
            {
                "title": {
                    "de": "Breite Player-Kompatibilität (Kurzüberblick)",
                    "en": "Wide player compatibility (quick view)",
                },
                "bullets": [
                    {
                        "name": "MP4 + H.264",
                        "description": {
                            "de": "Am sichersten und am weitesten verbreitet – „läuft fast überall“.",
                            "en": "Safest and most widely supported—“works almost everywhere”.",
                        },
                    },
                    {
                        "name": "MP4 + HEVC, WebM + VP9",
                        "description": {
                            "de": "Sehr gute Kompatibilität auf modernen Geräten und in aktuellen Browsern.",
                            "en": "Very good compatibility on modern devices and in current browsers.",
                        },
                    },
                    {
                        "name": "MP4/WebM + AV1",
                        "description": {
                            "de": "Sehr effizient; für ältere Geräte/Browsers die Unterstützung vorher testen.",
                            "en": "Highly efficient; test support on older devices/browsers first.",
                        },
                    },
                ],
            },
        ],
    },
}

LOSSLESS_DISCLAIMER = {
    "de": "Hinweis: Wenn Container und Codec eine echte, mathematisch verlustfreie Kodierung zulassen, wird diese genutzt (keine Qualitätsänderung, bitgenau). Für Kombinationen, bei denen echtes Lossless technisch nicht möglich oder unpraktisch ist, verwenden wir automatisch eine sehr hochwertige, visuell verlustfreie Konfiguration innerhalb des gewählten Containers und Codecs.",
    "en": "Note: If the chosen container/codec combination supports true, mathematically lossless encoding, it will be used (bit-exact, no quality change). For combinations where true lossless is not technically feasible or practical, we automatically use a very high-quality, visually lossless configuration within the selected container and codec.",
}


def general_info_for_preset(preset_key: str) -> Dict[str, Any]:
    """Gibt den passenden Info-Block (zweisprachig) für das gewählte Preset zurück."""
    key = (preset_key or "").lower()
    return _PRESET_INFO["lossless"] if key == "lossless" else _PRESET_INFO["default"]


# ======================= /info_helpers.py =======================
