# Einheitlicher Import für Pillow.Image mit Fallback-Dummy
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

PIL_OK: bool


# Nur für Typprüfung: eine Image-Schnittstelle bereitstellen
class _PILImageProto(Protocol):
    width: int
    height: int
    mode: str

    def save(self, *args, **kwargs) -> None: ...
    def transpose(self, *args, **kwargs) -> "_PILImageProto": ...
    def convert(self, *args, **kwargs) -> "_PILImageProto": ...


if TYPE_CHECKING:
    # Pylance bekommt die echten Stubs mit .save/.transpose/...
    from PIL.Image import Image as PILImageType  # noqa: F401
else:
    # Laufzeit: nur Protokoll (kein hartes Pillow-Import nötig)
    PILImageType = _PILImageProto  # type: ignore[misc,assignment]

# Für Type-Checker/IDE: sauberes Image-Objekt verfügbar machen, ohne zur Laufzeit zu importieren
if TYPE_CHECKING:
    from PIL.Image import Image as ImageType  # nur fürs Typing
else:
    ImageType = Any  # zur Laufzeit egal

try:
    from PIL import Image as _PILImageModule

    Image = _PILImageModule  # <- wie "from PIL import Image" verwendbar
    PIL_OK = True
except Exception:
    PIL_OK = False

    class _ImageDummy:
        # Damit "Image.Image" existiert und Autocomplete hat
        class Image:
            width: int = 0
            height: int = 0
            mode: str = "RGB"

            def save(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Pillow not available")

            def transpose(self, *args: Any, **kwargs: Any):
                raise RuntimeError("Pillow not available")

            def convert(self, *args: Any, **kwargs: Any):
                raise RuntimeError("Pillow not available")

        class Transpose:
            FLIP_LEFT_RIGHT = 0
            FLIP_TOP_BOTTOM = 1
            TRANSPOSE = 2

        @staticmethod
        def open(*_a: Any, **_k: Any):
            raise RuntimeError("Pillow not available")

        @staticmethod
        def fromarray(*_a: Any, **_k: Any):
            raise RuntimeError("Pillow not available")

    Image = _ImageDummy  # type: ignore[assignment]

__all__ = ["Image", "PIL_OK", "ImageType"]
