"""Best-effort game JPEG thumbnail for debug AEB clips (core/aeb/capture.py). Never raises."""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)

_MAX_PX = 480
_QUALITY = 50


def encode_thumbnail(image, max_px: int = _MAX_PX, quality: int = _QUALITY) -> str:
    """PIL Image -> base64 JPEG, downscaled so its long side is <= max_px."""
    img = image.convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grab_thumbnail(max_px: int = _MAX_PX, quality: int = _QUALITY) -> str | None:
    """Grab the primary screen; return a base64 JPEG thumbnail, or None on failure."""
    try:
        from PIL import ImageGrab
    except Exception:
        logger.debug("Pillow unavailable; AEB screenshot skipped")
        return None
    try:
        return encode_thumbnail(ImageGrab.grab(), max_px, quality)
    except Exception:
        logger.debug("AEB screenshot grab failed", exc_info=True)
        return None


def decode_thumbnail(b64: str) -> bytes:
    """base64 JPEG string -> raw JPEG bytes (for a viewer)."""
    return base64.b64decode(b64)
