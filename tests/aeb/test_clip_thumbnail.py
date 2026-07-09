"""Tests for the AEB clip screenshot thumbnail path."""

from __future__ import annotations

import base64
import io

import pytest

from core.aeb.recorder import AEBClipRecorder
from core.aeb.clip_store import deserialize_clip, serialize_clip

from tests.aeb.test_clip_capture import _StubWriter, _drive, _make_clip


def test_encode_thumbnail_downscales_and_roundtrips():
    Image = pytest.importorskip("PIL.Image")
    from core.aeb.screenshot import encode_thumbnail

    img = Image.new("RGB", (1920, 1080), (10, 20, 30))
    b64 = encode_thumbnail(img, max_px=480, quality=50)
    back = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert back.format == "JPEG"
    assert max(back.size) <= 480          # long side downscaled


def test_grab_thumbnail_never_raises():
    # No display / no Pillow in CI must yield None, not an exception.
    from core.aeb.screenshot import grab_thumbnail

    result = grab_thumbnail()
    assert result is None or isinstance(result, str)


def test_thumbnail_survives_clip_serialization():
    clip = _make_clip()
    clip.metadata.thumbnail_jpeg = base64.b64encode(b"fake-jpeg").decode("ascii")
    back = deserialize_clip(serialize_clip(clip))
    assert back.metadata.thumbnail_jpeg == clip.metadata.thumbnail_jpeg


def test_recorder_attaches_thumbnail_from_provider():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0,
                          screenshot_provider=lambda: "THUMB64")
    _drive(rec, 0.0, 3.0)
    rec.trigger("manual")
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips
    assert writer.clips[0].metadata.thumbnail_jpeg == "THUMB64"


def test_recorder_without_provider_has_no_thumbnail():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
    _drive(rec, 0.0, 3.0)
    rec.trigger("manual")
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips[0].metadata.thumbnail_jpeg is None


def test_recorder_thumbnail_provider_failure_is_swallowed():
    def _boom():
        raise RuntimeError("grab failed")

    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0, screenshot_provider=_boom)
    _drive(rec, 0.0, 3.0)
    rec.trigger("manual")
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips[0].metadata.thumbnail_jpeg is None
