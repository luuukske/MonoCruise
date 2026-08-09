"""Image handling in the shared markdown renderer (consent prompt, updater changelog)."""
from __future__ import annotations

import base64

import pytest

from shared.markdown_renderer import GitHubMarkdownRenderer

_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    b"IQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def img_dir(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "shot.png").write_bytes(_PNG)
    return tmp_path


def test_local_image_is_inlined_as_a_data_uri(img_dir):
    out = GitHubMarkdownRenderer(image_base=img_dir)._process_inline("![a shot](img/shot.png)")
    assert out.startswith('<img src="data:image/png;base64,')
    assert 'alt="a shot"' in out


def test_image_rule_runs_before_the_link_rule(img_dir):
    """The link rule also matches ![alt](src); running it first left a stray '!'."""
    out = GitHubMarkdownRenderer(image_base=img_dir)._process_inline("![a shot](img/shot.png)")
    assert not out.startswith("!")
    assert "[Image:" not in out


def test_remote_image_stays_a_link(img_dir):
    """QTextDocument does no networking, so a remote <img> would render broken."""
    out = GitHubMarkdownRenderer(image_base=img_dir)._process_inline(
        "![shot](https://example.com/a.png)"
    )
    assert out == '<a href="https://example.com/a.png">shot</a>'


def test_without_an_image_base_every_image_is_a_link():
    out = GitHubMarkdownRenderer()._process_inline("![a](img/shot.png)")
    assert out == '<a href="img/shot.png">a</a>'


def test_paths_outside_the_image_base_are_refused(img_dir):
    """Release-note markdown is untrusted; it must not inline arbitrary local files."""
    out = GitHubMarkdownRenderer(image_base=img_dir / "img")._process_inline(
        "![x](../../secret.png)"
    )
    assert out.startswith("<a href=")


def test_missing_and_oversized_files_fall_back_to_a_link(img_dir):
    big = img_dir / "img" / "big.png"
    big.write_bytes(b"\x89PNG" + b"0" * (2 * 1024 * 1024 + 1))
    r = GitHubMarkdownRenderer(image_base=img_dir)
    assert r._process_inline("![x](img/nope.png)").startswith("<a href=")
    assert r._process_inline("![x](img/big.png)").startswith("<a href=")


def test_unknown_extension_is_not_inlined(img_dir):
    (img_dir / "img" / "x.bmp").write_bytes(_PNG)
    out = GitHubMarkdownRenderer(image_base=img_dir)._process_inline("![x](img/x.bmp)")
    assert out.startswith("<a href=")


def test_lone_image_line_gets_the_line_height_reset(img_dir):
    """body line-height multiplies an inline image's own height, leaving a gap under it."""
    html = GitHubMarkdownRenderer(image_base=img_dir).render("text\n\n![x](img/shot.png)\n\ntext")
    assert '<p class="imgblock"><img' in html


def test_a_fallback_link_line_does_not_get_the_class(img_dir):
    """The class exists for real images only; a link needs the normal line height."""
    html = GitHubMarkdownRenderer(image_base=img_dir).render(
        "text\n\n![x](https://example.com/a.png)\n\ntext"
    )
    assert 'class="imgblock"><a' not in html


def test_corners_are_rounded_into_the_pixels(img_dir):
    """Qt rich text ignores CSS border-radius, so the mask has to be baked in."""
    qtgui = pytest.importorskip("PySide6.QtGui")
    QImage = qtgui.QImage
    big = img_dir / "img" / "big.png"
    opaque = QImage(60, 60, QImage.Format_ARGB32)
    opaque.fill(qtgui.QColor("red"))       # a fresh QImage is transparent, not opaque
    opaque.save(str(big))

    uri = GitHubMarkdownRenderer(image_base=img_dir)._image_data_uri("img/big.png")
    img = QImage()
    assert img.loadFromData(base64.b64decode(uri.split(",", 1)[1]))
    assert img.pixelColor(0, 0).alpha() == 0        # masked away
    assert img.pixelColor(30, 30).alpha() == 255    # interior untouched


def test_rounding_forces_png_because_jpeg_has_no_alpha(img_dir):
    QImage = pytest.importorskip("PySide6.QtGui").QImage
    jpg = img_dir / "img" / "photo.jpg"
    QImage(40, 40, QImage.Format_RGB32).save(str(jpg))

    uri = GitHubMarkdownRenderer(image_base=img_dir)._image_data_uri("img/photo.jpg")
    assert uri.startswith("data:image/png;base64,")


def test_svg_is_passed_through_unrounded(img_dir):
    """Rasterising an SVG to round it would throw away its scalability."""
    svg = img_dir / "img" / "icon.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><rect width="9" height="9"/></svg>')

    uri = GitHubMarkdownRenderer(image_base=img_dir)._image_data_uri("img/icon.svg")
    assert uri.startswith("data:image/svg+xml;base64,")


def test_ordinary_links_and_bare_urls_are_unchanged(img_dir):
    r = GitHubMarkdownRenderer(image_base=img_dir)
    assert r._process_inline("[docs](https://x.com)") == '<a href="https://x.com">docs</a>'
    assert r._process_inline("see https://x.com") == 'see <a href="https://x.com">https://x.com</a>'
