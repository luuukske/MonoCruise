"""Top-of-body video URL extraction for the updater player."""
from shared.markdown_renderer import GitHubMarkdownRenderer


def test_extracts_bare_mp4_at_top():
    r = GitHubMarkdownRenderer()
    r.render("https://example.com/clip.mp4\n\n# Hello")
    assert r.get_video_url() == "https://example.com/clip.mp4"


def test_extracts_bare_webm_at_top():
    r = GitHubMarkdownRenderer()
    html = r.render("https://example.com/clip.webm\n\n# Hello")
    assert r.get_video_url() == "https://example.com/clip.webm"
    assert "clip.webm" not in html


def test_extracts_webm_markdown_link_with_query():
    r = GitHubMarkdownRenderer()
    r.render("[demo](https://example.com/a.webm?raw=1)\n\ntext")
    assert r.get_video_url() == "https://example.com/a.webm?raw=1"


def test_ignores_video_after_content():
    r = GitHubMarkdownRenderer()
    r.render("# Hello\n\nhttps://example.com/clip.webm")
    assert r.get_video_url() is None
