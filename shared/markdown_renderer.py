"""GFM markdown to HTML for PySide6; palette from injected Theme. See shared/README.md."""
import re
import html
import base64
import logging
from pathlib import Path

from shared.theme import Theme

logger = logging.getLogger(__name__)

# QTextDocument does no networking, so only local images can be shown. They are
# inlined as data URIs; remote ones stay links. See shared/README.md.
_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
}
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
# A line holding only an image: body line-height would otherwise multiply the
# image's own height (1.6x), leaving a large gap under it.
_IMAGE_ONLY_LINE = re.compile(r'^!\[[^\]]*\]\([^)]+\)$')
# Qt's rich text engine ignores CSS border-radius, so corners are masked into
# the pixels. Needs alpha, hence PNG out regardless of what came in.
_IMAGE_CORNER_RADIUS_PX = 5


def _rounded_png(data: bytes) -> bytes | None:
    """Raster image bytes with rounded corners as PNG, or None to keep the original."""
    try:
        from PySide6.QtCore import QBuffer, QIODevice, Qt
        from PySide6.QtGui import QImage, QPainter, QPainterPath
    except Exception:
        return None
    try:
        img = QImage()
        if not img.loadFromData(data):
            return None
        img = img.convertToFormat(QImage.Format_ARGB32)
        out = QImage(img.size(), QImage.Format_ARGB32)
        out.fill(Qt.transparent)
        painter = QPainter(out)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            path = QPainterPath()
            r = float(_IMAGE_CORNER_RADIUS_PX)
            path.addRoundedRect(0.0, 0.0, float(img.width()), float(img.height()), r, r)
            painter.setClipPath(path)
            painter.drawImage(0, 0, img)
        finally:
            painter.end()
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        if not out.save(buf, "PNG"):
            return None
        return bytes(buf.data())
    except Exception:
        logger.debug("markdown image corner rounding failed", exc_info=True)
        return None


class GitHubMarkdownRenderer:
    """Converts GitHub-flavored Markdown to HTML for QTextBrowser."""

    SVG_ICONS = {
        'NOTE': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-info-icon lucide-info"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>''',
        'TIP': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-lightbulb-icon lucide-lightbulb"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>''',
        'IMPORTANT': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-message-square-warning-icon lucide-message-square-warning"><path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/><path d="M12 15h.01"/><path d="M12 7v4"/></svg>''',
        'WARNING': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert-icon lucide-triangle-alert"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>''',
        'CAUTION': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-octagon-alert-icon lucide-octagon-alert"><path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/></svg>''',
    }

    def __init__(self, theme: Theme | None = None, image_base: "Path | str | None" = None):
        self.theme = theme or Theme()
        # Local image sources resolve against this directory, and only inside it.
        # Left None (the updater's case) every image stays a link.
        self.image_base = Path(image_base).resolve() if image_base is not None else None
        self.video_url = None
        # Alert colours depend on the injected theme, so build the map per instance.
        t = self.theme
        self.ALERT_TYPES = {
            'NOTE': {'color': t.md_color_note, 'icon': 'ℹ️'},
            'TIP': {'color': t.md_color_tip, 'icon': '💡'},
            'IMPORTANT': {'color': t.md_color_important, 'icon': '❗'},
            'WARNING': {'color': t.md_color_warning, 'icon': '⚠️'},
            'CAUTION': {'color': t.md_color_caution, 'icon': '🛑'},
        }

    def render(self, markdown_text: str) -> str:
        """Convert markdown to HTML and extract video URL."""
        self.video_url = None

        if not markdown_text:
            return ""

        # Extract and remove the first video link from the top
        markdown_text = self._extract_and_remove_video(markdown_text)
        html_content = self._process_markdown(markdown_text)

        return self._wrap_html(html_content)

    def render_blocks(self, markdown_text: str) -> list[str]:
        """Like render(), but one HTML block per top-level element (split_lists for animation)."""
        self.video_url = None

        if not markdown_text:
            return []

        markdown_text = self._extract_and_remove_video(markdown_text)
        return self._process_markdown_blocks(markdown_text, split_lists=True)

    def style_css(self) -> str:
        """The <style> tag render()/render_blocks() content relies on, for a caller that renders each block into its own QLabel/QTextDocument. Body padding is horizontal-only: each block is its own document, so
        vertical padding would stack between every pair of labels instead of appearing once around the whole changelog like render() has it."""
        return self._style_tag(body_padding="0px 8px")

    def get_video_url(self) -> str | None:
        """Return the extracted video URL, or None."""
        return self.video_url

    def _extract_and_remove_video(self, text: str) -> str:
        """Extract the first .mp4 video link from the top and remove it."""
        lines = text.split('\n')
        new_lines = []
        video_found = False

        for i, line in enumerate(lines):
            if video_found:
                new_lines.append(line)
                continue

            stripped = line.strip()

            # Skip empty lines at the top before finding video
            if not stripped and not new_lines:
                new_lines.append(line)
                continue

            # Check for markdown link format: [text](url.mp4)
            md_link_match = re.match(r'^\[([^\]]*)\]\((https?://[^\s)]+\.mp4(?:\?[^\s)]*)?)\)\s*$', stripped, re.IGNORECASE)
            if md_link_match:
                self.video_url = md_link_match.group(2)
                video_found = True
                continue

            # Check for bare URL format: https://...mp4
            bare_url_match = re.match(r'^(https?://[^\s<>\[\]()]+\.mp4(?:\?[^\s<>\[\]()]*)?)\s*$', stripped, re.IGNORECASE)
            if bare_url_match:
                self.video_url = bare_url_match.group(1)
                video_found = True
                continue

            # If we hit non-empty, non-video content, stop looking
            if stripped:
                video_found = True  # Stop looking for video

            new_lines.append(line)

        return '\n'.join(new_lines)

    def _process_markdown(self, text: str) -> str:
        """Process markdown text to HTML."""
        return '\n'.join(self._process_markdown_blocks(text))

    def _process_markdown_blocks(self, text: str, split_lists: bool = False) -> list[str]:
        """Markdown to ordered HTML blocks; split_lists splits list items for animation."""
        # Filter out SourceForge badge lines
        text = re.sub(r'^\s*\[!\[.*?\]\(https://a\.fsdn\.com/.*?\)\]\(https://sourceforge\.net/.*?\)\s*$',
                    '', text, flags=re.MULTILINE)

        lines = text.split('\n')
        result = []
        i = 0
        in_code_block = False
        code_block_content = []
        code_language = ""
        in_list = False
        list_items = []
        list_indent_level = 0
        in_alert = False
        alert_type = ""
        alert_content = []
        alert_indent = 0
        in_quote = False
        quote_content = []
        last_was_header = False

        while i < len(lines):
            line = lines[i]

            if line.strip().startswith('```'):
                if in_code_block:
                    result.append(self._render_code_block('\n'.join(code_block_content), code_language))
                    code_block_content = []
                    code_language = ""
                    in_code_block = False
                    last_was_header = False
                else:
                    in_code_block = True
                    code_language = line.strip()[3:].strip()
                i += 1
                continue

            if in_code_block:
                code_block_content.append(line)
                i += 1
                continue

            alert_match = re.match(r'^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', line)
            if alert_match:
                if in_quote:
                    result.append(self._render_quote('\n'.join(quote_content)))
                    quote_content = []
                    in_quote = False

                if in_list:
                    alert_indent = list_indent_level + 1
                else:
                    alert_indent = 0

                alert_type = alert_match.group(1)
                in_alert = True
                alert_content = []
                i += 1
                continue

            if in_alert:
                if line.startswith('>'):
                    content = line[1:].strip() if len(line) > 1 else ""
                    alert_content.append(content)
                    i += 1
                    continue
                else:
                    alert_html = self._render_alert(alert_type, '\n'.join(alert_content))

                    if in_list and alert_indent > 0:
                        if list_items:
                            prev = list_items[-1]
                            list_items[-1] = (
                                prev[0],
                                prev[1],
                                prev[2] + '\n' + f'__ALERT__{alert_html}__ENDALERT__'
                            )
                        else:
                            result.append(alert_html)
                    else:
                        result.append(alert_html)

                    in_alert = False
                    alert_type = ""
                    alert_content = []
                    alert_indent = 0
                    last_was_header = False

            if line.startswith('>') and not alert_match:
                if not in_quote:
                    in_quote = True
                content = line[1:].strip() if len(line) > 1 else ""
                quote_content.append(content)
                i += 1
                continue
            elif in_quote:
                result.append(self._render_quote('\n'.join(quote_content)))
                quote_content = []
                in_quote = False

            list_match = re.match(r'^(\s*)[-*+]\s+(.+)$', line)
            ordered_match = re.match(r'^(\s*)\d+\.\s+(.+)$', line)

            if list_match or ordered_match:
                if not in_list:
                    in_list = True

                if list_match:
                    indent_level = len(list_match.group(1)) // 2
                    content = list_match.group(2)
                    list_items.append(('ul', indent_level, content))
                else:
                    indent_level = len(ordered_match.group(1)) // 2
                    content = ordered_match.group(2)
                    list_items.append(('ol', indent_level, content))

                list_indent_level = indent_level
                i += 1
                continue
            elif in_list and line.strip() == "":
                result.extend(self._render_list_blocks(list_items) if split_lists
                              else [self._render_list(list_items)])
                list_items = []
                in_list = False
                list_indent_level = 0
                last_was_header = False
                i += 1
                continue
            elif in_list:
                if list_items and line.strip() and not re.match(r'^[-\s]*(#{1,6})\s+', line) \
                and not line.strip().startswith('>') \
                and not re.match(r'^[-*_]{3,}\s*$', line):
                    prev = list_items[-1]
                    list_items[-1] = (
                        prev[0],
                        prev[1],
                        prev[2] + '\n' + line.strip()
                    )
                    i += 1
                    continue

                result.extend(self._render_list_blocks(list_items) if split_lists
                              else [self._render_list(list_items)])
                list_items = []
                in_list = False
                list_indent_level = 0

            header_match = re.match(r'^[-\s]*(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2)
                result.append(f'<h{level}>{self._process_inline(text)}</h{level}>')
                last_was_header = True
                i += 1
                continue

            if re.match(r'^[-*_]{3,}\s*$', line):
                result.append('<hr>')
                last_was_header = False
                i += 1
                continue

            if line.strip() == "":
                if last_was_header:
                    last_was_header = False
                else:
                    result.append("<p></p>")
                i += 1
                continue

            processed = self._process_inline(line)
            # Only a real <img> needs the class; a remote image fell back to a link.
            lone_image = (_IMAGE_ONLY_LINE.match(line.strip()) is not None
                          and processed.lstrip().startswith('<img'))
            result.append(f'<p{" class=\"imgblock\"" if lone_image else ""}>{processed}</p>')
            last_was_header = False
            i += 1

        if in_list:
            result.extend(self._render_list_blocks(list_items) if split_lists
                          else [self._render_list(list_items)])

        if in_alert:
            result.append(self._render_alert(alert_type, '\n'.join(alert_content)))

        if in_quote:
            result.append(self._render_quote('\n'.join(quote_content)))

        return result

    def _process_inline(self, text: str) -> str:
        """Process inline markdown elements."""
        text = html.escape(text)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Images must precede links: the link rule also matches ![alt](src) and
        # would leave a stray "!" plus a link, which is what it used to do.
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', self._replace_image, text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        text = re.sub(r'(?<!href=")(?<!src=")(https?://[^\s<>"]+)', r'<a href="\1">\1</a>', text)

        return text

    def _replace_image(self, match: "re.Match") -> str:
        """An inlinable local image becomes <img>; anything else stays a link."""
        alt, src = match.group(1), match.group(2)
        uri = self._image_data_uri(src)
        if uri is None:
            return f'<a href="{src}">{alt or "[Image]"}</a>'
        return f'<img src="{uri}" alt="{alt}">'

    def _image_data_uri(self, src: str) -> str | None:
        """Local image file to a base64 data URI, or None when it cannot be inlined."""
        if src.startswith("data:"):
            return src
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', src) or self.image_base is None:
            return None
        try:
            path = (self.image_base / src).resolve()
            # Confine to image_base: markdown from a release body is untrusted.
            if not path.is_file() or self.image_base not in path.parents:
                return None
            mime = _IMAGE_MIME.get(path.suffix.lower())
            if mime is None or path.stat().st_size > _MAX_IMAGE_BYTES:
                return None
            raw = path.read_bytes()
        except OSError:
            logger.debug("markdown image could not be read", exc_info=True)
            return None
        if mime != "image/svg+xml":
            rounded = _rounded_png(raw)
            if rounded is not None:
                mime, raw = "image/png", rounded
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    def _render_quote(self, content: str) -> str:
        """Render a standard blockquote with grey left border."""
        processed_content = self._process_inline(content).replace('\n', '<br>')
        return f'''
        <table class="quote" cellspacing="0" cellpadding="0">
            <tr>
                <td class="quote-border"></td>
                <td class="quote-content">{processed_content}</td>
            </tr>
        </table>
        '''

    def _render_alert(self, alert_type: str, content: str) -> str:
        """Render a GitHub-style alert box using table for QLabel compatibility."""
        config = self.ALERT_TYPES.get(alert_type, self.ALERT_TYPES['NOTE'])
        alert_color = config['color']
        alert_title = alert_type.capitalize()

        svg_template = self.SVG_ICONS.get(alert_type, '')
        svg_icon = svg_template.replace('ALERT_COLOR', alert_color)

        svg_encoded = base64.b64encode(svg_icon.encode('utf-8')).decode('utf-8')
        svg_data_uri = f'data:image/svg+xml;base64,{svg_encoded}'

        processed_content = self._process_inline(content)

        return f'''
        <table cellpadding="0" cellspacing="0" class="alert alert-{alert_type.lower()}">
            <tr>
                <td class="alert-border"></td>
                <td class="alert-content">
                    <div class="alert-title">
                        <img src="{svg_data_uri}" width="16" height="16"/>
                        {alert_title}
                    </div>
                    <div class="alert-text">
                        {processed_content}
                    </div>
                </td>
            </tr>
        </table>
        '''

    def _render_code_block(self, code: str, language: str = "") -> str:
        """Render a code block."""
        escaped_code = html.escape(code)
        lang_label = f'<span class="code-lang">{language}</span><br>' if language else ''

        return f'''
        <div class="code-block">
            {lang_label}
            <pre>{escaped_code}</pre>
        </div>
        '''

    def _render_list_blocks(self, items: list) -> list[str]:
        """Split one markdown list into one HTML block per top-level item, each carrying its nested
        children. Ordered lists keep counting across blocks through the <ol start=...> attribute
        (honoured by Qt rich text since QTextListFormat gained a start property in Qt 6)."""
        blocks = []
        group = []
        group_start = 1
        ol_run = 0  # consecutive top-level ordered items so far

        def flush():
            if group:
                blocks.append(self._render_list(group, start=group_start))
                group.clear()

        for tag, indent, content in items:
            if indent == 0 or not group:
                flush()
                if indent == 0 and tag == 'ol':
                    ol_run += 1
                elif indent == 0:
                    ol_run = 0  # a top-level bullet breaks the numbering run
                group_start = max(1, ol_run)
            group.append((tag, indent, content))
        flush()
        return blocks

    def _render_list(self, items: list, start: int = 1) -> str:
        if not items:
            return ""

        html_out = []
        stack = []

        def open_list(tag, indent):
            if tag == 'ul' and indent > 0:
                html_out.append(f'<{tag} class="nested-list">')
            elif tag == 'ol' and indent == 0 and start > 1:
                html_out.append(f'<ol start="{start}">')
            else:
                html_out.append(f'<{tag}>')
            stack.append(tag)

        def close_list():
            tag = stack.pop()
            html_out.append(f'</{tag}>')

        prev_indent = 0

        for tag, indent, content in items:
            if '__ALERT__' in content:
                parts = re.split(r'__ALERT__|__ENDALERT__', content)
                processed_parts = []
                for j, part in enumerate(parts):
                    if j % 2 == 0:
                        # This is regular text content
                        if part.strip():
                            processed = self._process_inline(part)
                            # Don't add <br> at the end if an alert follows
                            processed_parts.append(processed.replace('\n', '<br>').rstrip('<br>'))
                    else:
                        # This is alert HTML - add it directly without extra spacing
                        processed_parts.append(part)
                content = ''.join(processed_parts)
            else:
                content = self._process_inline(content)
                content = content.replace('\n', '<br>')

            while indent < prev_indent:
                close_list()
                prev_indent -= 1

            while indent > prev_indent:
                open_list(tag, indent)
                prev_indent += 1

            if not stack or stack[-1] != tag:
                open_list(tag, indent)

            html_out.append(f'<li>{content}</li>')
            prev_indent = indent

        while stack:
            close_list()

        return "\n".join(html_out)

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for rgba()."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"

    def _wrap_html(self, content: str) -> str:
        """Wrap content in HTML document structure with GitHub-style CSS."""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            {self._style_tag()}
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        '''

    def _style_tag(self, body_padding: str = "8px") -> str:
        """The GitHub-style CSS shared by render() and render_blocks(); each block from
        render_blocks() needs its own copy since every QLabel is         an independent rich-text
        document."""
        t = self.theme
        TEXT_PRIMARY = t.md_text_primary
        TEXT_SECONDARY = t.md_text_secondary
        BG_SECTION_BORDER = t.md_section_border
        FONT_FAMILY = t.md_font_family
        COLOR_NOTE = t.md_color_note
        COLOR_TIP = t.md_color_tip
        COLOR_IMPORTANT = t.md_color_important
        COLOR_WARNING = t.md_color_warning
        COLOR_CAUTION = t.md_color_caution
        COLOR_QUOTE = t.md_color_quote
        return f'''
            <style>
                * {{
                    max-width: 100%;
                    overflow-wrap: break-word;
                    text-overflow: break-word;
                    overflow-wrap: anywhere;
                }}
                body {{
                    color: {TEXT_PRIMARY};
                    font-family: {FONT_FAMILY};
                    font-size: 14px;
                    line-height: 1.6;
                    margin: 0;
                    padding: {body_padding};
                    overflow-wrap: break-word;
                    word-break: break-word;
                }}
                .container {{
                    overflow-wrap: break-word;
                    word-break: break-word;
                    overflow-x: clip;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: {TEXT_PRIMARY};
                    font-weight: 600;
                    line-height: 1.25;
                    margin-top: 16px;
                    margin-bottom: 8px;
                    padding-top: 8px;
                    padding-bottom: 4px;
                }}
                h1 {{
                    font-size: 2em;
                    border-bottom: 1px solid {BG_SECTION_BORDER};
                    padding-bottom: 0.3em;
                }}
                h2 {{
                    font-size: 1.5em;
                    border-bottom: 1px solid {BG_SECTION_BORDER};
                    padding-bottom: 0.3em;
                }}
                h3 {{ font-size: 1.25em; }}
                h4 {{ font-size: 1em; }}
                h5 {{ font-size: 0.875em; }}
                h6 {{
                    font-size: 0.85em;
                    color: {TEXT_SECONDARY};
                }}
                p {{
                    margin: 5px 0;
                }}
                p.imgblock {{
                    line-height: 1;
                    margin: 5px 0;
                }}
                a {{
                    color: {COLOR_NOTE};
                }}
                code {{
                    background-color: {BG_SECTION_BORDER};
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                    font-size: 85%;
                }}
                pre {{
                    margin: 0;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                blockquote {{
                    border-left: 3px solid {BG_SECTION_BORDER};
                    padding-left: 10px;
                    margin: 10px 0;
                    color: {TEXT_SECONDARY};
                }}
                ul, ol {{
                    margin: 6px 0;
                    padding-left: 25px;
                }}
                ul.nested-list {{
                    margin: 6px 0;
                    padding-left: 25px;
                    list-style-type: circle;
                }}
                li {{
                    margin: 5px 0;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid {BG_SECTION_BORDER};
                    margin: 15px 0;
                }}
                .code-block {{
                    background-color: {BG_SECTION_BORDER};
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
                    font-family: monospace;
                    overflow-x: auto;
                }}
                .code-lang {{
                    color: {TEXT_SECONDARY};
                    font-size: 11px;
                }}
                .alert {{
                    margin: 10px 0;
                    border-collapse: collapse;
                }}
                .alert-border {{
                    width: 0;
                    padding-left: 1px;
                    padding-right: 2px;
                }}
                .alert-content {{
                    padding: 10px 15px;
                }}
                .alert-title {{
                    margin-bottom: 5px;
                    display: flex;
                    align-items: center;
                }}
                .alert-title img {{
                    margin-right: 6px;
                    vertical-align: middle;
                }}
                .alert-text {{
                    color: {TEXT_SECONDARY};
                }}
                .alert-note .alert-border {{ background-color: {COLOR_NOTE}; }}
                .alert-note .alert-title {{ color: {COLOR_NOTE}; }}
                .alert-tip .alert-border {{ background-color: {COLOR_TIP}; }}
                .alert-tip .alert-title {{ color: {COLOR_TIP}; }}
                .alert-important .alert-border {{ background-color: {COLOR_IMPORTANT}; }}
                .alert-important .alert-title {{ color: {COLOR_IMPORTANT}; }}
                .alert-warning .alert-border {{ background-color: {COLOR_WARNING}; }}
                .alert-warning .alert-title {{ color: {COLOR_WARNING}; }}
                .alert-caution .alert-border {{ background-color: {COLOR_CAUTION}; }}
                .alert-caution .alert-title {{ color: {COLOR_CAUTION}; }}
                .quote {{
                    margin: 10px 0;
                    border-collapse: collapse;
                }}
                .quote-border {{
                    width: 0;
                    padding-left: 1px;
                    padding-right: 2px;
                    background-color: {COLOR_QUOTE};
                }}
                .quote-content {{
                    padding: 10px 15px;
                    color: {TEXT_SECONDARY};
                }}
            </style>
        '''
