"""
Offline GitHub-flavored Markdown to HTML converter for PyQt6.
Supports GitHub alerts (NOTE, TIP, IMPORTANT, WARNING, CAUTION),
code blocks, links, images, lists, and basic formatting.
"""

import re
import html
from styles import (
    COLOR_NOTE, COLOR_TIP, COLOR_IMPORTANT, COLOR_WARNING, COLOR_CAUTION,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_SECTION_BORDER, FONT_FAMILY, COLOR_QUOTE
)


class GitHubMarkdownRenderer:
    """Converts GitHub-flavored Markdown to HTML for QTextBrowser."""
    
    SVG_ICONS = {
        'NOTE': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-info-icon lucide-info"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>''',
        'TIP': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-lightbulb-icon lucide-lightbulb"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>''',
        'IMPORTANT': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-message-square-warning-icon lucide-message-square-warning"><path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/><path d="M12 15h.01"/><path d="M12 7v4"/></svg>''',
        'WARNING': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert-icon lucide-triangle-alert"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>''',
        'CAUTION': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-octagon-alert-icon lucide-octagon-alert"><path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/></svg>''',
    }
    
    ALERT_TYPES = {
        'NOTE': {'color': COLOR_NOTE, 'icon': 'ℹ️'},
        'TIP': {'color': COLOR_TIP, 'icon': '💡'},
        'IMPORTANT': {'color': COLOR_IMPORTANT, 'icon': '❗'},
        'WARNING': {'color': COLOR_WARNING, 'icon': '⚠️'},
        'CAUTION': {'color': COLOR_CAUTION, 'icon': '🛑'},
    }
    
    def __init__(self):
        self.video_urls = []
    
    def render(self, markdown_text: str) -> str:
        """Convert markdown to HTML and extract video URLs."""
        self.video_urls = []
        
        if not markdown_text:
            return ""
        
        self._extract_video_urls(markdown_text)
        html_content = self._process_markdown(markdown_text)
        
        return self._wrap_html(html_content)
    
    def get_first_video_url(self) -> str | None:
        """Return the first .mp4 URL found, or None."""
        return self.video_urls[0] if self.video_urls else None
    
    def _extract_video_urls(self, text: str):
        """Find all .mp4 URLs in the text."""
        mp4_pattern = r'https?://[^\s<>\[\]()]+\.mp4(?:\?[^\s<>\[\]()]*)?'
        self.video_urls = re.findall(mp4_pattern, text, re.IGNORECASE)
    
    def _process_markdown(self, text: str) -> str:
        """Process markdown text to HTML."""
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
                result.append(self._render_list(list_items))
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

                result.append(self._render_list(list_items))
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
            result.append(f'<p>{processed}</p>')
            last_was_header = False
            i += 1
        
        if in_list:
            result.append(self._render_list(list_items))
        
        if in_alert:
            result.append(self._render_alert(alert_type, '\n'.join(alert_content)))
        
        if in_quote:
            result.append(self._render_quote('\n'.join(quote_content)))
        
        return '\n'.join(result)
    
    def _process_inline(self, text: str) -> str:
        """Process inline markdown elements."""
        text = html.escape(text)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<a href="\2">[Image: \1]</a>', text)
        text = re.sub(r'(?<!href=")(https?://[^\s<>"]+)', r'<a href="\1">\1</a>', text)
        
        return text
    
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
        
        import base64
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
    
    def _render_list(self, items: list) -> str:
        if not items:
            return ""

        html_out = []
        stack = []

        def open_list(tag, indent):
            if tag == 'ul' and indent > 0:
                html_out.append(f'<{tag} class="nested-list">')
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
                    padding: 8px;
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
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        '''