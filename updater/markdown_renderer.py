"""
Offline GitHub-flavored Markdown to HTML converter for PyQt6.
Supports GitHub alerts (NOTE, TIP, IMPORTANT, WARNING, CAUTION),
code blocks, links, images, lists, and basic formatting.
"""

import re
import html
from styles import (
    COLOR_NOTE, COLOR_TIP, COLOR_IMPORTANT, COLOR_WARNING, COLOR_CAUTION,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_SECTION_BORDER, FONT_FAMILY, BG_SECTION
)


class GitHubMarkdownRenderer:
    """Converts GitHub-flavored Markdown to HTML for QTextBrowser."""
    
    # SVG icon templates (color will be replaced dynamically)
    SVG_ICONS = {
        'NOTE': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-info-icon lucide-info"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>''',
        'TIP': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-lightbulb-icon lucide-lightbulb"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>''',
        'IMPORTANT': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-message-square-warning-icon lucide-message-square-warning"><path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/><path d="M12 15h.01"/><path d="M12 7v4"/></svg>''',
        'WARNING': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert-icon lucide-triangle-alert"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>''',
        'CAUTION': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="ALERT_COLOR" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-octagon-alert-icon lucide-octagon-alert"><path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/></svg>''',
    }
    
    # Alert type configurations
    ALERT_TYPES = {
        'NOTE': {'color': COLOR_NOTE, 'icon': 'ℹ️'},
        'TIP': {'color': COLOR_TIP, 'icon': '💡'},
        'IMPORTANT': {'color': COLOR_IMPORTANT, 'icon': '❗'},
        'WARNING': {'color': COLOR_WARNING, 'icon': '⚠️'},
        'CAUTION': {'color': COLOR_CAUTION, 'icon': '🛑'},
    }
    
    def __init__(self):
        self.video_urls = []  # Store found video URLs
    
    def render(self, markdown_text: str) -> str:
        """Convert markdown to HTML and extract video URLs."""
        self.video_urls = []
        
        if not markdown_text:
            return ""
        
        # Extract video URLs before processing
        self._extract_video_urls(markdown_text)
        
        # Process the markdown
        html_content = self._process_markdown(markdown_text)
        
        # Wrap in styled container
        return self._wrap_html(html_content)
    
    def get_first_video_url(self) -> str | None:
        """Return the first .mp4 URL found, or None."""
        return self.video_urls[0] if self.video_urls else None
    
    def _extract_video_urls(self, text: str):
        """Find all .mp4 URLs in the text."""
        # Match URLs ending in .mp4 (with optional query params)
        mp4_pattern = r'https?://[^\s<>\[\]()]+\.mp4(?:\?[^\s<>\[\]()]*)?'
        self.video_urls = re.findall(mp4_pattern, text, re.IGNORECASE)
    
    def _process_markdown(self, text: str) -> str:
        """Process markdown text to HTML."""
        lines = text.split('\n')
        result = []
        i = 0
        in_code_block = False
        code_block_content = []
        code_language = ""
        in_list = False
        list_items = []
        in_alert = False
        alert_type = ""
        alert_content = []
        last_was_header = False  # Track if last element was a header
        
        while i < len(lines):
            line = lines[i]
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End code block
                    result.append(self._render_code_block('\n'.join(code_block_content), code_language))
                    code_block_content = []
                    code_language = ""
                    in_code_block = False
                    last_was_header = False
                else:
                    # Start code block
                    in_code_block = True
                    code_language = line.strip()[3:].strip()
                i += 1
                continue
            
            if in_code_block:
                code_block_content.append(line)
                i += 1
                continue
            
            # Handle GitHub alerts (>[!TYPE])
            alert_match = re.match(r'^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', line)
            if alert_match:
                # Close any open list
                if in_list:
                    result.append(self._render_list(list_items))
                    list_items = []
                    in_list = False
                    last_was_header = False
                
                alert_type = alert_match.group(1)
                in_alert = True
                alert_content = []
                i += 1
                continue
            
            # Continue alert content
            if in_alert:
                if line.startswith('>'):
                    # Strip the > prefix and add to alert
                    content = line[1:].strip() if len(line) > 1 else ""
                    alert_content.append(content)
                    i += 1
                    continue
                else:
                    # End of alert
                    result.append(self._render_alert(alert_type, '\n'.join(alert_content)))
                    in_alert = False
                    alert_type = ""
                    alert_content = []
                    last_was_header = False
            
            # Handle unordered lists
            list_match = re.match(r'^(\s*)[-*+]\s+(.+)$', line)
            if list_match:
                if not in_list:
                    in_list = True
                list_items.append(list_match.group(2))
                i += 1
                continue
            elif in_list and line.strip() == "":
                # Empty line might continue list
                i += 1
                continue
            elif in_list:
                # End of list
                result.append(self._render_list(list_items))
                list_items = []
                in_list = False
                last_was_header = False
            
            # Handle ordered lists
            ordered_match = re.match(r'^(\s*)\d+\.\s+(.+)$', line)
            if ordered_match:
                if not in_list:
                    in_list = True
                list_items.append(('ol', ordered_match.group(2)))
                i += 1
                continue
            
            # Handle headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                content = self._process_inline(header_match.group(2))
                result.append(f'<h{level}>{content}</h{level}>')
                last_was_header = True
                i += 1
                continue
            
            # Handle horizontal rules
            if re.match(r'^[-*_]{3,}\s*$', line):
                result.append(f'<hr style="border: none; border-top: 1px solid {BG_SECTION_BORDER}; margin: 15px 0;">')
                last_was_header = False
                i += 1
                continue
            
            # Handle blockquotes (non-alert)
            if line.startswith('>') and not in_alert:
                content = line[1:].strip() if len(line) > 1 else ""
                content = self._process_inline(content)
                result.append(f'<blockquote style="border-left: 3px solid {BG_SECTION_BORDER}; padding-left: 10px; margin: 10px 0; color: {TEXT_SECONDARY};">{content}</blockquote>')
                last_was_header = False
                i += 1
                continue
            
            # Handle empty lines
            if line.strip() == "":
                # Skip all breakpoints - GitHub markdown doesn't add extra spacing from empty lines
                last_was_header = False
                i += 1
                continue
            
            # Regular paragraph
            content = self._process_inline(line)
            result.append(f'<p style="margin: 5px 0;">{content}</p>')
            last_was_header = False
            i += 1
        
        # Close any remaining open elements
        if in_code_block:
            result.append(self._render_code_block('\n'.join(code_block_content), code_language))
        if in_list:
            result.append(self._render_list(list_items))
        if in_alert:
            result.append(self._render_alert(alert_type, '\n'.join(alert_content)))
        
        return '\n'.join(result)
    
    def _process_inline(self, text: str) -> str:
        """Process inline markdown elements."""
        # Escape HTML first
        text = html.escape(text)
        
        # Bold + Italic (must come before individual)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'___(.+?)___', r'<b><i>\1</i></b>', text)
        
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
        
        # Strikethrough
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        
        # Inline code
        text = re.sub(r'`([^`]+)`', rf'<code style="background-color: {BG_SECTION_BORDER}; padding: 2px 5px; border-radius: 3px;">\1</code>', text)
        
        # Links [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', rf'<a href="\2" style="color: {COLOR_NOTE};">\1</a>', text)
        
        # Images ![alt](url) - render as linked text since QTextBrowser has limited image support
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', rf'<a href="\2" style="color: {COLOR_NOTE};">[Image: \1]</a>', text)
        
        # Auto-link URLs (not already in links)
        text = re.sub(r'(?<!href=")(https?://[^\s<>"]+)', rf'<a href="\1" style="color: {COLOR_NOTE};">\1</a>', text)
        
        return text
    
    def _render_alert(self, alert_type: str, content: str) -> str:
        """Render a GitHub-style alert box using table for QLabel compatibility."""
        config = self.ALERT_TYPES.get(alert_type, self.ALERT_TYPES['NOTE'])
        alert_color = config['color']
        alert_title = alert_type.capitalize()
        
        # Get SVG icon and replace color placeholder
        svg_template = self.SVG_ICONS.get(alert_type, '')
        svg_icon = svg_template.replace('ALERT_COLOR', alert_color)
        
        # Encode SVG for inline data URI
        import base64
        svg_encoded = base64.b64encode(svg_icon.encode('utf-8')).decode('utf-8')
        svg_data_uri = f'data:image/svg+xml;base64,{svg_encoded}'
        
        processed_content = self._process_inline(content)
        
        return f'''
        <table cellpadding="0" cellspacing="0" style="margin: 10px 0; border-collapse: collapse;">
            <tr>
                <td style="width: 0; background-color: {alert_color}; padding-left: 1px; padding-right: 2px;"></td>
                <td style="padding: 10px 15px;">
                    <div style="color: {alert_color}; margin-bottom: 5px; display: flex; align-items: center;">
                        <img src="{svg_data_uri}" style="margin-right: 6px; vertical-align: middle;" width="16" height="16"/>
                        {alert_title}
                    </div>
                    <div style="color: {TEXT_SECONDARY};">
                        {processed_content}
                    </div>
                </td>
            </tr>
        </table>
        '''
    
    def _render_code_block(self, code: str, language: str = "") -> str:
        """Render a code block."""
        escaped_code = html.escape(code)
        lang_label = f'<span style="color: {TEXT_SECONDARY}; font-size: 11px;">{language}</span><br>' if language else ''
        
        return f'''
        <div style="
            background-color: {BG_SECTION_BORDER};
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            font-family: monospace;
            overflow-x: auto;
        ">
            {lang_label}
            <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{escaped_code}</pre>
        </div>
        '''
    
    def _render_list(self, items: list) -> str:
        """Render a list (ordered or unordered)."""
        if not items:
            return ""
        
        # Check if ordered list
        is_ordered = isinstance(items[0], tuple) and items[0][0] == 'ol'
        
        tag = 'ol' if is_ordered else 'ul'
        list_items = []
        
        for item in items:
            content = item[1] if isinstance(item, tuple) else item
            content = self._process_inline(content)
            list_items.append(f'<li style="margin: 3px 0;">{content}</li>')
        
        return f'<{tag} style="margin: 10px 0; padding-left: 25px;">{" ".join(list_items)}</{tag}>'
    
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
                body {{
                    color: {TEXT_PRIMARY};
                    font-family: {FONT_FAMILY};
                    font-size: 14px;
                    line-height: 1.6;
                    margin: 0;
                    padding: 8px;
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
                h1 {{ font-size: 2em; border-bottom: 1px solid {BG_SECTION_BORDER}; padding-bottom: 0.3em; }}
                h2 {{ font-size: 1.5em; border-bottom: 1px solid {BG_SECTION_BORDER}; padding-bottom: 0.3em; }}
                h3 {{ font-size: 1.25em; }}
                h4 {{ font-size: 1em; }}
                h5 {{ font-size: 0.875em; }}
                h6 {{ font-size: 0.85em; color: {TEXT_SECONDARY}; }}
                p {{
                    margin-top: 0;
                    margin-bottom: 10px;
                }}
                a {{
                    color: {COLOR_NOTE};
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                code {{
                    background-color: {BG_SECTION_BORDER};
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                    font-size: 85%;
                }}
                pre {{
                    background-color: {BG_SECTION};
                    border: 1px solid {BG_SECTION_BORDER};
                    border-radius: 6px;
                    padding: 16px;
                    overflow: auto;
                    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                    font-size: 85%;
                    line-height: 1.45;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                    border-radius: 0;
                }}
                blockquote {{
                    border-left: 3px solid {BG_SECTION_BORDER};
                    padding-left: 16px;
                    margin: 0;
                    color: {TEXT_SECONDARY};
                }}
                ul, ol {{
                    margin-top: 0;
                    margin-bottom: 16px;
                    padding-left: 1em;
                }}
                li {{
                    margin-top: 0.25em;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid {BG_SECTION_BORDER};
                    margin: 24px 0;
                }}
                table {{
                    border-collapse: collapse;
                    border-spacing: 0;
                    width: 100%;
                    margin: 16px 0;
                }}
                table th, table td {{
                    border: 1px solid {BG_SECTION_BORDER};
                    padding: 6px 13px;
                }}
                table th {{
                    background-color: {BG_SECTION};
                    font-weight: 600;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        '''