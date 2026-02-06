"""
Offline GitHub-flavored Markdown to HTML converter for PyQt6.
Supports GitHub alerts (NOTE, TIP, IMPORTANT, WARNING, CAUTION),
code blocks, links, images, lists, and basic formatting.
"""

import re
import html
from styles import (
    COLOR_NOTE, COLOR_TIP, COLOR_IMPORTANT, COLOR_WARNING, COLOR_CAUTION,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_SECTION_BORDER
)


class GitHubMarkdownRenderer:
    """Converts GitHub-flavored Markdown to HTML for QTextBrowser."""
    
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
                result.append(f'<h{level} style="color: {TEXT_PRIMARY}; margin: 10px 0;">{content}</h{level}>')
                i += 1
                continue
            
            # Handle horizontal rules
            if re.match(r'^[-*_]{3,}\s*$', line):
                result.append(f'<hr style="border: none; border-top: 1px solid {BG_SECTION_BORDER}; margin: 15px 0;">')
                i += 1
                continue
            
            # Handle blockquotes (non-alert)
            if line.startswith('>') and not in_alert:
                content = line[1:].strip() if len(line) > 1 else ""
                content = self._process_inline(content)
                result.append(f'<blockquote style="border-left: 3px solid {BG_SECTION_BORDER}; padding-left: 10px; margin: 10px 0; color: {TEXT_SECONDARY};">{content}</blockquote>')
                i += 1
                continue
            
            # Handle empty lines
            if line.strip() == "":
                result.append('<br>')
                i += 1
                continue
            
            # Regular paragraph
            content = self._process_inline(line)
            result.append(f'<p style="margin: 5px 0;">{content}</p>')
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
        """Render a GitHub-style alert box."""
        config = self.ALERT_TYPES.get(alert_type, self.ALERT_TYPES['NOTE'])
        color = config['color']
        icon = config['icon']
        
        # Process content
        processed_content = self._process_inline(content)
        
        return f'''
        <div style="
            border-left: 4px solid {color};
            background-color: rgba({self._hex_to_rgb(color)}, 0.1);
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        ">
            <div style="color: {color}; font-weight: bold; margin-bottom: 5px;">
                {icon} {alert_type}
            </div>
            <div style="color: {TEXT_PRIMARY};">
                {processed_content}
            </div>
        </div>
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
        """Wrap content in HTML document structure."""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    color: {TEXT_PRIMARY};
                    font-family: Inter, Sans-serif;
                    font-size: 14px;
                    line-height: 1.5;
                    margin: 0;
                    padding: 5px;
                }}
                a {{
                    color: {COLOR_NOTE};
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        '''