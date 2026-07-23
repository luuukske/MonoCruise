import html as _html
import requests
import re
from typing import Optional

_REQUEST_TIMEOUT = 15  # seconds


class GitHubAPIError(Exception):
    """GitHub could not be reached or returned an error response."""


class GitHubAPI:
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
        self._releases_cache: list[dict] | None = None
    
    def get_branches(self) -> list[dict]:
        """Fetch all branches from the repository."""
        response = requests.get(f"{self.base_url}/branches", timeout=_REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return []
    
    def get_releases(self) -> list[dict]:
        """Fetch all releases including pre-releases; raises GitHubAPIError when unreachable."""
        if self._releases_cache is not None:
            return self._releases_cache
        try:
            response = requests.get(f"{self.base_url}/releases", timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as e:
            raise GitHubAPIError(f"Cannot reach GitHub: {e}") from e
        if response.status_code != 200:
            raise GitHubAPIError(f"GitHub returned HTTP {response.status_code}")
        self._releases_cache = response.json()
        return self._releases_cache
    
    def invalidate_cache(self):
        """Clear cached releases so the next call fetches fresh data."""
        self._releases_cache = None
    
    def get_releases_for_branch(self, branch: str) -> list[dict]:
        """Filter releases by target branch (target_commitish)."""
        releases = self.get_releases()
        # Filter by branch - releases have target_commitish field
        return [r for r in releases if r.get('target_commitish') == branch]
    
    def get_latest_release_for_branch(self, branch: str) -> Optional[dict]:
        """Get the latest non-prerelease for a branch."""
        releases = self.get_releases_for_branch(branch)
        for release in releases:
            if not release.get('prerelease', False):
                return release
        # If no stable release, return latest prerelease
        return releases[0] if releases else None

    def get_releases_for_channel(self, channel: str) -> list[dict]:
        """Channel-filtered releases (preview=prerelease only); ignores target_commitish."""
        releases = self.get_releases()
        if channel == "preview":
            return [r for r in releases if r.get('prerelease', False)]
        return [r for r in releases if not r.get('prerelease', False)]

    def get_latest_release_for_channel(self, channel: str) -> Optional[dict]:
        """Newest release available on the given channel, or None."""
        releases = self.get_releases_for_channel(channel)
        return releases[0] if releases else None
    
    def get_release_asset_url(self, release: dict, asset_name_contains: str = "Update") -> Optional[str]:
        """Get download URL for an asset in a release."""
        for asset in release.get('assets', []):
            if asset_name_contains in asset['name']:
                return asset['browser_download_url']
        return None
    
    def download_asset(self, url: str, dest: str, progress_callback=None):
        """Download an asset with progress reporting."""
        response = requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(downloaded / total)
        
        if total > 0 and downloaded != total:
            raise IOError(f"Download incomplete: expected {total} bytes, got {downloaded}")
    
    def render_markdown(self, text: str) -> str:
        """Render markdown to HTML using GitHub's API."""
        url = "https://api.github.com/markdown"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "mode": "gfm",  # GitHub Flavored Markdown
            "context": f"{self.owner}/{self.repo}"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error rendering markdown: {e}")
        
        # Fallback: return plain text wrapped in <pre> (escaped to prevent XSS)
        return f"<pre>{_html.escape(text)}</pre>"
    
    @staticmethod
    def extract_first_mp4_url(text: str) -> Optional[str]:
        """Extract the first .mp4 URL from markdown text."""
        # Match URLs ending with .mp4 (with optional query params)
        # Handles both direct URLs and markdown links
        patterns = [
            r'https?://[^\s\)\]"\'<>]+\.mp4(?:\?[^\s\)\]"\'<>]*)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None