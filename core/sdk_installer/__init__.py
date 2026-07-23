"""Public SDK installer API. Backend only; see README.md."""

from .game_paths import (
    GAME_TYPES,
    close_game,
    find_game_installations,
    get_plugins_dir,
    is_game_running,
    is_steam_installed,
)
from .manager import (
    COURTESY_FILES,
    DLL_FILES,
    FORCE_REFETCH,
    SUPPORTED_GAME_VERSION,
    GameApplyResult,
    GameSdkState,
    ManagedFileState,
    SdkCheckResult,
    SdkManager,
    check_sdk,
    get_manager,
    start_boot_check,
    start_reinstall,
)
from .remote import SdkSource, SdkSourceError, SdkVersionUnsupported

__all__ = [
    "GAME_TYPES",
    "close_game",
    "find_game_installations",
    "get_plugins_dir",
    "is_game_running",
    "is_steam_installed",
    "COURTESY_FILES",
    "DLL_FILES",
    "FORCE_REFETCH",
    "SUPPORTED_GAME_VERSION",
    "GameApplyResult",
    "GameSdkState",
    "ManagedFileState",
    "SdkCheckResult",
    "SdkManager",
    "check_sdk",
    "get_manager",
    "start_boot_check",
    "start_reinstall",
    "SdkSource",
    "SdkSourceError",
    "SdkVersionUnsupported",
]
