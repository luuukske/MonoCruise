"""Backend for installing and updating the ETS2 / ATS SDK DLLs.

Public entry points:

* :func:`start_boot_check` - run the detection off the boot thread and hand a
  :class:`SdkCheckResult` to a callback (used at startup).
* :func:`check_sdk` - synchronous one-shot detection.
* :class:`SdkManager` - drives detection and installation; call
  :meth:`SdkManager.apply` once the user has agreed to install/update.

Everything here is backend only and shows no UI; a front-end reads the result
types and decides what to display.
"""

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
)
from .remote import SdkSource, SdkSourceError

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
    "SdkSource",
    "SdkSourceError",
]
