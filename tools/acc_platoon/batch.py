"""Independent convoy runs in parallel worker processes. See tools/acc_platoon/README.md."""
from __future__ import annotations

import atexit
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .sim import Run, Scenario, run

# Each worker imports the whole stack; more than this starved a 32-core desktop of memory.
MAX_WORKERS: int = 8
_SANDBOX: Path | None = None


def _sandbox_settings() -> None:
    """Workers never load or save the live config: point core.settings at a throwaway copy."""
    global _SANDBOX
    import core.settings as settings_mod

    _SANDBOX = Path(tempfile.mkdtemp(prefix="monocruise-platoon-"))
    atexit.register(shutil.rmtree, str(_SANDBOX), True)
    real = Path(settings_mod.CONFIG_PATH)
    settings_mod.CONFIG_PATH = _SANDBOX / "config.json"
    settings_mod.BACKUP_PATH = settings_mod.CONFIG_PATH.with_suffix(".json.bak")
    if real.is_file():
        shutil.copy2(str(real), str(settings_mod.CONFIG_PATH))


def run_many(scenarios: list[Scenario], workers: int | None = None) -> list[Run]:
    """Run every scenario, in parallel where the machine allows; results keep input order."""
    workers = min(len(scenarios), workers or min(MAX_WORKERS, os.cpu_count() or 1))
    if workers <= 1:
        return [run(sc) for sc in scenarios]
    # Spawn, never fork: forking a test process that already runs Qt or other threads can deadlock.
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_sandbox_settings) as pool:
            return list(pool.map(run, scenarios))
    except (BrokenProcessPool, OSError):
        return [run(sc) for sc in scenarios]
