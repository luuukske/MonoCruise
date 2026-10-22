"""Fault-injection run of the shipped monocruise_tmp.dll against a fake TruckersMP client.

tmp_plugin/tests/fault_host.cpp is built here with MSVC and loads the DLL that
MonoCruise installs (core/sdk_installer/own/). Each scenario runs in its own
process: it survives or the exit code says how it died. The host redirects the
plugin's shared memory to private names, so a running game is never touched.
See "Fault-injection harness" in tmp_plugin/README.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_msvc

_REPO = Path(__file__).resolve().parents[1]
_SOURCE = _REPO / "tmp_plugin" / "tests" / "fault_host.cpp"
_SDK_INCLUDE = _REPO / "tmp_plugin" / "vendor" / "truckersmp_sdk" / "include"
_DLL = _REPO / "core" / "sdk_installer" / "own" / "monocruise_tmp.dll"

_FAULT_APIS = (
    "Core.ClientVersion",
    "Core.AddListener",
    "Core.LogMessage",
    "Core.RemoveListener",
    "Network.IsConnected",
    "Players.GetLocalPlayer",
    "Players.GetAllPlayers",
    "PlayerHandle.GetVehicle",
    "PlayerHandle.GetDistanceFromLocalPlayer",
    "Vehicle.GetTrailer",
    "Vehicle.GetPlacement",
    "Vehicle.GetLinearVelocity",
    "Vehicle.GetAngularVelocity",
    "Trailer.GetPlacement",
    "Trailer.GetLinearVelocity",
    "Trailer.GetAngularVelocity",
)

SCENARIOS = tuple(sorted((
    "healthy",
    "old_client",
    "older_major_client",
    "unparseable_client",
    "suffixed_client",
    "minimum_client",
    "newer_client",
    "short_newer_client",
    "version_unavailable",
    "version_junk",
    "core_missing",
    "list_fails",
    "queries_fail",
    "list_junk_huge",
    "list_null_items",
    "null_payloads",
    "bad_payload",
    "bad_payload_other_thread",
    "after_shutdown",
    "reinit",
    "unload_without_shutdown",
    "exit_without_shutdown",
    "slow_host",
    "update_race",
    *(f"fault.{api}.{act}" for api in _FAULT_APIS for act in ("crash", "throw")),
)))


def _vcvars64() -> Path | None:
    if sys.platform != "win32":
        return None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    found = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    vcvars = Path(found) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if found else None
    return vcvars if vcvars is not None and vcvars.is_file() else None


@pytest.fixture(scope="module")
def host_exe(tmp_path_factory) -> Path:
    vcvars = _vcvars64()
    if vcvars is None:
        pytest.skip("MSVC (vcvars64.bat) not found")
    out = tmp_path_factory.mktemp("faulthost")
    script = out / "build.bat"
    script.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1 || exit /b 1\r\n'
        f'cl /nologo /EHsc /MT /O1 /W3 /I "{_SDK_INCLUDE}" "{_SOURCE}" /Fo"{out}\\\\" /Fe"{out}\\fault_host.exe"\r\n',
        encoding="ascii",
    )
    built = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True, timeout=300)
    exe = out / "fault_host.exe"
    assert built.returncode == 0 and exe.is_file(), f"fault host did not build:\n{built.stdout}\n{built.stderr}"
    return exe


def test_the_shipped_dll_is_present():
    assert _DLL.is_file()


def test_every_host_scenario_is_listed_here(host_exe):
    listed = subprocess.run([str(host_exe), "--list"], capture_output=True, text=True, timeout=60)
    assert tuple(listed.stdout.split()) == SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario(host_exe, scenario):
    run = subprocess.run([str(host_exe), str(_DLL), scenario], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, f"{scenario} exited {run.returncode}:\n{run.stdout}{run.stderr}"
