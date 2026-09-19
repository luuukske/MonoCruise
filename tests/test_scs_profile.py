"""Selected ETS2/ATS profile reader: Steam Cloud split, log identity, mappings."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.scs_profile.intensity import (
    DEFAULT_BRAKE_INTENSITY,
    TUNE_BRAKE_INTENSITY,
    BrakeIntensityCache,
    apply_brake_intensity,
    learn_decel_scale,
)
from core.scs_profile.reader import (
    _adaptive_label,
    _brake_slider_pct,
    brake_ui_scale,
    decode_profile_hex,
    parse_controls_constants,
    parse_uset,
    read_selected_profile,
)


def _hex(name: str) -> str:
    return name.encode("utf-8").hex().upper()


def _uset(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# prism3d variable config data"]
    for key, value in values.items():
        lines.append(f'uset {key} "{value}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _controls(path: Path, brake_dz: str = "0.000000") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "SiiNunit\n{\ninput_config : x {\n"
        f' config_lines[22]: "constant c_brake_dz {brake_dz}"\n'
        "}\n}\n",
        encoding="utf-8",
    )


def _log(user_dir: Path, hex_id: str, name: str) -> None:
    (user_dir / "game.log.txt").write_text(
        "\n".join(
            [
                f"00:00:19.343 : Set profile finished: '{name}'",
                f"00:00:19.373 : New profile selected: '{name}'",
                f"00:00:29.166 : Loading save. path: /steam/profiles/{hex_id}/save/autosave/game.sii",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_hex_roundtrip_and_uset_parse():
    assert decode_profile_hex(_hex("DriverA")) == "DriverA"
    assert decode_profile_hex("zz") is None
    parsed = parse_uset('uset g_trans "3"\nuset g_brake_intensity "1.0"\n')
    assert parsed["g_trans"] == "3"
    consts = parse_controls_constants(
        ' config_lines[22]: "constant c_brake_dz 0.150000"\n'
    )
    assert consts["c_brake_dz"] == "0.150000"


def test_game_log_selects_the_older_profile_not_the_newest_folder(tmp_path: Path):
    user = tmp_path / "user"
    active = _hex("DriverA")
    other = _hex("DriverB")
    _uset(
        user / "steam_profiles" / other / "config_local.cfg",
        g_trans="0",
        g_brake_intensity="3",
    )
    _uset(
        user / "steam_profiles" / active / "config_local.cfg",
        g_trans="3",
        g_brake_intensity="1",
    )
    _controls(user / "steam_profiles" / active / "controls.sii", "0.050000")
    _log(user, active, "DriverA")
    # Make the arcade profile look "more recently used" on disk.
    (user / "steam_profiles" / other / "config_local.cfg").write_text(
        (user / "steam_profiles" / other / "config_local.cfg").read_text(encoding="utf-8")
        + 'uset g_input_configured "1"\n',
        encoding="utf-8",
    )

    settings = read_selected_profile(
        "ets2",
        user_dir=user,
        steam_remotes=[],
        live_shifter_type=None,
    )
    assert settings.profile_hex == active
    assert settings.profile_name == "DriverA"
    assert settings.identity_source == "game_log"
    assert settings.g_trans == 3
    assert settings.transmission == "automatic"
    assert settings.g_brake_intensity == 1.0
    assert settings.c_brake_dz == 0.05


def test_steam_cloud_config_fills_adaptive_shift(tmp_path: Path):
    user = tmp_path / "user"
    remote = tmp_path / "remote_profiles"
    hex_id = _hex("DriverA")
    _uset(
        user / "steam_profiles" / hex_id / "config_local.cfg",
        g_trans="3",
        g_brake_intensity="1",
    )
    _uset(remote / hex_id / "config.cfg", g_adaptive_shift="10")
    _log(user, hex_id, "DriverA")

    settings = read_selected_profile(
        "ets2",
        user_dir=user,
        steam_remotes=[remote],
        live_shifter_type=None,
    )
    assert settings.g_adaptive_shift == 10.0
    assert settings.adaptive == "eco"
    assert settings.g_trans == 3
    assert settings.brake_from == "config_local.cfg"


def test_live_shifter_overrides_stale_g_trans(tmp_path: Path):
    user = tmp_path / "user"
    hex_id = _hex("DriverA")
    _uset(
        user / "steam_profiles" / hex_id / "config_local.cfg",
        g_trans="3",
        g_brake_intensity="1",
    )
    _log(user, hex_id, "DriverA")

    settings = read_selected_profile(
        "ets2",
        user_dir=user,
        steam_remotes=[],
        live_shifter_type="arcade",
    )
    assert settings.g_trans == 0
    assert settings.transmission == "arcade"
    assert settings.trans_from == "telemetry"
    assert settings.live_shifter_type == "arcade"


def test_mtime_fallback_when_log_is_missing(tmp_path: Path):
    user = tmp_path / "user"
    older = _hex("DriverA")
    newer = _hex("DriverB")
    _uset(user / "profiles" / older / "config_local.cfg", g_trans="3", g_brake_intensity="1")
    _uset(user / "profiles" / newer / "config_local.cfg", g_trans="1", g_brake_intensity="0.5")

    settings = read_selected_profile(
        "ets2",
        user_dir=user,
        steam_remotes=[],
        live_shifter_type=None,
    )
    assert settings.profile_hex == newer
    assert settings.identity_source == "mtime"
    assert settings.store == "profiles"
    assert settings.g_trans == 1
    assert settings.transmission == "sequential"


def test_mappings_for_adaptive_and_brake_slider():
    assert _adaptive_label(0.0) == "off"
    assert _adaptive_label(1.6667) == "power"
    assert _adaptive_label(3.0) == "normal"
    assert _adaptive_label(10.0) == "eco"
    assert _adaptive_label(7.5) == "custom"
    pct = _brake_slider_pct(1.0)
    assert pct is not None
    assert abs(pct - 100.0) < 0.2
    assert _brake_slider_pct(1.0 / 3.0) == 50.0
    assert _brake_slider_pct(3.0) == 150.0


def test_encrypted_sii_is_ignored(tmp_path: Path):
    user = tmp_path / "user"
    hex_id = _hex("DriverA")
    folder = user / "steam_profiles" / hex_id
    folder.mkdir(parents=True)
    (folder / "config_local.cfg").write_bytes(b"ScsC" + b"\x00" * 20)
    _uset(folder / "config.cfg", g_trans="2", g_brake_intensity="1")
    _log(user, hex_id, "DriverA")

    settings = read_selected_profile(
        "ets2",
        user_dir=user,
        steam_remotes=[],
        live_shifter_type=None,
    )
    assert settings.g_trans == 2
    assert settings.transmission == "h-shifter"


def test_apply_at_100pct_scales_by_tune_over_cvar():
    p = 0.4
    sent = apply_brake_intensity(p, 1.0)
    assert sent == pytest.approx(p * TUNE_BRAKE_INTENSITY)


def test_apply_is_identity_at_tune_cvar():
    assert apply_brake_intensity(0.37, TUNE_BRAKE_INTENSITY) == pytest.approx(0.37)


def test_apply_inverts_the_raw_cvar_multiply():
    p = 0.5
    assert apply_brake_intensity(p, 3.0) == pytest.approx(p * TUNE_BRAKE_INTENSITY / 3.0)
    assert apply_brake_intensity(p, 1.0 / 3.0) == pytest.approx(1.0)
    assert apply_brake_intensity(1.0, 3.0) == pytest.approx(TUNE_BRAKE_INTENSITY / 3.0)


def test_apply_none_behaves_as_default_100pct():
    p = 0.4
    assert apply_brake_intensity(p, None) == pytest.approx(
        apply_brake_intensity(p, DEFAULT_BRAKE_INTENSITY)
    )


def test_apply_keeps_endpoints():
    assert apply_brake_intensity(0.0, 3.0) == 0.0
    assert apply_brake_intensity(1.0, 1.0 / 3.0) == 1.0


def test_lower_intensity_sends_more_mid_pedal():
    mid = 0.5
    assert apply_brake_intensity(mid, 1.0) > mid
    assert apply_brake_intensity(mid, 2.0) < mid


def test_learn_decel_scale_maps_cvar_to_tune_units():
    assert learn_decel_scale(TUNE_BRAKE_INTENSITY) == pytest.approx(1.0)
    assert learn_decel_scale(1.0) == pytest.approx(TUNE_BRAKE_INTENSITY)
    assert learn_decel_scale(3.0) == pytest.approx(TUNE_BRAKE_INTENSITY / 3.0)
    assert learn_decel_scale(None) == pytest.approx(TUNE_BRAKE_INTENSITY)
    assert brake_ui_scale(1.0 / 3.0) == pytest.approx(0.5)
    assert brake_ui_scale(3.0) == pytest.approx(1.5)


def test_brake_intensity_cache_rate_limits(monkeypatch):
    import core.scs_profile.intensity as mod

    calls = {"n": 0}

    class _S:
        g_brake_intensity = 2.0

    def fake_read(game, live_shifter_type=None):
        calls["n"] += 1
        return _S()

    monkeypatch.setattr(mod, "read_selected_profile", fake_read)
    clk = {"t": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clk["t"])
    cache = BrakeIntensityCache()
    assert cache.get("ets2") == pytest.approx(2.0)
    assert calls["n"] == 1
    clk["t"] = 1.5
    assert cache.get("ets2") == pytest.approx(2.0)
    assert calls["n"] == 1
    clk["t"] = 2.1
    assert cache.get("ets2") == pytest.approx(2.0)
    assert calls["n"] == 2


def test_cache_retries_on_a_timer_when_unread(monkeypatch):
    import core.scs_profile.intensity as mod

    calls = {"n": 0}

    class _S:
        g_brake_intensity = None

    def fake_read(game, live_shifter_type=None):
        calls["n"] += 1
        return _S()

    monkeypatch.setattr(mod, "read_selected_profile", fake_read)
    clk = {"t": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clk["t"])
    cache = BrakeIntensityCache()
    assert cache.get("ets2") == pytest.approx(DEFAULT_BRAKE_INTENSITY)
    assert calls["n"] == 1
    clk["t"] = 0.5
    cache.get("ets2")
    assert calls["n"] == 1
