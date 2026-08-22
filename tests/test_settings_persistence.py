"""Settings load/save: round-trip, dirty check, backup rotation, corruption recovery.

Every test runs against a tmp config path and restores the singleton afterwards,
so nothing here can reach a real config.json.
"""
from __future__ import annotations

import json

import pytest

import core.settings as settings_mod
from core.settings import Settings
from core.thread_management.registry import registry


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Point Settings at a tmp config pair and restore the singleton after."""
    primary = tmp_path / "config.json"
    backup = tmp_path / "config.json.bak"
    monkeypatch.setattr(settings_mod, "CONFIG_PATH", primary)
    monkeypatch.setattr(settings_mod, "BACKUP_PATH", backup)

    instance = Settings.instance()
    saved_fields = instance._public_fields()
    saved_state = dict(instance._saved_state)
    yield primary, backup
    for key, value in saved_fields.items():
        setattr(instance, key, value)
    instance._saved_state = saved_state


def _write(path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_save_then_load_round_trips(cfg):
    primary, _ = cfg
    Settings.save(values={"last_game": 2, "global_speed_limit_kmh": 85.0})
    assert primary.is_file()

    Settings.instance().last_game = 1
    Settings.load()
    assert Settings.last_game == 2
    assert Settings.global_speed_limit_kmh == 85.0


def test_save_is_a_no_op_when_nothing_changed(cfg):
    primary, _ = cfg
    Settings.save(values={"last_game": 2})
    first = primary.stat().st_mtime_ns

    Settings.save()
    assert primary.stat().st_mtime_ns == first, "rewrote an unchanged config"

    Settings.save(force=True)
    assert primary.stat().st_mtime_ns != first


def test_backup_holds_the_previous_good_config(cfg):
    primary, backup = cfg
    Settings.save(values={"last_game": 1}, force=True)
    Settings.save(values={"last_game": 2}, force=True)

    assert json.loads(backup.read_text(encoding="utf-8"))["last_game"] == 1
    assert json.loads(primary.read_text(encoding="utf-8"))["last_game"] == 2


def test_a_corrupt_primary_never_overwrites_a_good_backup(cfg):
    """The backup is the last line of defence; a broken primary must not eat it."""
    primary, backup = cfg
    Settings.save(values={"last_game": 2}, force=True)
    Settings.save(values={"last_game": 1}, force=True)
    assert json.loads(backup.read_text(encoding="utf-8"))["last_game"] == 2

    primary.write_text("{ this is not json", encoding="utf-8")
    Settings.save(values={"last_game": 1}, force=True)
    assert json.loads(backup.read_text(encoding="utf-8"))["last_game"] == 2


def test_load_recovers_from_the_backup_and_quarantines_the_corrupt_primary(cfg, tmp_path):
    primary, backup = cfg
    _write(backup, {"last_game": 2})
    primary.write_text("{ broken", encoding="utf-8")

    Settings.load()
    assert Settings.last_game == 2
    quarantined = list(tmp_path.glob("config.json.corrupt-*"))
    assert quarantined, "corrupt config was deleted instead of quarantined"


def test_load_recovers_when_the_primary_is_missing(cfg):
    primary, backup = cfg
    _write(backup, {"last_game": 2})

    Settings.load()
    assert Settings.last_game == 2
    assert primary.is_file(), "recovered settings were not persisted back"


def test_load_falls_back_to_defaults_when_both_files_are_unreadable(cfg):
    primary, backup = cfg
    primary.write_text("{ broken", encoding="utf-8")
    backup.write_text("also broken", encoding="utf-8")

    Settings.load()
    field = Settings.instance().__dataclass_fields__["last_game"]
    assert Settings.last_game == Settings._dataclass_field_default(field)


def test_an_incomplete_file_is_merged_with_defaults_and_rewritten(cfg):
    primary, _ = cfg
    _write(primary, {"last_game": 2})

    Settings.load()
    written = json.loads(primary.read_text(encoding="utf-8"))
    assert written["last_game"] == 2
    assert "update_channel" in written, "missing keys were not backfilled on disk"


def test_a_json_list_is_treated_as_corrupt(cfg, tmp_path):
    primary, _ = cfg
    primary.write_text("[1, 2, 3]", encoding="utf-8")

    Settings.load()
    assert list(tmp_path.glob("config.json.corrupt-*"))


@pytest.mark.parametrize("stored, expected", [
    ("PREVIEW", "preview"),
    ("stable", "stable"),
    ("nonsense", "stable"),
])
def test_update_channel_is_normalised_on_load(cfg, stored, expected):
    """The updater reads this to decide which releases to offer."""
    primary, _ = cfg
    _write(primary, {"update_channel": stored})
    Settings.load()
    assert Settings.update_channel == expected


def test_saving_polling_rate_notifies_running_threads(cfg):
    class _Worker:
        name = "polling_rate_probe"
        notified = 0

        def update_polling_rate(self):
            type(self).notified += 1

    worker = _Worker()
    registry.replace(worker)
    try:
        Settings.save(values={"polling_rate": 80})
    finally:
        registry.unregister("polling_rate_probe")

    assert _Worker.notified == 1
    assert Settings.polling_rate == 80


def test_coast_fit_rolling_resistance_0_01_is_remapped(cfg):
    """The coast-fit 0.01 default made mapping worse; load rewrites it."""
    primary, _ = cfg
    Settings.save(force=True)
    data = json.loads(primary.read_text(encoding="utf-8"))
    data["mapper_rolling_resistance"] = 0.01
    _write(primary, data)

    Settings.load()
    assert Settings.mapper_rolling_resistance == pytest.approx(0.07)
    written = json.loads(primary.read_text(encoding="utf-8"))
    assert written["mapper_rolling_resistance"] == pytest.approx(0.07)


def test_custom_rolling_resistance_is_kept(cfg):
    primary, _ = cfg
    Settings.save(values={"mapper_rolling_resistance": 0.02}, force=True)
    Settings.load()
    assert Settings.mapper_rolling_resistance == pytest.approx(0.02)


def test_a_failed_write_leaves_the_on_disk_config_untouched(cfg, monkeypatch):
    primary, _ = cfg
    Settings.save(values={"last_game": 2}, force=True)
    before = primary.read_text(encoding="utf-8")

    def _boom(payload):
        raise OSError("disk full")

    monkeypatch.setattr(Settings, "_atomic_write", classmethod(lambda cls, payload: _boom(payload)))
    Settings.save(values={"last_game": 1}, force=True)
    assert primary.read_text(encoding="utf-8") == before


def test_legacy_accel_ceiling_is_remapped_to_the_safety_rail(cfg):
    """cc_accel_max_ms2 was the accel ceiling; it is now only a rail above the envelope.

    Leaving the old shipped 1.0 in place would cap every acceleration profile at
    today's flat value and silently undo the speed-scheduled envelope.
    """
    primary, _ = cfg
    _write(primary, {"cc_accel_max_ms2": 1.0})
    Settings.load()
    assert Settings.cc_accel_max_ms2 == 2.5


def test_a_deliberately_tuned_accel_ceiling_survives_the_remap(cfg):
    primary, _ = cfg
    _write(primary, {"cc_accel_max_ms2": 1.4})
    Settings.load()
    assert Settings.cc_accel_max_ms2 == 1.4


def test_accel_profile_round_trips_and_rejects_junk(cfg):
    primary, _ = cfg
    _write(primary, {"cc_accel_profile": "sport"})
    Settings.load()
    assert Settings.cc_accel_profile == "Sport"

    _write(primary, {"cc_accel_profile": "ludicrous"})
    Settings.load()
    assert Settings.cc_accel_profile == "Normal"
