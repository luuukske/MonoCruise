"""Regression: ego_has_trailer must read the nested trailer array, not flat keys."""

from __future__ import annotations

from core.telemetry_thread.thread import TelemetryThreadData, _apply_telemetry


def _trailer(attached: bool = True, wheels: int = 8) -> dict:
    return {
        "attached": attached,
        "wheelCount": wheels,
        "wheelOnGround": [True] * wheels,
    }


def _raw(trailers: list[dict] | None = None, **over) -> dict:
    raw = {
        "trailer": trailers if trailers is not None else [],
        "truckWheelOnGround": [True] * 6,
        "cargoMass": 0.0,
        "fuel": 0.0,
    }
    raw.update(over)
    return raw


def _apply(raw: dict) -> TelemetryThreadData:
    data = TelemetryThreadData()
    _apply_telemetry(data, raw)
    return data


def test_attached_trailer_sets_flag():
    data = _apply(_raw([_trailer()]))
    assert data.ego_has_trailer is True
    assert data.trailer_count == 1


def test_no_trailer_clears_flag():
    data = _apply(_raw([]))
    assert data.ego_has_trailer is False
    assert data.trailer_count == 0


def test_detached_trailer_slot_does_not_count():
    data = _apply(_raw([_trailer(attached=False)]))
    assert data.ego_has_trailer is False


def test_attached_but_wheelless_slot_does_not_count():
    data = _apply(_raw([_trailer(wheels=0)]))
    assert data.ego_has_trailer is False


def test_multiple_trailers_counted():
    data = _apply(_raw([_trailer(), _trailer()]))
    assert data.ego_has_trailer is True
    assert data.trailer_count == 2


def test_flag_agrees_with_trailer_count():
    """The two were derived independently once, and disagreed for every real frame."""
    for trailers in ([], [_trailer()], [_trailer(attached=False)], [_trailer(), _trailer()]):
        data = _apply(_raw(trailers))
        assert data.ego_has_trailer == (data.trailer_count > 0)


def test_flat_indexed_keys_are_not_the_source():
    """A raw dict carrying only flat "trailer[0].*" keys must not set the flag."""
    raw = _raw([], **{"trailer[0].wheelCount": 8, "trailer[0].attached": True})
    assert _apply(raw).ego_has_trailer is False


def test_trailer_adds_estimated_mass():
    empty = _apply(_raw([]))
    loaded = _apply(_raw([_trailer()]))
    assert loaded.estimated_total_mass_kg > empty.estimated_total_mass_kg
