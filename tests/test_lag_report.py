"""tools/lag_report.py ranks the GUI stall probe's windows by place and thread, and still reads older logs."""
from __future__ import annotations

import json

from tools import lag_report as lr

_H = "20:10:{:02d} [gui_stall   ] DEBUG    "
_W = "20:10:{:02d} [gui_stall   ] WARNING  "
_GUI_AT = "GUI thread at: 1x monocruise.py:627 main: sys.exit(app.exec()). "


def _window(sec: int, cpu_ms: int, x: int, vehicles: int, aeb_ms: int, tmp: str = "SP") -> list[str]:
    return [
        _H.format(sec) + f"GUI longest gap 120 ms in the last 10 s. CPU {cpu_ms} ms over 10000 ms: "
        f"aeb_thread {aeb_ms} ms (0%), map_thread 500 ms (5%). At x {x} z 200, 60 km/h, {tmp}, "
        f"{vehicles} vehicles + 2 trailers, map ready 300 lanes 4 routed",
        _H.format(sec) + "Loops in the last 10 s: aeb_thread 25/30 Hz, p50 12.0 p99 40.0 max 61 ms of 33, cpu 9.1 ms; "
        "radar_thread 30/30 Hz, p50 2.0 p99 4.0 max 9 ms of 33",
    ]


def _drive() -> list[str]:
    lines = []
    for i in range(6):
        lines += _window(i, 2000 + 100 * i, 500, 5 + i, 300 + 50 * i)
    for i in range(6, 12):
        lines += _window(i, 9000, 15500, 30 + i, 3000 + 100 * i, "TMP")
    lines.append(
        _W.format(13) + "GUI stalled 480 ms. " + _GUI_AT + "From 260 ms in: CPU 210 ms over 220 ms: "
        "aeb_thread 150 ms (68%). At x 15500 z 200, 50 km/h, TMP, 40 vehicles + 2 trailers, map ready 300 lanes 4 routed"
    )
    lines.append(
        _W.format(14) + "GUI stalled 300 ms. " + _GUI_AT + "From 255 ms in: CPU 0 ms over 45 ms: idle. "
        "At x 500 z 200, 50 km/h, SP, 5 vehicles + 0 trailers, map ready 300 lanes 4 routed"
    )
    return lines


def test_the_busiest_place_ranks_first_with_its_traffic_and_thread():
    windows, stalls = lr.parse(_drive())
    report = lr.analyse(windows, stalls, cell_m=1000.0)
    top = report["cells"][0]
    assert (top["x"], top["z"], top["tmp"]) == (15500, 500, True)
    assert top["cpu_mean_pct"] == 90 and top["vehicles_max"] == 41
    assert top["busiest"] == "aeb_thread"
    assert (top["stalls_busy"], top["stalls_quiet"]) == (1, 0)
    assert report["stalls"] == {"busy": 1, "quiet": 1, "unknown": 0}


def test_thread_rows_count_late_windows_and_follow_the_traffic():
    windows, stalls = lr.parse(_drive())
    rows = {r["thread"]: r for r in lr.analyse(windows, stalls)["threads"]}
    aeb = rows["aeb_thread"]
    assert aeb["p99_over_budget"] == 12 and aeb["rate_under_90_pct"] == 12
    assert aeb["corr"]["vehicles"] > 0.9
    assert rows["radar_thread"]["p99_over_budget"] == 0


def test_an_older_log_without_place_or_loops_still_reports_threads():
    old = [
        _H.format(1) + "GUI longest gap 262 ms in the last 10 s. CPU 9859 ms over 10000 ms: "
        "aeb_thread 2688 ms (27%), acc_thread 1781 ms (18%)",
        _W.format(2) + "GUI stalled 306 ms. " + _GUI_AT + "CPU 0 ms over 306 ms: idle",
        _W.format(3) + "GUI stalled 290 ms. GUI thread at: no Python frame. thread CPU unknown",
    ]
    windows, stalls = lr.parse(old)
    report = lr.analyse(windows, stalls)
    assert not report["has_place"] and report["cells"] == []
    assert report["threads"][0]["thread"] == "aeb_thread"
    # Before the From prefix the stall CPU was read late but divided by the whole stall.
    assert report["stalls"] == {"busy": 0, "quiet": 0, "unknown": 2}
    assert "predates" in lr.render(report, 1000.0)


def test_paused_windows_are_not_driving():
    lines = _window(1, 9000, 500, 5, 3000)
    lines[0] = lines[0].replace(", SP,", ", SP, paused,")
    windows, stalls = lr.parse(lines)
    assert lr.analyse(windows, stalls)["driving_windows"] == 0


def test_text_and_json_output_render(tmp_path, capsys):
    log = tmp_path / "monocruise.log"
    log.write_text("\n".join(_drive()), encoding="utf-8")
    assert lr.main([str(log)]) == 0
    text = capsys.readouterr().out
    assert "Hotspots" in text and "aeb_thread" in text
    assert lr.main([str(log), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["driving_windows"] == 12
