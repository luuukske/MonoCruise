"""A guardrail curve follows the node headings. The chord between the ends does not."""

from tools.map_obstacle_props import _assign_colors
from core.road_layout.sector import MapNode
from tools.map_obstacle_probe import bend_points, is_clutter


def test_a_turning_guardrail_leaves_the_straight_chord():
    start = MapNode(1, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    end = MapNode(2, 80.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    span = bend_points([start, end])
    assert len(span) >= 8
    mid = span[len(span) // 2]
    assert abs(mid[2]) > 5.0


def test_paint_wires_and_low_curbs_are_clutter_and_a_barrier_is_not():
    assert is_clutter("/model2/props/road/at_lines.pmd")
    assert is_clutter("/model2/power_line/power_wire_straight_crv.pmd")
    assert is_clutter("/model2/props/road/curb_blke.pmd")
    assert is_clutter("/model2/props/road/curb_stripe_blke.pmd")
    assert not is_clutter("/model2/railing/barrier/crash_barrier_01_es_ibe_crv.pmd")
    assert is_clutter("/props/street/sign_board.pmd")
    assert is_clutter("/model/sign/traffic/at/chevron_right_at.pmd")
    assert is_clutter("sign:at_2i00g")
    assert not is_clutter("/model2/pillar/cylinder_01.pmd")


def test_each_model_path_gets_its_own_color():
    labels = [f"/model/item_{i}.pmd" for i in range(40)]
    colors = _assign_colors(labels)
    assert len(colors) == 40
    assert len(set(colors.values())) == 40
    assert all(value.startswith("#") and len(value) == 7 for value in colors.values())
