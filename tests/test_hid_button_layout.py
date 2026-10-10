"""Report-descriptor parsing: only real buttons may be captured or bound.

A pedal set the joystick scan did not own was raw-scanned for button
assignment, and its axis bits were bound as cruise buttons. See
core/button_device_thread/README.md.
"""

from __future__ import annotations

from core.button_device_thread.hid_descriptor import parse_button_layout


def _hex(text: str) -> list[int]:
    return [int(b, 16) for b in text.split()]


# MOZA Multi-function Stalk (346e:0024): 32 buttons, 4 constant bytes, no report IDs.
STALK = _hex(
    "05 01 09 04 a1 01 05 09 19 01 29 20 15 00 25 01 75 01 95 20 81 02"
    " 75 08 95 04 81 03 c0"
)

# Three 16-bit pedal axes and nothing else.
PEDALS = _hex(
    "05 01 09 04 a1 01 09 01 a1 00 09 30 09 31 09 32 15 00 26 ff 0f"
    " 75 10 95 03 81 02 c0 c0"
)

# Report 1: two 8-bit axes, an output byte, 12 buttons, a 4-bit pad, then
# PID-page status flags. Report 2: 8 buttons.
NUMBERED = _hex(
    "05 01 09 05 a1 01 85 01"
    " 09 30 09 31 15 00 26 ff 00 75 08 95 02 81 02"
    " 75 08 95 01 91 02"
    " 05 09 19 01 29 0c 15 00 25 01 75 01 95 0c 81 02"
    " 75 04 95 01 81 03"
    " 05 0f 09 9f 09 a0 75 01 95 02 81 02"
    " 85 02 05 09 19 01 29 08 75 01 95 08 81 02"
    " c0"
)


def test_stalk_buttons_are_the_first_four_bytes():
    layout = parse_button_layout(STALK)
    assert layout is not None and not layout.uses_report_ids
    assert layout.all_bits == frozenset(range(32))
    # The shipped stalk bindings (byte 3, bits 1 to 3) stay valid.
    assert all(layout.is_button(b) for b in (25, 26, 27))
    assert not layout.is_button(32)


def test_pedal_axes_are_never_buttons():
    layout = parse_button_layout(PEDALS)
    assert layout is not None
    assert layout.all_bits == frozenset()
    assert layout.bits_for([0xFF] * 6) == frozenset()


def test_numbered_reports_shift_by_the_id_byte_and_skip_outputs():
    layout = parse_button_layout(NUMBERED)
    assert layout is not None and layout.uses_report_ids
    # ID byte (8) + axes (16): output items do not move input offsets.
    assert layout.bits_by_report[1] == frozenset(range(24, 36))
    assert layout.bits_by_report[2] == frozenset(range(8, 16))
    assert layout.bits_for([1, 0, 0, 0, 0]) == frozenset(range(24, 36))
    assert layout.bits_for([2, 0]) == frozenset(range(8, 16))
    assert layout.bits_for([9, 0]) == frozenset()


def test_one_bit_fields_off_the_button_page_are_not_buttons():
    """PID status flags are 1-bit inputs too; only the Button page counts."""
    layout = parse_button_layout(NUMBERED)
    assert not layout.is_button(40) and not layout.is_button(41)


def test_extended_usage_selects_the_button_page():
    # Global page stays Generic Desktop; the 4-byte usage names page 0x09.
    desc = _hex("05 01 09 04 a1 01 1b 01 00 09 00 2b 04 00 09 00 75 01 95 04 81 02 c0")
    layout = parse_button_layout(desc)
    assert layout is not None and layout.all_bits == frozenset(range(4))


def test_constant_and_array_fields_are_skipped():
    desc = _hex(
        "05 09 19 01 29 08 75 01 95 08 81 03"  # constant
        " 75 01 95 08 81 00"                   # array
        " 75 01 95 08 81 02"                   # variable buttons, bits 16..23
    )
    layout = parse_button_layout(desc)
    assert layout is not None and layout.all_bits == frozenset(range(16, 24))


def test_truncated_descriptor_is_rejected():
    assert parse_button_layout(STALK[:-3] + [0x26]) is None
