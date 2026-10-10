"""Which raw HID report bits are buttons, read from the report descriptor. See README.md."""

from __future__ import annotations

from dataclasses import dataclass, field

_BUTTON_PAGE = 0x09
_LONG_ITEM = 0xFE
_ITEM_SIZES = (0, 1, 2, 4)

# Main item tags (bits 7..4 of the prefix).
_MAIN_INPUT = 0x8
# Global item tags.
_GLOBAL_USAGE_PAGE = 0x0
_GLOBAL_REPORT_SIZE = 0x7
_GLOBAL_REPORT_ID = 0x8
_GLOBAL_REPORT_COUNT = 0x9
_GLOBAL_PUSH = 0xA
_GLOBAL_POP = 0xB
# Local item tags that may carry an extended (page << 16 | usage) value.
_LOCAL_USAGE = 0x0
_LOCAL_USAGE_MIN = 0x1


@dataclass(frozen=True)
class ButtonLayout:
    """Button bit positions per report, indexed like `button_id` in INPUT_BINDINGS.md."""

    uses_report_ids: bool
    bits_by_report: dict[int, frozenset[int]] = field(default_factory=dict)
    all_bits: frozenset[int] = frozenset()

    def bits_for(self, report: list[int]) -> frozenset[int]:
        """Button bits of this raw report (the first byte names the report when IDs are used)."""
        if self.uses_report_ids:
            if not report:
                return frozenset()
            return self.bits_by_report.get(report[0], frozenset())
        return self.bits_by_report.get(0, frozenset())

    def is_button(self, button_id: int) -> bool:
        return button_id in self.all_bits


def parse_button_layout(descriptor: list[int] | bytes) -> ButtonLayout | None:
    """Parse a HID report descriptor; None if it is malformed."""
    data = list(descriptor)
    usage_page = 0
    report_size = 0
    report_count = 0
    report_id = 0
    stack: list[tuple[int, int, int, int]] = []
    local_page: int | None = None
    uses_ids = False
    offsets: dict[int, int] = {}
    bits: dict[int, set[int]] = {}

    i = 0
    while i < len(data):
        prefix = data[i]
        if prefix == _LONG_ITEM:
            if i + 1 >= len(data):
                return None
            i += 3 + data[i + 1]
            continue
        size = _ITEM_SIZES[prefix & 0x03]
        item_type = (prefix >> 2) & 0x03
        tag = prefix >> 4
        if i + 1 + size > len(data):
            return None
        value = int.from_bytes(bytes(data[i + 1:i + 1 + size]), "little")
        i += 1 + size

        if item_type == 0:
            if tag == _MAIN_INPUT:
                width = report_size * report_count
                start = offsets.get(report_id, 0)
                constant = bool(value & 0x01)
                variable = bool(value & 0x02)
                page = local_page if local_page is not None else usage_page
                if not constant and variable and report_size == 1 and page == _BUTTON_PAGE:
                    bits.setdefault(report_id, set()).update(range(start, start + width))
                offsets[report_id] = start + width
            # Every main item ends the local state; output and feature
            # reports do not move input bit offsets.
            local_page = None
        elif item_type == 1:
            if tag == _GLOBAL_USAGE_PAGE:
                usage_page = value
            elif tag == _GLOBAL_REPORT_SIZE:
                report_size = value
            elif tag == _GLOBAL_REPORT_ID:
                report_id = value
                uses_ids = True
            elif tag == _GLOBAL_REPORT_COUNT:
                report_count = value
            elif tag == _GLOBAL_PUSH:
                stack.append((usage_page, report_size, report_count, report_id))
            elif tag == _GLOBAL_POP:
                if not stack:
                    return None
                usage_page, report_size, report_count, report_id = stack.pop()
        elif item_type == 2:
            if tag in (_LOCAL_USAGE, _LOCAL_USAGE_MIN) and size == 4 and local_page is None:
                local_page = value >> 16

    # hidapi hands numbered reports over with the ID in byte 0.
    shift = 8 if uses_ids else 0
    by_report = {
        rid: frozenset(b + shift for b in rid_bits) for rid, rid_bits in bits.items()
    }
    return ButtonLayout(
        uses_report_ids=uses_ids,
        bits_by_report=by_report,
        all_bits=frozenset().union(*by_report.values()),
    )


def read_button_layout(device) -> ButtonLayout | None:
    """Layout of an open hidapi device, or None when the descriptor is unavailable."""
    try:
        descriptor = device.get_report_descriptor()
    except Exception:
        return None
    if not descriptor:
        return None
    return parse_button_layout(descriptor)
