"""Excel column schema and helpers for ROSA summary export."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

from scoring.monitor_peripherals import MonitorPeripheralResult
from scoring.rosa_total import ROSATotalResult
from scoring.sectiona import SectionAResult
from scoring.sectionb import SectionBResult
from scoring.sectionc import SectionCResult

# Column order matches the list provided by the user (single sheet).
EXCEL_HEADERS = [
    "Timestamp",
    "Elbows supported in line with shoulder, shoulders relaxed",
    "Too High (Shoulders Shrugged) / Low (Arms Unsupported)",
    "Too Wide",
    "Work Surface too High",
    "Headset / One Hand on Phone & Neutral Neck Posture",
    "Too Far of Reach (outside of 30 cm)",
    "Neck and Shoulder Hold",
    "No Hands-Free Options",
    "Mouse in line with shoulder",
    "Reaching to mouse",
    "Mouse/Keyboard on different surfaces",
    "Pinch grip on mouse",
    "Palmrest in front of mouse",
    "Keyboard too high – shoulders shrugged",
    "Knees at 90°",
    "Too Low – Knee Angle < 90°",
    "Too High – Knee Angle > 90°",
    "No foot contact on ground",
    "Insufficient space under desk – ability to cross legs",
    "Approximately 3 inches of space between knee and edge of seat",
    "Too Long – Less than 3 inches of space",
    "Too Short – More than 3 inches of space",
    "Adequate lumbar support – chair reclined between 95°–110°",
    "No lumbar support or lumbar support not positioned in small of back",
    "Angled too far back (>110°) or too far forward (<95°)",
    "No back support (e.g., stool or worker leaning forward)",
    "Hard / damaged surface",
    "Neck twist greater than 30°",
    "Deviation while typing",
    "Section A Score",
    "Section B Score",
    "Section C Score",
    "Monitor Peripherals Score",
    "ROSA Final Score (Monitor Peripherals Chair Score)",
]

# Map human-readable headers to the underlying query keys per section.
_QUERY_HEADER_MAP: Dict[str, Tuple[str, str]] = {
    "Elbows supported in line with shoulder, shoulders relaxed": ("front", "elbows_supported_in_line_with_shoulder_shoulders_relaxed"),
    "Too High (Shoulders Shrugged) / Low (Arms Unsupported)": ("front", "too_high_shoulders_shrugged_or_low_arms_unsupported"),
    "Too Wide": ("front", "too_wide"),
    "Work Surface too High": ("side", "work_surface_too_high"),
    "Headset / One Hand on Phone & Neutral Neck Posture": ("front", "headset_or_one_hand_on_phone_neutral_neck_posture"),
    "Too Far of Reach (outside of 30 cm)": ("front", "too_far_of_reach_outside_30_cm"),
    "Neck and Shoulder Hold": ("front", "neck_and_shoulder_hold"),
    "No Hands-Free Options": ("front", "no_hands_free_options"),
    "Mouse in line with shoulder": ("overhead", "mouse_in_line_with_shoulder"),
    "Reaching to mouse": ("overhead", "reaching_to_mouse"),
    "Mouse/Keyboard on different surfaces": ("overhead", "mouse_keyboard_on_different_surfaces"),
    "Pinch grip on mouse": ("overhead", "pinch_grip_on_mouse"),
    "Palmrest in front of mouse": ("overhead", "palmrest_in_front_of_mouse"),
    "Keyboard too high – shoulders shrugged": ("front", "keyboard_too_high_shoulders_shrugged"),
    "Knees at 90°": ("side", "knees_at_90_deg"),
    "Too Low – Knee Angle < 90°": ("side", "too_low_knee_angle_less_than_90_deg"),
    "Too High – Knee Angle > 90°": ("side", "too_high_knee_angle_greater_than_90_deg"),
    "No foot contact on ground": ("side", "no_foot_contact_on_ground"),
    "Insufficient space under desk – ability to cross legs": ("side", "insufficient_space_under_desk_ability_to_cross_legs"),
    "Approximately 3 inches of space between knee and edge of seat": ("side", "approximately_three_inches_between_knee_and_seat_edge"),
    "Too Long – Less than 3 inches of space": ("side", "too_long_less_than_three_inches_of_space"),
    "Too Short – More than 3 inches of space": ("side", "too_short_more_than_three_inches_of_space"),
    "Adequate lumbar support – chair reclined between 95°–110°": ("side", "adequate_lumbar_support_chair_reclined_between_95_110_deg"),
    "No lumbar support or lumbar support not positioned in small of back": ("side", "no_lumbar_support_or_not_positioned_in_small_of_back"),
    "Angled too far back (>110°) or too far forward (<95°)": ("side", "angled_too_far_back_greater_than_110_or_too_far_forward_less_than_95"),
    "No back support (e.g., stool or worker leaning forward)": ("side", "no_back_support_or_worker_leaning_forward"),
    "Hard / damaged surface": ("side", "hard_or_damaged_surface"),
    "Neck twist greater than 30°": ("overhead", "neck_twist_greater_than_30_deg"),
    "Deviation while typing": ("overhead", "deviation_while_typing"),
}


def build_excel_row(
    section_a: SectionAResult,
    section_b: SectionBResult,
    section_c: SectionCResult,
    monitor_peripherals: MonitorPeripheralResult,
    rosa_total: ROSATotalResult,
) -> Dict[str, object]:
    """Build a single Excel row with friendly headers and query values."""
    row: Dict[str, object] = {}
    sources = {
        "front": section_b.query_breakdown,
        "side": section_a.query_breakdown,
        "overhead": section_c.query_breakdown,
    }

    timestamp = max(
        getattr(section_a, "timestamp", 0.0),
        getattr(section_b, "timestamp", 0.0),
        getattr(section_c, "timestamp", 0.0),
    )
    if timestamp:
        row["Timestamp"] = datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        row["Timestamp"] = "null"

    for header, (section_key, query_key) in _QUERY_HEADER_MAP.items():
        source_queries = sources.get(section_key, {})
        if query_key in source_queries:
            raw = source_queries.get(query_key)
            if raw is None:
                row[header] = "null"
            else:
                try:
                    row[header] = int(raw)
                except (TypeError, ValueError):
                    row[header] = raw
        else:
            row[header] = "null"

    row["Section A Score"] = int(section_a.chair_score_final)
    row["Section B Score"] = int(section_b.section_score)
    row["Section C Score"] = int(section_c.section_score)
    row["Monitor Peripherals Score"] = int(monitor_peripherals.combined_score)
    row["ROSA Final Score (Monitor Peripherals Chair Score)"] = int(rosa_total.rosa_total)
    return row


__all__ = ["EXCEL_HEADERS", "build_excel_row"]
