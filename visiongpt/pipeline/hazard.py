"""
hazard.py — scans detections for hazardous objects and returns warnings.

Uses the "distance" key added by depth.py. No new models needed.

Usage:
    from visiongpt.pipeline.hazard import check_hazards

    warnings = check_hazards(detections)
    for w in warnings:
        print(w)  # e.g. "Warning. Car very close ahead of you."
"""

from typing import List, Dict, Optional
import time


# Objects that are hazardous at ANY distance
ALWAYS_HAZARDOUS = {"knife", "scissors"}

# Objects hazardous only when close or nearby
CLOSE_HAZARDOUS = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Objects hazardous only when very close
VERY_CLOSE_HAZARDOUS = {"person", "dog", "chair", "dining table", "fire hydrant"}

# Human-friendly names for warnings
HAZARD_NAMES = {
    "car":          "car",
    "truck":        "truck",
    "bus":          "bus",
    "motorcycle":   "motorcycle",
    "bicycle":      "bicycle",
    "person":       "person",
    "dog":          "dog",
    "knife":        "knife",
    "scissors":     "scissors",
    "chair":        "chair",
    "dining table": "table",
    "fire hydrant": "fire hydrant",
}

# Cooldown between repeated warnings for the same hazard (seconds)
HAZARD_COOLDOWN = 10.0

# Track last warning time per hazard label
_last_warned: Dict[str, float] = {}


def _is_hazardous(label: str, distance: Optional[str]) -> bool:
    """Return True if this object+distance combo is a hazard."""
    if label in ALWAYS_HAZARDOUS:
        return True
    if label in CLOSE_HAZARDOUS and distance in ("close", "nearby"):
        return True
    if label in VERY_CLOSE_HAZARDOUS and distance == "close":
        return True
    return False


def _on_cooldown(label: str) -> bool:
    """Return True if we warned about this label recently."""
    last = _last_warned.get(label, 0.0)
    return (time.time() - last) < HAZARD_COOLDOWN


def _mark_warned(label: str):
    _last_warned[label] = time.time()


def _build_warning(label: str, distance: Optional[str], side: Optional[str]) -> str:
    """Build a natural warning sentence."""
    name = HAZARD_NAMES.get(label, label)

    if distance == "close":
        dist_phrase = "very close to you"
    elif distance == "nearby":
        dist_phrase = "nearby"
    else:
        dist_phrase = "ahead"

    if side:
        return f"Warning. {name.capitalize()} {dist_phrase}, {side}."
    return f"Warning. {name.capitalize()} {dist_phrase}."


def _side_from_box(box: List[float], frame_width: int = 640) -> str:
    cx = (box[0] + box[2]) / 2
    third = frame_width / 3
    if cx < third:
        return "on your left"
    elif cx > 2 * third:
        return "on your right"
    else:
        return "ahead of you"


def check_hazards(
    detections: List[Dict],
    frame_width: int = 640,
) -> List[str]:
    """
    Scan detections for hazards and return a list of warning strings.
    Respects cooldown — won't repeat the same warning within HAZARD_COOLDOWN seconds.

    Args:
        detections:  List of detection dicts (with optional "distance" key from depth.py)
        frame_width: Used to determine left/right position

    Returns:
        List of warning strings (may be empty). Caller should speak each one.
    """
    warnings = []

    for det in detections:
        label    = det.get("label", "")
        distance = det.get("distance")   # "close", "nearby", "far", or None
        box      = det.get("box", [0, 0, 0, 0])

        if not _is_hazardous(label, distance):
            continue
        if _on_cooldown(label):
            continue

        side = _side_from_box(box, frame_width)
        warning = _build_warning(label, distance, side)
        warnings.append(warning)
        _mark_warned(label)

    return warnings


def reset_cooldowns():
    """Clear all cooldown timers — useful for testing."""
    _last_warned.clear()