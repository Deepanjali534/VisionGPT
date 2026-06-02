"""
depth.py — estimates relative depth for detected objects using MiDaS.

Takes a PIL image and a list of detections, returns the same detections
with a "distance" key added to each: "close", "nearby", or "far".

Usage:
    from visiongpt.pipeline.depth import load_depth_model, estimate_distances

    load_depth_model()   # call once at startup

    detections = [
        {"label": "person", "score": 0.97, "box": [50, 100, 200, 400]},
        ...
    ]
    detections = estimate_distances("path/to/image.jpg", detections)
    # detections[0]["distance"] → "close"
"""

import numpy as np
from PIL import Image
from typing import List, Dict

# Module-level model cache
_depth_pipe = None


def load_depth_model():
    """
    Load MiDaS depth estimation pipeline from HuggingFace.
    Safe to call multiple times — loads only once.
    """
    global _depth_pipe
    if _depth_pipe is not None:
        return

    from transformers import pipeline as hf_pipeline
    _depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="Intel/dpt-hybrid-midas",
    )


def _get_box_depth(depth_array: np.ndarray, box: List[float]) -> float:
    """
    Return the median depth value inside a bounding box region.
    Higher value = closer to camera in MiDaS output.
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # Clamp to image bounds
    h, w = depth_array.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = depth_array[y1:y2, x1:x2]
    return float(np.median(region))


def _label_distance(depth_val: float, d_min: float, d_max: float) -> str:
    """
    Convert a raw depth value into a human-readable distance label.
    MiDaS produces relative depth — we bucket into thirds of the scene range.
    """
    if d_max == d_min:
        return "nearby"

    # Normalise 0→1 within this frame's depth range
    norm = (depth_val - d_min) / (d_max - d_min)

    if norm > 0.66:
        return "close"
    elif norm > 0.33:
        return "nearby"
    else:
        return "far"


def estimate_distances(
    image_path: str,
    detections: List[Dict],
) -> List[Dict]:
    """
    Add a "distance" key to each detection dict.

    Args:
        image_path:  Path to the source image file.
        detections:  List of detection dicts from detector.py.

    Returns:
        Same list with "distance" added to each dict.
        Falls back to "nearby" for all objects if depth model isn't loaded.
    """
    if not detections:
        return detections

    # Graceful fallback if model not loaded
    if _depth_pipe is None:
        for det in detections:
            det["distance"] = "nearby"
        return detections

    # Run depth estimation
    image = Image.open(image_path).convert("RGB")
    result = _depth_pipe(image)
    depth_pil = result["depth"]   # PIL image, same size as input

    # Convert to numpy — resize to match original image if needed
    orig_w, orig_h = image.size
    depth_resized = depth_pil.resize((orig_w, orig_h), Image.BILINEAR)
    depth_array = np.array(depth_resized, dtype=np.float32)

    # Global min/max for normalisation
    d_min = float(depth_array.min())
    d_max = float(depth_array.max())

    # Assign distance label to each detection
    for det in detections:
        val = _get_box_depth(depth_array, det["box"])
        det["distance"] = _label_distance(val, d_min, d_max)

    return detections