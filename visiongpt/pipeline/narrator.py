from collections import Counter
from typing import List, Dict

PRIORITY_LABELS = {"person", "car", "truck", "bus", "bicycle", "motorcycle", "dog", "cat"}
LOW_PRIORITY_LABELS = {"potted plant", "vase", "clock", "book", "bottle"}

LABEL_ALIASES = {
    "person": "person", "cell phone": "phone", "tv": "television",
    "remote": "remote control", "potted plant": "plant",
    "dining table": "table", "sports ball": "ball",
}

def _alias(label):
    return LABEL_ALIASES.get(label, label)

def _pluralise(word, count):
    if count == 1:
        return word
    irregulars = {"person": "people", "child": "children", "knife": "knives"}
    if word in irregulars:
        return irregulars[word]
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"

def _count_phrase(label, count):
    aliased = _alias(label)
    if count == 1:
        article = "an" if aliased[0].lower() in "aeiou" else "a"
        return f"{article} {aliased}"
    return f"{count} {_pluralise(aliased, count)}"

def _side_hint(box, frame_width=640):
    cx = (box[0] + box[2]) / 2
    third = frame_width / 3
    if cx < third:
        return "on your left"
    elif cx > 2 * third:
        return "on your right"
    else:
        return "ahead of you"

def _sort_detections(detections):
    priority = [d for d in detections if d["label"] in PRIORITY_LABELS]
    normal   = [d for d in detections if d["label"] not in PRIORITY_LABELS and d["label"] not in LOW_PRIORITY_LABELS]
    low      = [d for d in detections if d["label"] in LOW_PRIORITY_LABELS]
    key = lambda d: -d["score"]
    return sorted(priority, key=key) + sorted(normal, key=key) + sorted(low, key=key)

def build_narration(detections, relationships, frame_width=640, max_objects=5):
    if not detections:
        return "I don't see anything recognisable in the frame."

    counts = Counter(d["label"] for d in detections)
    sorted_dets = _sort_detections(detections)

    seen_labels = {}
    for det in sorted_dets:
        if det["label"] not in seen_labels:
            seen_labels[det["label"]] = det

    top_labels = list(seen_labels.keys())[:max_objects]
    phrases = [_count_phrase(label, counts[label]) for label in top_labels]

    if len(phrases) == 1:
        opening = f"I can see {phrases[0]}."
    elif len(phrases) == 2:
        opening = f"I can see {phrases[0]} and {phrases[1]}."
    else:
        opening = "I can see " + ", ".join(phrases[:-1]) + f", and {phrases[-1]}."

    direction_parts = []
    for label in top_labels:
        if label in PRIORITY_LABELS:
            det = seen_labels[label]
            count = counts[label]
            side = _side_hint(det["box"], frame_width)
            aliased = _alias(label)
            noun = _pluralise(aliased, count) if count > 1 else aliased
            article = "" if count > 1 else "The "
            direction_parts.append(f"{article}{noun} {'are' if count > 1 else 'is'} {side}")

    direction_sentence = ". ".join(direction_parts) + "." if direction_parts else ""

    relationship_sentence = ""
    if relationships:
        rel_phrases = []
        for rel in relationships[:2]:
            parts = rel.split()
            if len(parts) >= 3:
                subj = _alias(parts[0])
                prep = parts[1]
                obj  = _alias(" ".join(parts[2:]))
                rel_phrases.append(f"There's a {subj} {prep} a {obj}")
        if rel_phrases:
            relationship_sentence = ". ".join(rel_phrases) + "."

    parts = [p for p in [opening, direction_sentence, relationship_sentence] if p]
    return " ".join(parts)

def narrate_empty_frame():
    return "The frame appears empty. No objects detected."

def narrate_error(error_msg=""):
    base = "There was a problem analysing the image."
    return f"{base} {error_msg}".strip() if error_msg else base