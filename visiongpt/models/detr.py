# ✅ FIXED CODE — replace your entire detr.py with this
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import config

# automatically picks GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None

def load_model():
    global _processor, _model
    if _model is None:
        _processor = DetrImageProcessor.from_pretrained(config.DETR_MODEL_NAME)
        _model = DetrForObjectDetection.from_pretrained(config.DETR_MODEL_NAME)
        _model.to(DEVICE).eval()  # ← this line is the fix
    return _processor, _model
