from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def load_vqa_model():
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(config.BLIP2_MODEL_NAME)
        _model = BlipForQuestionAnswering.from_pretrained(config.BLIP2_MODEL_NAME)
        _model.to(DEVICE).eval()
    return _processor, _model


def answer_question(image_path: str, question: str) -> str:
    processor, model = load_vqa_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(generated[0], skip_special_tokens=True).strip()