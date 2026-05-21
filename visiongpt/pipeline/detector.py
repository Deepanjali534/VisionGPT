from PIL import Image
import torch
from visiongpt.models.detr import load_model , DEVICE
import config


def detect(image_path: str) -> list[dict]:
    processor, model = load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # ← add this line

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, threshold=config.DETECTION_THRESHOLD, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "box": [round(v, 2) for v in box.tolist()],
            "score": round(score.item(), 4),
        })

    return detections
