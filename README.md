# VisionGPT — Intelligent Object Interaction Agent

Most object detectors just tell you *"there's a car"*. VisionGPT tells you *"there are 12 cars on the highway, 3 of them near each other in the left lane"* — and then answers your questions about it.



---

## What it can do

-  **Detect objects** in any photo or live webcam feed
-  **Understand relationships** between objects — "car near truck", "person on chair"
-  **Answer questions** about images in plain English — "How many cars are on the road?"
-  **Live webcam detection** — real time object detection from your laptop camera

---

## Models used

| Task | Model | Size |
|---|---|---|
| Object detection | [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) 
| Visual question answering | [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) 



---

## Project structure

```
visiongpt/
│
├── visiongpt/                  ← all Python source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── detr.py             # loads DETR model + device setup
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── detector.py         # image → detected objects
│   │   ├── scene_graph.py      # objects → spatial relationships
│   │   └── vqa.py              # image + question → answer
│   └── utils/
│       ├── __init__.py
│       └── visualizer.py       # draws bounding boxes, saves result
│
├── test_images/                ← sample images for testing
│   ├── iitg4.jpg
│   ├── image.png               ← highway car detection example
│   └── image2.png
│
├── outputs/                    ← saved result images go here
│
├── app.py                      # Streamlit web UI (upload + webcam tabs)
├── webcam.py                   # standalone live webcam detection
├── main.py                     # CLI entry point
├── config.py                   # model names, thresholds, paths
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourteam/visiongpt.git
cd visiongpt
```

### 2. Create and activate virtual environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```



---

## Usage

### Web UI — upload image + live webcam
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser. Two tabs:

**  Upload Image tab**
1. Upload any photo (JPG or PNG)
2. Click **Analyze** — see detected objects with coloured bounding boxes and scene relationships
3. Type a question like `"How many cars are on the road?"` → click **Ask** → get a text answer

**🎥 Live Webcam tab**
1. Click **▶ Start Webcam** — your camera opens inside the browser
2. Move objects in front of the camera — boxes appear in real time
3. Click **⏹ Stop Webcam** to end the session

### Standalone webcam (runs outside browser)
```bash
python webcam.py
```
Opens a dedicated camera window with live detection. Press **Q** to quit.

### CLI
```bash
python main.py test_images/image.png
python main.py test_images/image.png --question "How many cars are on the road?"
```

---

## Example output

Input image — highway with multiple vehicles (`test_images/image.png`):

| | |
|---|---|
| **Input** | Real highway photo with cars and trucks |
| **Detected** | 12 cars, 1 truck — each with a coloured bounding box |
| **Relationships** | car near car · truck near car |
| **VQA answer** | "How many cars are on the road?" → `"12"` |

The model correctly identified all vehicles on the highway and drew tight bounding boxes around each one — even the smaller cars in the distance.

---

## Configuration

All settings live in `config.py` — change them without touching any other file:

```python
DETR_MODEL_NAME      = "facebook/detr-resnet-50"
BLIP2_MODEL_NAME     = "Salesforce/blip-vqa-base"
DETECTION_THRESHOLD  = 0.9       # minimum confidence to show a detection
OUTPUT_DIR           = "outputs"
NEAR_THRESHOLD       = 150       # pixel distance to count as "near"
```

**Tip:** Lower `DETECTION_THRESHOLD` to `0.7` if the model is missing objects. Raise it to `0.95` to reduce false detections.

---

## Requirements

```
torch
torchvision
transformers
Pillow
streamlit
accelerate
sentencepiece
timm
opencv-python
```

---

## How it works — simple version

```
Your image or webcam frame
        ↓
DETR scans it → finds every object → draws bounding boxes
        ↓
Scene Graph looks at box positions → "car near truck"
        ↓
BLIP looks at image + your question → plain English answer
        ↓
Streamlit shows everything in the browser
```

No model training involved — both DETR and BLIP are pretrained models loaded directly from HuggingFace. The scene graph logic is pure Python math on bounding box positions.

---

## Applications

- **Traffic monitoring** — count vehicles, detect congestion, identify object types on roads
- **Home automation** — understand room context for smart home triggers
- **Robotics** — spatial reasoning for object navigation and manipulation
- **AR perception** — real time scene understanding for augmented reality overlays

---

## Team

| Role | Files owned |
|---|---|
| ML Core | `visiongpt/models/detr.py`, `visiongpt/pipeline/detector.py` |
| Scene + VQA | `visiongpt/pipeline/scene_graph.py`, `visiongpt/pipeline/vqa.py` |
| Visualizer | `visiongpt/utils/visualizer.py`, `main.py`, `config.py` |
| UI + Docs | `app.py`, `webcam.py`, `README.md` |