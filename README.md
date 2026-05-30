# VisionGPT — Visual Assistant for the Visually Impaired

VisionGPT started as an intelligent object detection agent. It has since been rebuilt into a real-world assistive tool — a visual assistant that describes your surroundings out loud and answers your questions about what it sees.

> Point your camera at the world. VisionGPT tells you what's there.

---

## What it does

- **Describes your surroundings aloud** — "I can see a person on your left, close to a chair. There's also a car ahead of you."
- **Live webcam narration** — analyses the scene every few seconds and speaks only when something changes
- **Voice Q&A** — ask a question out loud, get a spoken answer back — "What colour is the chair?" → "brown"
- **Text Q&A fallback** — type questions if you prefer
- **High-contrast accessible UI** — large text, large buttons, keyboard-friendly layout

---

## Models used

| Task | Model |
| ---- | ----- |
| Object detection | [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) |
| Visual question answering | [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) |

No model training involved — both models are pretrained and loaded directly from HuggingFace.

---

## Project structure

```
VisionGPT/
│
├── visiongpt/
│   ├── models/
│   │   └── detr.py               # loads DETR + device setup
│   ├── pipeline/
│   │   ├── detector.py           # image → detected objects
│   │   ├── scene_graph.py        # objects → spatial relationships
│   │   ├── vqa.py                # image + question → answer
│   │   └── narrator.py           # detections → natural spoken sentences
│   ├── tts/
│   │   ├── __init__.py
│   │   └── speaker.py            # TTS wrapper (pyttsx3 via subprocess)
│   └── voice/
│       ├── __init__.py
│       └── listener.py           # mic input → text (SpeechRecognition)
│
├── app.py                        # original UI (upload image + webcam)
├── app_blind.py                  # accessible UI for visually impaired
├── speak_worker.py               # pyttsx3 worker (called as subprocess)
├── config.py                     # model names, thresholds, paths
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Deepanjali534/VisionGPT.git
cd VisionGPT
```

### 2. Create and activate virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Visual Assistant (blind-friendly UI)

```bash
streamlit run app_blind.py
```

Opens at `http://localhost:8501`. Features:

- Click **▶ Start** — camera opens, scene is described aloud every few seconds
- Click **🎤 Ask with voice** — speak a question, hear the answer
- Or type a question in the text box and click **Ask**
- Click **🔁 Repeat** to hear the last description or answer again
- Use the slider to control how often the scene is described (2–10 seconds)

### Original object detection UI

```bash
streamlit run app.py
```

Upload any image or use the live webcam tab — see detected objects with bounding boxes, spatial relationships, and text-based Q&A.

---

## How it works

```
Webcam frame
      ↓
DETR detects every object → bounding boxes
      ↓
Scene graph finds spatial relationships → "person near chair"
      ↓
Narrator builds a natural sentence → "I can see a person on your left, near a chair."
      ↓
pyttsx3 speaks it aloud (offline, no API key needed)
      ↓
User asks a question (voice or text)
      ↓
BLIP answers from the image → spoken back via pyttsx3
```

---

## Configuration

All settings live in `config.py`:

```python
DETR_MODEL_NAME     = "facebook/detr-resnet-50"
BLIP2_MODEL_NAME    = "Salesforce/blip-vqa-base"
DETECTION_THRESHOLD = 0.9    # lower to 0.7 if objects are being missed
NEAR_THRESHOLD      = 150    # pixel distance to count as "near"
OUTPUT_DIR          = "outputs"
```

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
pyttsx3
SpeechRecognition
```