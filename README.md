# VisionGPT — Visual Assistant for the Visually Impaired

VisionGPT is a real-world assistive tool that describes your surroundings out loud and answers your questions about what it sees — built for the visually impaired.

> Point your camera at the world. VisionGPT tells you what's there.

---

## What it does

- **Describes your surroundings aloud** — "I can see a person very close to you on your left, and a car in the distance."
- **Live webcam narration** — analyses the scene every few seconds and speaks only when something changes
- **Depth-aware descriptions** — tells you if objects are close, nearby, or far
- **Hazard alerts** — immediately warns you when dangerous objects (cars, knives, people) are too close — "Warning. Car very close to you, on your right."
- **Voice Q&A** — ask a question out loud, get a spoken answer back
- **Text Q&A fallback** — type questions if you prefer
- **High-contrast accessible UI** — large text, large buttons, keyboard-friendly layout

---

## Models used

| Task | Model |
|------|-------|
| Object detection | [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) |
| Visual question answering | [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) |
| Depth estimation | [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas) |

No model training involved — all models are pretrained and downloaded automatically from HuggingFace on first run.

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
│   │   ├── narrator.py           # detections → natural spoken sentences
│   │   ├── depth.py              # depth estimation → close/nearby/far labels
│   │   └── hazard.py             # hazard detection → warning sentences
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
- Hazard warnings play automatically — no action needed

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
DETR detects every object + bounding boxes
      ↓
MiDaS estimates depth → labels each object close / nearby / far
      ↓
Hazard check → speaks warning immediately if dangerous object is too close
      ↓
Scene graph finds spatial relationships → "person near chair"
      ↓
Narrator builds a natural sentence
→ "I can see a person very close to you on your left, and a chair nearby."
      ↓
pyttsx3 speaks it aloud (offline, no API key needed)
      ↓
User asks a question (voice or text)
      ↓
BLIP answers from the image → spoken back via pyttsx3
```

---

## Hazard reference

| Object | Triggers warning when |
|--------|-----------------------|
| Car, truck, bus | Close or nearby |
| Motorcycle, bicycle | Close or nearby |
| Person | Very close |
| Dog | Very close |
| Knife, scissors | Any distance |
| Chair, table | Very close (tripping hazard) |
| Fire hydrant | Very close (tripping hazard) |

Warnings have a 10-second cooldown per object type to avoid spamming.

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

## First run note

On first launch, three models will download automatically:
- DETR (~160MB)
- BLIP VQA (~900MB)
- MiDaS (~490MB)

This only happens once — models are cached locally after that.