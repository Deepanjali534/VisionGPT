"""
app_blind.py — Visual Assistant for the Visually Impaired
Run with: streamlit run app_blind.py

Features:
- Live webcam feed with auto-narration every N seconds
- Speaks scene description aloud via pyttsx3
- Voice or text Q&A — asks BLIP, speaks the answer back
- High-contrast, large-text accessible UI
"""
import subprocess
import sys
import streamlit as st
import cv2
import tempfile
import os
import time
import threading
from PIL import Image
import numpy as np

from visiongpt.pipeline.detector import detect
from visiongpt.pipeline.scene_graph import build_scene_graph
from visiongpt.pipeline.vqa import load_vqa_model, answer_question
from visiongpt.pipeline.narrator import build_narration, narrate_error
from visiongpt.tts.speaker import get_speaker

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Visual Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Accessible CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Large readable text throughout */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    /* Big prominent buttons */
    .stButton > button {
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 18px 36px !important;
        border-radius: 12px !important;
        width: 100% !important;
        margin-bottom: 12px !important;
    }

    /* Primary action button — high contrast green */
    .stButton > button[kind="primary"] {
        background-color: #1a7a1a !important;
        color: #ffffff !important;
        border: 3px solid #0f5c0f !important;
    }

    /* Stop button — high contrast red */
    .stop-btn > button {
        background-color: #cc0000 !important;
        color: #ffffff !important;
        border: 3px solid #990000 !important;
    }

    /* Narration box */
    .narration-box {
        background-color: #1a1a2e;
        color: #e0e0ff;
        font-size: 26px !important;
        font-weight: 600;
        padding: 24px 28px;
        border-radius: 14px;
        border-left: 8px solid #4a90e2;
        margin: 16px 0;
        line-height: 1.6;
    }

    /* Answer box */
    .answer-box {
        background-color: #1a2e1a;
        color: #d0f0d0;
        font-size: 24px !important;
        font-weight: 600;
        padding: 20px 24px;
        border-radius: 14px;
        border-left: 8px solid #4caf50;
        margin: 16px 0;
        line-height: 1.6;
    }

    /* Warning box */
    .warning-box {
        background-color: #2e1a00;
        color: #ffd080;
        font-size: 22px !important;
        font-weight: 700;
        padding: 16px 24px;
        border-radius: 14px;
        border-left: 8px solid #ff9800;
        margin: 12px 0;
    }

    /* Section headers */
    h1 { font-size: 36px !important; }
    h2 { font-size: 28px !important; }
    h3 { font-size: 24px !important; }

    /* Text input larger */
    .stTextInput > div > div > input {
        font-size: 20px !important;
        padding: 14px !important;
    }

    /* Slider label */
    .stSlider label {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────

def _init_state():
    defaults = {
        "running":          False,
        "last_narration":   "",
        "last_frame_path":  None,
        "speak_thread":     None,
        "vqa_loaded":       False,
        "last_answer":      "",
        "last_question":    "",
        "detections":       [],
        "relationships":    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ────────────────────────────────────────────────────────────────

def speak_async(text: str):
    """Launch speak_worker.py as a fully detached process."""
    if not text or not text.strip():
        return
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak_worker.py")
    subprocess.Popen(
        [sys.executable, worker, text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

def analyse_frame(frame_bgr, frame_width: int):
    """Run detection + scene graph on a raw BGR frame. Returns (dets, rels, narration)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_img.save(tmp.name)
        tmp_path = tmp.name

    try:
        dets = detect(tmp_path)
        rels = build_scene_graph(dets)
        narration = build_narration(dets, rels, frame_width=frame_width)
        return dets, rels, narration, tmp_path
    except Exception as e:
        return [], [], narrate_error(str(e)), tmp_path


# ── UI Layout ──────────────────────────────────────────────────────────────

st.title("👁️ Visual Assistant")
st.markdown("#### Helping you understand what's around you")
st.divider()

# Two columns: controls left, camera right
col_ctrl, col_cam = st.columns([1, 2])

with col_ctrl:
    st.markdown("### Controls")

    # Narration interval slider
    interval = st.slider(
        "Describe scene every (seconds)",
        min_value=2, max_value=10, value=4, step=1
    )

    st.markdown("---")

    # Start / Stop buttons
    if not st.session_state["running"]:
        if st.button("▶ Start — Describe my surroundings", type="primary"):
            st.session_state["running"] = True
            st.rerun()
    else:
        with st.container():
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("⏹ Stop", type="secondary"):
                st.session_state["running"] = False
                get_speaker().stop()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Last narration display
    st.markdown("### 🔊 Last description")
    if st.session_state["last_narration"]:
        st.markdown(
            f'<div class="narration-box">{st.session_state["last_narration"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="narration-box">Press Start to begin.</div>',
            unsafe_allow_html=True
        )

    # Repeat last narration button
    if st.session_state["last_narration"]:
        if st.button("🔁 Repeat last description"):
            speak_async(st.session_state["last_narration"])

    st.markdown("---")

    # ── Q&A section ──
    st.markdown("### ❓ Ask about the scene")

    question = st.text_input(
        "Type your question",
        placeholder="What colour is the chair? Is there a person?",
        key="question_input"
    )

    if st.button("Ask") and question.strip():
        if st.session_state["last_frame_path"] and os.path.exists(st.session_state["last_frame_path"]):
            with st.spinner("Thinking..."):
                if not st.session_state["vqa_loaded"]:
                    load_vqa_model()
                    st.session_state["vqa_loaded"] = True
                answer = answer_question(st.session_state["last_frame_path"], question)
                st.session_state["last_answer"]   = answer
                st.session_state["last_question"]  = question
                speak_async(answer)
        else:
            st.warning("No frame captured yet — start the camera first.")

    if st.session_state["last_answer"]:
        st.markdown(
            f'<div class="answer-box">'
            f'<b>Q:</b> {st.session_state["last_question"]}<br>'
            f'<b>A:</b> {st.session_state["last_answer"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("🔁 Repeat answer"):
            speak_async(st.session_state["last_answer"])


# ── Camera loop ────────────────────────────────────────────────────────────

with col_cam:
    st.markdown("### 📷 Camera feed")

    frame_placeholder   = st.empty()
    status_placeholder  = st.empty()
    objects_placeholder = st.empty()

    if st.session_state["running"]:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam. Make sure your camera is connected.")
            st.session_state["running"] = False
        else:
            speak_async("Visual assistant started. Analysing your surroundings.")

            frame_count   = 0
            last_analysed = 0.0   # timestamp of last full analysis
            FRAME_SKIP    = 3     # display every Nth frame for smoothness

            detections    = []

            while st.session_state["running"]:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera read failed.")
                    break

                frame_count += 1
                now = time.time()

                # ── Full analysis at set interval ──
                if now - last_analysed >= interval:
                    dets, rels, narration, tmp_path = analyse_frame(frame, frame.shape[1])
                    detections = dets

                    # Save frame path for Q&A
                    if st.session_state["last_frame_path"] and \
                       os.path.exists(st.session_state["last_frame_path"]):
                        try:
                            os.unlink(st.session_state["last_frame_path"])
                        except OSError:
                            pass
                    st.session_state["last_frame_path"] = tmp_path

                    # Only speak if narration changed
                    if narration != st.session_state["last_narration"]:
                        st.session_state["last_narration"] = narration
                        speak_async(narration)

                    st.session_state["detections"]     = dets
                    st.session_state["relationships"]  = rels
                    last_analysed = now

                # ── Draw bounding boxes ──
                COLORS = [
                    (0, 200, 100), (0, 120, 255), (255, 160, 0),
                    (220, 0, 220), (0, 220, 220), (255, 80, 80),
                ]
                for i, det in enumerate(detections):
                    color = COLORS[i % len(COLORS)]
                    x1, y1, x2, y2 = [int(v) for v in det["box"]]
                    label = f"{det['label']} {det['score']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 28), (x1 + len(label) * 11, y1), (0,0,0), -1)
                    cv2.putText(frame, label, (x1 + 4, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # ── Show frame ──
                if frame_count % FRAME_SKIP == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # ── Object list below camera ──
                if detections:
                    obj_text = " · ".join(
                        f"`{d['label']}` {d['score']:.2f}" for d in detections
                    )
                    objects_placeholder.markdown(f"**Detected:** {obj_text}")

                status_placeholder.markdown(
                    f"🟢 **Running** — next description in "
                    f"`{max(0, interval - int(now - last_analysed))}s`"
                )

            cap.release()
            frame_placeholder.empty()
            status_placeholder.markdown("⚪ Stopped.")
            speak_async("Visual assistant stopped.")