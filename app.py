import streamlit as st
import tempfile
import os
from visiongpt.pipeline.detector import detect
from visiongpt.pipeline.scene_graph import build_scene_graph
from visiongpt.pipeline.vqa import load_vqa_model, answer_question
from visiongpt.utils.visualizer import visualize

st.set_page_config(page_title="VisionGPT", layout="wide")
st.title("VisionGPT — Intelligent Object Interaction Agent")

# ── two tabs ──────────────────────────────────────────────
tab1, tab2 = st.tabs([" Upload Image", " Live Webcam"])


# ════════════════════════════════════════════════════════
# TAB 1 — Upload Image (your existing code, unchanged)
# ════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        col1, col2 = st.columns(2)
        with col1:
            st.image(tmp_path, caption="Input", use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Detecting objects..."):
                detections = detect(tmp_path)

            with st.spinner("Building scene graph..."):
                relationships = build_scene_graph(detections)

            output_path = visualize(tmp_path, detections)

            with col2:
                st.image(output_path, caption="Detected Objects", use_container_width=True)

            st.subheader("Detected Objects")
            for d in detections:
                st.write(f"- **{d['label']}** — confidence: `{d['score']:.2f}`")

            if relationships:
                st.subheader("Scene Relationships")
                for r in relationships:
                    st.write(f"- {r}")

            st.session_state["ready_for_vqa"] = True
            st.session_state["tmp_path"] = tmp_path

        if st.session_state.get("ready_for_vqa"):
            st.subheader("Ask a Question")
            question = st.text_input("Question", placeholder="What is on the table?")
            if st.button("Ask") and question:
                with st.spinner("Thinking..."):
                    load_vqa_model()
                    answer = answer_question(st.session_state["tmp_path"], question)
                st.success(f"**Answer:** {answer}")


# ════════════════════════════════════════════════════════
# TAB 2 — Live Webcam
# ════════════════════════════════════════════════════════
with tab2:
    st.write("Click **Start Webcam** to open your camera and detect objects in real time.")
    st.info(" Detection runs every 5 frames so there will be a small delay.")

    # start/stop button using session state
    if "webcam_running" not in st.session_state:
        st.session_state["webcam_running"] = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Webcam"):
            st.session_state["webcam_running"] = True
    with col2:
        if st.button("⏹ Stop Webcam"):
            st.session_state["webcam_running"] = False

    # placeholder where webcam frames will show
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if st.session_state["webcam_running"]:
        import cv2
        from PIL import Image
        import tempfile
        import numpy as np

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam. Make sure your camera is connected.")
            st.session_state["webcam_running"] = False
        else:
            detections = []
            frame_count = 0
            FRAME_SKIP = 5

            while st.session_state["webcam_running"]:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not read from webcam.")
                    break

                frame_count += 1

                # run DETR every 5th frame
                if frame_count % FRAME_SKIP == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)

                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        pil_image.save(tmp.name)
                        tmp_path = tmp.name

                    try:
                        detections = detect(tmp_path)
                    except Exception as e:
                        st.warning(f"Detection error: {e}")

                    os.unlink(tmp_path)

                # draw boxes on frame
                COLORS = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 0, 255), (0, 255, 255), (255, 165, 0),
                ]
                for i, det in enumerate(detections):
                    color = COLORS[i % len(COLORS)]
                    x1, y1, x2, y2 = [int(v) for v in det["box"]]
                    label = f"{det['label']} {det['score']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 10, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # convert BGR to RGB for Streamlit
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # show frame inside Streamlit
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # show detected objects below frame
                if detections:
                    info_placeholder.markdown(
                        "**Detected:** " + " · ".join([f"`{d['label']}` {d['score']:.2f}" for d in detections])
                    )

            cap.release()
            frame_placeholder.empty()
            info_placeholder.empty()
            st.success("Webcam stopped.")