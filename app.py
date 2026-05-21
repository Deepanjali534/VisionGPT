import streamlit as st
import tempfile
import os
from visiongpt.pipeline.detector import detect
from visiongpt.pipeline.scene_graph import build_scene_graph
from visiongpt.pipeline.vqa import load_vqa_model, answer_question
from visiongpt.utils.visualizer import visualize

st.set_page_config(page_title="VisionGPT", layout="wide")
st.title("VisionGPT — Intelligent Object Interaction Agent")

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
