"""
Multimodal Deepfake Detection System
=====================================
A Streamlit app for detecting deepfakes across image, audio, and video modalities.

To plug in real model inference:
  - Replace the predict_* functions below with your actual model loading and inference logic.
  - Each function receives a file-like object (BytesIO or UploadedFile) and should return
    a dict with keys: "label" (str: "Real" or "Fake") and "confidence" (float: 0.0–1.0).
"""
from predict import predict_image,predict_audio,predict_video,predict_video_audio
import time
import streamlit as st


# ──────────────────────────────────────────────
#  PREDICTION FUNCTIONS (mock implementations)
# ──────────────────────────────────────────────

def predict_image(file) -> dict:
    """
    Predict whether an image is real or fake.

    TODO: Replace the body of this function with your image deepfake model.
          Example:
              model = load_image_model("weights/image_model.pt")
              tensor = preprocess(file)
              output = model(tensor)
              label = "Fake" if output > 0.5 else "Real"
              return {"label": label, "confidence": float(output)}
    """
    # --- Mock prediction ---
    time.sleep(1.2)  # Simulate inference latency
    return {"label": "Fake", "confidence": 0.87}


def predict_audio(file) -> dict:
    """
    Predict whether an audio clip is real or fake (voice cloning / TTS detection).

    TODO: Replace with your audio deepfake model.
          Example:
              features = extract_mfcc(file)
              output = audio_model.predict(features)
              return {"label": "Real" if output < 0.5 else "Fake", "confidence": float(output)}
    """
    time.sleep(1.0)
    return {"label": "Real", "confidence": 0.91}


def predict_video(file) -> dict:
    """
    Predict whether a video (no audio) is real or fake using visual frames only.

    TODO: Replace with your video deepfake model (frame-level or clip-level).
          Example:
              frames = extract_frames(file, fps=1)
              embeddings = vision_encoder(frames)
              output = classifier(embeddings.mean(0))
              return {"label": ..., "confidence": ...}
    """
    time.sleep(1.5)
    return {"label": "Fake", "confidence": 0.93}


def predict_video_audio(file) -> dict:
    """
    Predict whether a video (with audio) is real or fake using both modalities.

    TODO: Replace with your multimodal (video + audio) deepfake model.
          Example:
              frames   = extract_frames(file)
              waveform = extract_audio(file)
              v_emb    = vision_encoder(frames)
              a_emb    = audio_encoder(waveform)
              fused    = fusion_layer(v_emb, a_emb)
              output   = classifier(fused)
              return {"label": ..., "confidence": ...}
    """
    time.sleep(2.0)
    return {"label": "Real", "confidence": 0.76}


# ──────────────────────────────────────────────
#  UI HELPER FUNCTIONS
# ──────────────────────────────────────────────

def render_file_uploader(input_type: str):
    """Render the appropriate file uploader widget based on selected input type."""
    if input_type == "Image":
        return st.file_uploader(
            "Upload an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
    elif input_type == "Audio":
        return st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3"],
            help="Supported formats: WAV, MP3"
        )
    else:  # Video (with or without audio)
        return st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov"],
            help="Supported formats: MP4, AVI, MOV"
        )


def render_preview(input_type: str, uploaded_file):
    """Render a preview of the uploaded file (image / audio player / video player)."""
    st.markdown("#### 🔍 Preview")

    if input_type == "Image":
        st.image(uploaded_file, use_container_width=True, caption=uploaded_file.name)

    elif input_type == "Audio":
        st.audio(uploaded_file, format="audio/wav")
        st.caption(f"📄 File: **{uploaded_file.name}**")

    else:  # Video (with or without audio)
        st.video(uploaded_file)
        st.caption(f"📄 File: **{uploaded_file.name}**")


def render_result(result: dict):
    """Render the prediction result with label and confidence score."""
    label      = result["label"]
    confidence = result["confidence"]
    pct        = confidence * 100

    # Choose colour theme based on label
    is_fake    = label == "Fake"
    badge_color = "#FF4B4B" if is_fake else "#21C354"
    icon        = "🚨" if is_fake else "✅"

    st.markdown("#### 📊 Detection Result")

    # Styled result card
    st.markdown(
        f"""
        <div style="
            background: {'#2d1b1b' if is_fake else '#1b2d1e'};
            border: 2px solid {badge_color};
            border-radius: 12px;
            padding: 24px 28px;
            margin-top: 8px;
        ">
            <p style="font-size: 2rem; margin: 0; font-weight: 700; color: {badge_color};">
                {icon} {label}
            </p>
            <p style="font-size: 1rem; color: #cccccc; margin: 6px 0 0 0;">
                Confidence: <strong style="color: {badge_color};">{pct:.1f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar as a visual confidence indicator
    st.markdown("")
    st.markdown(f"**Confidence score:** `{pct:.1f}%`")
    st.progress(confidence)

    # Risk interpretation note
    if pct >= 85:
        risk_msg = "⚠️ **High confidence** — very likely a deepfake." if is_fake else "✅ **High confidence** — very likely authentic."
    elif pct >= 60:
        risk_msg = "🟡 **Moderate confidence** — manual review recommended."
    else:
        risk_msg = "🔵 **Low confidence** — result is inconclusive."

    st.info(risk_msg)


# ──────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────

def main():
    # ── Page config ──────────────────────────
    st.set_page_config(
        page_title="Multimodal Deepfake Detection",
        page_icon="🕵️",
        layout="centered",
    )

    # ── Header ───────────────────────────────
    st.title("🕵️ Multimodal Deepfake Detection")
    st.markdown(
        "Detect AI-generated or manipulated media across **image**, **audio**, and **video** modalities."
    )
    st.divider()

    # ── Sidebar: About ───────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            This tool analyses uploaded media files and predicts whether they
            are **Real** or **Fake** (AI-generated / deepfake).

            **Supported modalities:**
            - 🖼️ Image (JPG, PNG)
            - 🔊 Audio (WAV, MP3)
            - 🎬 Video without audio (MP4, AVI, MOV)
            - 🎥 Video with audio (MP4, AVI, MOV)

            ---
            > **Note:** Prediction functions currently return mock results.
            > Plug in your own models to enable real inference.
            """
        )
        st.divider()
        st.markdown("**Model status:** `Mock / Demo mode`")

    # ── Step 1: Select input type ─────────────
    st.subheader("Step 1 — Select Input Type")
    input_type = st.radio(
        "What kind of media do you want to analyse?",
        options=["Image", "Audio", "Video (without audio)", "Video (with audio)"],
        horizontal=True,
    )

    # Normalise label for branching logic
    input_type_key = input_type.replace(" (without audio)", "").replace(" (with audio)", "")
    is_video_with_audio = input_type == "Video (with audio)"

    st.divider()

    # ── Step 2: Upload file ────────────────────
    st.subheader("Step 2 — Upload File")
    uploaded_file = render_file_uploader(input_type_key)

    if uploaded_file is not None:
        st.divider()

        # ── Step 3: Preview ───────────────────
        st.subheader("Step 3 — Preview")
        render_preview(input_type_key, uploaded_file)

        st.divider()

        # ── Step 4: Predict ────────────────────
        st.subheader("Step 4 — Run Detection")

        predict_btn = st.button(
            "🔍 Predict",
            type="primary",
            use_container_width=True,
        )

        if predict_btn:
            with st.spinner("Analysing media… please wait."):
                # Route to the correct prediction function
                if input_type_key == "Image":
                    result = predict_image(uploaded_file)

                elif input_type_key == "Audio":
                    result = predict_audio(uploaded_file)

                elif input_type_key == "Video" and not is_video_with_audio:
                    result = predict_video(uploaded_file)

                else:  # Video with audio
                    result = predict_video_audio(uploaded_file)

            st.divider()
            render_result(result)

    else:
        # Friendly prompt when nothing is uploaded yet
        st.markdown("")
        st.markdown(
            "<div style='text-align:center; color: #888; padding: 40px 0;'>"
            "⬆️ Upload a file above to get started."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()