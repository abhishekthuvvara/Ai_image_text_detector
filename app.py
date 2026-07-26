"""
AI Vision Scanner
Open-source image text & object recognition — powered by EasyOCR + YOLOv8.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    https://share.streamlit.io  -> New app -> point at this repo -> app.py
"""

import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from utils.preprocess import preprocess_for_ocr
from utils.ocr_detect import detect_text
from utils.object_detect import detect_objects

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Vision Scanner",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — this is what makes it feel like a real product, not a demo
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0F1117; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C5CE7, #00CEC9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .hero-subtitle {
        color: #A0A0B8;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    .stat-card {
        background: #1A1D29;
        border: 1px solid #2A2D3E;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6C5CE7;
    }
    .stat-label {
        color: #A0A0B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-chip {
        display: inline-block;
        background: #1A1D29;
        border: 1px solid #2A2D3E;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        margin: 0.25rem;
        font-size: 0.9rem;
        color: #E6E6FA;
    }
    .conf-high { border-color: #00CEC9; color: #00CEC9; }
    .conf-mid  { border-color: #FDCB6E; color: #FDCB6E; }
    .conf-low  { border-color: #FF7675; color: #FF7675; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🔎 AI Vision Scanner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload a photo — extract readable text and '
    'identify objects, both powered by open-source AI models running entirely '
    'in this app.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "What do you want to detect?",
        ["Text (OCR)", "Objects", "Both"],
        index=2,
    )
    confidence_pct = st.slider(
        "Minimum confidence to display", min_value=10, max_value=95, value=50, step=5,
    )
    show_preprocessed = st.checkbox("Show pre-processing steps", value=False)

    st.markdown("---")
    st.caption(
        "Built with [EasyOCR](https://github.com/JaidedAI/EasyOCR) and "
        "[YOLOv8](https://github.com/ultralytics/ultralytics). "
        "100% open source — see the README for setup and deployment."
    )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("👆 Upload a photo to get started — try a receipt, a street sign, or a room full of objects.")
    st.stop()

# ---------------------------------------------------------------------------
# Load & convert image
# ---------------------------------------------------------------------------
pil_image = Image.open(uploaded_file).convert("RGB")
image_rgb = np.array(pil_image)
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def resize_for_speed(image, max_dim=1280):
    """Downscale large images before running detection — big speed win,
    negligible accuracy loss for typical photos."""
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


image_bgr = resize_for_speed(image_bgr)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # keep original/resized in sync for display

confidence_threshold = confidence_pct / 100.0

col_original, col_result = st.columns(2)

with col_original:
    st.subheader("Original")
    st.image(image_rgb, use_container_width=True)

# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
text_results, ocr_annotated = [], None
object_results, object_annotated = [], None

with st.spinner("Running AI models — first run downloads model weights, please wait..."):
    start = time.time()

    if mode in ("Text (OCR)", "Both"):
        text_results, ocr_annotated = detect_text(image_bgr, confidence_threshold)

    if mode in ("Objects", "Both"):
        object_results, object_annotated = detect_objects(image_bgr, confidence_threshold)

    elapsed = time.time() - start

# Pick which annotated image to show as the primary "Result" view.
# For "Both", draw object boxes on top of the already-annotated OCR image
# instead of re-running both models a second time.
if mode == "Both" and ocr_annotated is not None and object_annotated is not None:
    display_image = ocr_annotated.copy()
    for r in object_results:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 206, 201), 2)
        caption = f'{r["label"]} ({r["confidence"]:.0f}%)'
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_image, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 206, 201), -1)
        cv2.putText(display_image, caption, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 17, 23), 2)
elif ocr_annotated is not None:
    display_image = ocr_annotated
elif object_annotated is not None:
    display_image = object_annotated
else:
    display_image = image_bgr

with col_result:
    st.subheader("Detected")
    st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), use_container_width=True)

st.caption(f"⏱️ Processed in {elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
stat_cols = st.columns(3)
with stat_cols[0]:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{len(text_results)}</div>'
        f'<div class="stat-label">Text segments</div></div>',
        unsafe_allow_html=True,
    )
with stat_cols[1]:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{len(object_results)}</div>'
        f'<div class="stat-label">Objects found</div></div>',
        unsafe_allow_html=True,
    )
with stat_cols[2]:
    avg_conf = 0.0
    all_confs = [r["confidence"] for r in text_results] + [r["confidence"] for r in object_results]
    if all_confs:
        avg_conf = sum(all_confs) / len(all_confs)
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{avg_conf:.0f}%</div>'
        f'<div class="stat-label">Avg. confidence</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Detailed results — tabs
# ---------------------------------------------------------------------------
def confidence_class(conf):
    if conf >= 80:
        return "conf-high"
    if conf >= 50:
        return "conf-mid"
    return "conf-low"


tab_text, tab_objects, tab_debug = st.tabs(["📝 Text Results", "📦 Object Results", "🔧 Debug View"])

with tab_text:
    if not text_results:
        st.write("No text detected above the confidence threshold.")
    else:
        for r in text_results:
            cls = confidence_class(r["confidence"])
            st.markdown(
                f'<span class="result-chip {cls}">{r["text"]} — {r["confidence"]:.1f}%</span>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "⬇️ Download text results (.txt)",
            data="\n".join(f'{r["text"]}  ({r["confidence"]:.1f}%)' for r in text_results),
            file_name="ocr_results.txt",
        )

with tab_objects:
    if not object_results:
        st.write("No objects detected above the confidence threshold.")
    else:
        for r in object_results:
            cls = confidence_class(r["confidence"])
            st.markdown(
                f'<span class="result-chip {cls}">{r["label"]} — {r["confidence"]:.1f}%</span>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "⬇️ Download object results (.txt)",
            data="\n".join(f'{r["label"]}  ({r["confidence"]:.1f}%)' for r in object_results),
            file_name="object_results.txt",
        )

with tab_debug:
    if show_preprocessed:
        preprocessed = preprocess_for_ocr(image_bgr)
        st.image(preprocessed, caption="Grayscale -> Blur -> Adaptive Threshold -> Deskew", use_container_width=True)
    else:
        st.write("Enable 'Show pre-processing steps' in the sidebar to see the OCR pipeline stages.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("AI Vision Scanner · Open source under the MIT License · Built with Streamlit")
