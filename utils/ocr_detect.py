"""
Text Detection Module (EasyOCR)
AI Vision Scanner

Pure pip-installable OCR — no external system binary required.
Returns per-detection bounding boxes, text, and confidence scores.
"""

import easyocr
import cv2
import numpy as np
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_reader():
    """
    Loads the EasyOCR model once and caches it across reruns/sessions.
    This is the slow part (model download + load), so caching matters
    a lot for a smooth Streamlit experience.
    """
    return easyocr.Reader(["en"], gpu=False)


def detect_text(image_bgr, confidence_threshold=0.50):
    """
    Runs OCR on a BGR numpy image.

    Args:
        image_bgr: numpy array (H, W, 3) in BGR order (OpenCV default)
        confidence_threshold: float 0.0-1.0, minimum confidence to keep

    Returns:
        results: list of dicts [{"text": str, "confidence": float, "bbox": [...]}]
        annotated_image: numpy array (BGR) with boxes + labels drawn
    """
    reader = load_reader()
    raw_results = reader.readtext(image_bgr)

    results = []
    annotated = image_bgr.copy()

    for bbox, text, conf in raw_results:
        text = text.strip()
        if text == "" or conf < confidence_threshold:
            continue

        results.append({
            "text": text,
            "confidence": round(conf * 100, 2),
            "bbox": bbox,
        })

        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(108, 92, 231), thickness=2)

        label = f"{text} ({conf * 100:.0f}%)"
        top_left = tuple(pts[0])
        cv2.putText(
            annotated, label,
            (top_left[0], max(top_left[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (108, 92, 231), 2,
        )

    return results, annotated
