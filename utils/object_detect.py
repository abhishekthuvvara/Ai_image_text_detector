
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the YOLOv8-nano model once and caches it across reruns."""
    return YOLO("yolov8n.pt")


def detect_objects(image_bgr, confidence_threshold=0.50):
    """
    Runs object detection on a BGR numpy image.

    Args:
        image_bgr: numpy array (H, W, 3) in BGR order
        confidence_threshold: float 0.0-1.0, minimum confidence to keep

    Returns:
        results: list of dicts [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}]
        annotated_image: numpy array (BGR) with boxes + labels drawn
    """
    model = load_model()
    predictions = model(image_bgr, verbose=False)[0]

    results = []
    annotated = image_bgr.copy()

    for box in predictions.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        results.append({
            "label": label,
            "confidence": round(conf * 100, 2),
            "box": [x1, y1, x2, y2],
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 206, 201), 2)
        caption = f"{label} ({conf * 100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 206, 201), -1)
        cv2.putText(
            annotated, caption, (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 17, 23), 2,
        )

    return results, annotated
