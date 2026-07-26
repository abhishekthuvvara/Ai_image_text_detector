
import os
import cv2
import easyocr

# Reader is created once and reused (loading the model is the slow part)
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        # gpu=False works everywhere; set gpu=True if you have CUDA set up
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def run_ocr(processed_image, confidence_threshold=80):
    """
    Runs EasyOCR on a pre-processed image and returns only the text
    segments that meet the minimum confidence threshold.

    Args:
        processed_image: image as a numpy array (grayscale or BGR both work)
        confidence_threshold: minimum confidence percent (0-100) to accept

    Returns:
        List of tuples: [(text, confidence), ...]
    """
    reader = _get_reader()

    # EasyOCR expects a 3-channel image; convert if input is single-channel
    if len(processed_image.shape) == 2:
        image_for_ocr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        image_for_ocr = processed_image

    raw_results = reader.readtext(image_for_ocr)  # [(bbox, text, conf), ...]

    results = []
    for bbox, text, conf in raw_results:
        confidence_pct = conf * 100  # EasyOCR returns 0.0-1.0
        text = text.strip()

        if text == "":
            continue

        if confidence_pct >= confidence_threshold:
            results.append((text, confidence_pct))

    return results


def save_ocr_results(original_image, results, output_dir):
    """
    Saves two deliverables for Gate 4 (Visual Confirmation):
      1. results.txt  -> plain formatted text output
      2. result_image.jpg -> original image annotated with detected text
    """
    os.makedirs(output_dir, exist_ok=True)

    text_path = os.path.join(output_dir, "results.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        if not results:
            f.write("No text detected above the confidence threshold.\n")
        else:
            f.write("Project 4 - OCR Recognition Results (EasyOCR)\n")
            f.write("=" * 40 + "\n\n")
            for text, conf in results:
                f.write(f"Text: {text}\n")
                f.write(f"Confidence: {conf:.2f}%\n")
                f.write("-" * 40 + "\n")

    annotated = original_image.copy()
    banner_height = 40 + (25 * max(len(results), 1))
    h, w = annotated.shape[:2]

    canvas = cv2.copyMakeBorder(
        annotated, 0, banner_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    y_offset = h + 30
    if not results:
        cv2.putText(
            canvas, "No text detected", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    else:
        for text, conf in results:
            label = f"{text} ({conf:.1f}%)"
            cv2.putText(
                canvas, label, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2,
            )
            y_offset += 25

    image_path = os.path.join(output_dir, "result_image.jpg")
    cv2.imwrite(image_path, canvas)

    print(f"[INFO] Text results written to: {text_path}")
    print(f"[INFO] Annotated image written to: {image_path}")