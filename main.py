
import os
import sys
import cv2

from utils.preprocess import preprocess_image
from utils.detect import run_ocr, save_ocr_results

# ---- Configuration ----
INPUT_IMAGE = os.path.join("input", "sample_image.jpg")
OUTPUT_DIR = "output"
CONFIDENCE_THRESHOLD = 80  # percent, per project requirement (Gate 3)


def main():
    # 1. Validate input exists
    if not os.path.exists(INPUT_IMAGE):
        print(f"[ERROR] Input image not found at: {INPUT_IMAGE}")
        print("Place an image named 'sample_image.jpg' inside the 'input/' folder.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load original image
    original = cv2.imread(INPUT_IMAGE)
    if original is None:
        print(f"[ERROR] Failed to load image: {INPUT_IMAGE}")
        sys.exit(1)

    print(f"[INFO] Loaded image: {INPUT_IMAGE} (shape={original.shape})")

    # 3. Pre-process (grayscale -> blur -> adaptive threshold -> deskew)
    processed = preprocess_image(original)
    processed_path = os.path.join(OUTPUT_DIR, "preprocessed.jpg")
    cv2.imwrite(processed_path, processed)
    print(f"[INFO] Pre-processed image saved to: {processed_path}")

    # 4. Run OCR with confidence filtering (Gate 3: >= 80%)
    results = run_ocr(processed, confidence_threshold=CONFIDENCE_THRESHOLD)

    if not results:
        print("[WARN] No text detected above the confidence threshold.")
    else:
        print(f"[INFO] Detected {len(results)} text segment(s) above {CONFIDENCE_THRESHOLD}% confidence:")
        for text, conf in results:
            print(f"   -> '{text}'  (confidence: {conf:.2f}%)")

    # 5. Save annotated output + results.txt (Gate 4: Visual Confirmation)
    save_ocr_results(original, results, OUTPUT_DIR)
    print(f"[INFO] Final results saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()