import cv2
import numpy as np

img = np.ones((400, 800, 3), dtype=np.uint8) * 255
cv2.putText(img, "DECODELABS PROJECT 4", (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
cv2.putText(img, "Optical Character Recognition Test", (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
cv2.putText(img, "Confidence Gate: 80 Percent Minimum", (40, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

cv2.imwrite("input/sample_image.jpg", img)
print("Sample test image created at input/sample_image.jpg")