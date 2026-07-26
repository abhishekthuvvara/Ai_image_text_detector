"""
Image Pre-Processing Module
AI Vision Scanner

Pipeline: Grayscale -> Gaussian Blur -> Adaptive Threshold -> Deskew
"""

import cv2
import numpy as np


def to_grayscale(image):
    """Collapse the 3D RGB matrix into a 1D intensity matrix."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    """Smooth the image to eliminate micro-imperfections and noise."""
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def apply_adaptive_threshold(blurred_image):
    """Force every pixel to a binary decision using Otsu's method."""
    _, thresholded = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def deskew(image):
    """Calculate rotation angle of text/content and straighten it."""
    coords = np.column_stack(np.where(image > 0))

    if coords.shape[0] == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_for_ocr(image):
    """
    Full pre-processing pipeline optimized for text recognition.
    Input: original BGR image (numpy array)
    Output: deskewed, thresholded, single-channel binary image
    """
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    thresholded = apply_adaptive_threshold(blurred)
    deskewed = deskew(thresholded)
    return deskewed
