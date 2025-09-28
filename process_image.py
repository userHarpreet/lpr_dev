import os
import cv2
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='lpr_dev.log', filemode='a')
logger = logging.getLogger(__name__)


load_dotenv()

RESIZE_FACTOR = float(os.getenv("RESIZE_FACTOR"))


def resize_plate(image, multiplier):
    # Get image dimensions
    given_height, given_width, *_ = image.shape
    if RESIZE_FACTOR != 1:
        aspect_ratio = given_width / given_height
        new_height = int(given_height * multiplier)  # multiplier to resize image
        new_width = int(new_height * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image
    else:
        if RESIZE_FACTOR == 0:
            print(f"Resize factor cannot be zero")
            return image
        else:
            return image


def prepare_plate_for_ocr(image):
    """Prepare a plate image for OCR: resize, convert to grayscale, apply
    CLAHE and a bilateral filter to preserve edges while reducing noise.
    Returns a 3-channel BGR image which is OCR-friendly."""
    try:
        # Resize to a reasonable working size keeping aspect
        h, w = image.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        # Reduce noise but keep edges
        filtered = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)

        # Slight adaptive thresholding can help some OCR models but keep it soft
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Convert back to BGR for downstream consumers
        ocr_ready = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return ocr_ready
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception('Failed to prepare plate for OCR, falling back to original image')
        # Fallback: ensure 3-channel
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image


def enhance_plate(imgx):
    # Resizing the number plate
    resized_img = resize_plate(imgx, RESIZE_FACTOR)
    # Grayscale image (images loaded with OpenCV are in BGR order)
    grayed_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # Blurred image
    blurred_img = cv2.GaussianBlur(grayed_img, (7, 7), 0)
    # Binary image
    _, binary_img = cv2.threshold(blurred_img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eroded_img = cv2.erode(binary_img, (3, 3))
    dilated_img = cv2.dilate(eroded_img, (3, 3))

    # cv2.imshow("Plate", imgx)
    # cv2.imshow("Resized", resized_img)
    # cv2.imshow("Grayed", grayed_img)
    # cv2.imshow("Blurred", blurred_img)
    # cv2.imshow("Binary", binary_img)
    # cv2.imshow("Eroded", eroded_img)
    # cv2.imshow("Dilated", dilated_img)
    # cv2.waitKey(1)
    # PaddleOCR / downstream code expects a 3-channel image. Convert the
    # processed single-channel image back to BGR so callers always receive
    # a consistent (H, W, 3) uint8 image. This prevents tuple index errors
    # when libraries access img.shape[2].
    try:
        bgr_img = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
    except Exception:
        # Fallback: if conversion fails, return the original dilated image
        # (downstream will handle/validate).
        logger.exception('Failed to convert processed plate to BGR, returning single-channel image')
        return dilated_img

    return bgr_img


# Example usage
# input_file = "./output_dir/2024-11-18/plates/8/20241119_150740.988128.jpg"
# output_file = "./output_dir/img.jpg"
# cv2.imwrite(output_file, enhance_plate(cv2.imread(input_file)))
