import os
import cv2
import configparser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='lpr_dev.log', filemode='a')
logger = logging.getLogger(__name__)


def read_config(config_path='requirements/config.ini'):
    configration = configparser.ConfigParser()
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found!")
        raise FileNotFoundError(f"Configuration file {config_path} not found!")
    configration.read(config_path)
    return configration


config = read_config()
RESIZE_FACTOR = config.getint('General', 'RESIZE_FACTOR')


def resize_plate(image, given_width, given_height, multiplier):
    if RESIZE_FACTOR != 1:
        aspect_ratio = given_width / given_height
        new_height = int(given_width * multiplier)  # multiplier to blast a large image
        new_width = int(new_height * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image
    else:
        if RESIZE_FACTOR == 0:
            print(f"Resize factor cannot be zero")
            return image
        else:
            return image


def enhance_plate(imgx):
    # Get image dimensions
    high, wide, *_ = imgx.shape
    # Resizing the number plate
    resized_img = resize_plate(imgx, wide, high, RESIZE_FACTOR)
    # cv2.imshow("Plate", resized_img)
    # Grayscale image
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    # Blurred image
    blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # Threshold image
    _, threshold_img = cv2.threshold(blurred_img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    erode_img = cv2.erode(threshold_img, (3, 3))
    dilate_img = cv2.dilate(erode_img, (3, 3))
    return dilate_img


# Example usage
# input_file = "./output_dir/2024-11-18/plates/8/20241119_150740.988128.jpg"
# output_file = "./output_dir/img.jpg"
# cv2.imwrite(output_file, enhance_plate(cv2.imread(input_file)))
