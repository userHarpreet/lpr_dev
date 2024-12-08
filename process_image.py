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
RESIZE_FACTOR = config.getfloat('General', 'RESIZE_FACTOR')


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


def enhance_plate(imgx):
    # Resizing the number plate
    resized_img = resize_plate(imgx, RESIZE_FACTOR)
    # Grayscale image
    grayed_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
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
    return dilated_img


# Example usage
# input_file = "./output_dir/2024-11-18/plates/8/20241119_150740.988128.jpg"
# output_file = "./output_dir/img.jpg"
# cv2.imwrite(output_file, enhance_plate(cv2.imread(input_file)))
