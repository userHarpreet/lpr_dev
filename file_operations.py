import os
import cv2
import logging
from datetime import datetime
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")

def get_date_folder():
    return datetime.now().strftime('%Y-%m-%d')

def get_output_dirs(output_dir):
    date_folder = get_date_folder()
    output_dir = os.path.join(output_dir, date_folder)
    plates_dir = os.path.join(output_dir, "plates")
    frames_dir = os.path.join(output_dir, "frames")
    return output_dir, plates_dir, frames_dir

def save_image(directory, filename, image):
    ensure_dir(directory)
    cv2.imwrite(os.path.join(directory, filename), image)
    logger.debug(f"Saved image: {os.path.join(directory, filename)}")

def save_blank_excel(output_file, headers):
    workbook = Workbook()
    workbook.active.append(headers)
    workbook.save(output_file)
    logger.info(f"Created blank Excel file: {output_file}")

def insert_image_to_excel(image_path, sheet, cell):
    img = XLImage(image_path)
    sheet.add_image(img, cell)

def get_sorted_image_files(directory):
    return sorted(os.listdir(directory))

def get_timestamp_from_filename(filename):
    tStamp = filename[0:13]
    pos_date = [4, 6]
    pos_date.sort()
    for i, pos in enumerate(pos_date):
        tStamp = tStamp[:pos + i] + '/' + tStamp[pos + i:]
    tStamp = tStamp.replace("_", " ")
    tStamp = tStamp[:13] + ':' + tStamp[13:]
    return tStamp