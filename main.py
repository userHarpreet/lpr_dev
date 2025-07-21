import base64
import os
import re

import cv2
from datetime import datetime, timedelta
import multiprocessing as mp
import logging

import psutil
import pytesseract
import schedule
import configparser
import time
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from ultralytics import YOLO

from process_image import enhance_plate, resize_plate
from validate_number import validate_hsrp, validate_and_format_plate
from crop_images import crop_images_in_folder

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

# Load configuration
config = read_config()
SHOW_LIVE = config.getboolean('General', 'SHOW_LIVE')
PLATE_CONF_MIN = config.getfloat('General', 'PLATE_CONF_MIN')
VEHICLE_CONF_MIN = config.getfloat('General', 'VEHICLE_CONF_MIN')
VIDEO_SOURCE = config.get('General', 'VIDEO_SOURCE')
RESIZE_FACTOR = config.getfloat('General', 'RESIZE_FACTOR')
TIME_FORMAT = config.get('General', 'TIME_FORMAT')
OUTPUT_DIR = config.get('General', 'OUTPUT_DIR')
VEHICLE_CLASSES = [int(cls) for cls in config.get('General', 'VEHICLE_CLASSES').split(',')]
HTML_HEADERS = config.get('HTML', 'HEADERS').split(',')
# Configurable watchdog interval (in seconds) from config.ini
WATCHDOG_INTERVAL = config.getint('General', 'WATCHDOG_INTERVAL')

# Initialize models
try:
    vehicle_model = YOLO(config.get('Models', 'VEHICLE_MODEL_PATH'))
    plate_model = YOLO(config.get('Models', 'PLATE_MODEL_PATH'))
except Exception as e:
    logger.error(f"Error loading YOLO models: {e}")
    raise

logger.info("Models loaded successfully")

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def get_date_folder():
    return datetime.now().strftime('%Y-%m-%d')


def get_output_dirs():
    date_folder = get_date_folder()
    output_dir = os.path.join(OUTPUT_DIR, date_folder)
    plates_dir = os.path.join(output_dir, "plates")
    frames_dir = os.path.join(output_dir, "frames")
    return output_dir, plates_dir, frames_dir


def recognize_plate(plate_img):
    """
    Enhanced license plate recognition that handles both single-line and 2-line plates.
    Supports motorcycle/2-wheeler plates with 2-line format.
    """
    import pytesseract
    
    logger.info("Starting enhanced license plate recognition with Tesseract")

    # Log image properties
    img_height, img_width = plate_img.shape[:2] if len(plate_img.shape) >= 2 else (0, 0)
    img_size_kb = plate_img.size * plate_img.itemsize / 1024
    logger.info(f"Image dimensions: {img_width}x{img_height}, Size: {img_size_kb:.2f} KB")

    # Log memory usage before OCR
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory usage before OCR: {mem_before:.2f} MB")

    start_time = time.time()

    try:
        # Detect if this might be a 2-line plate (height > width indicates vertical layout)
        aspect_ratio = img_height / img_width if img_width > 0 else 1
        is_likely_two_line = aspect_ratio > 0.7  # Threshold for 2-line detection
        
        logger.info(f"Aspect ratio: {aspect_ratio:.2f}, Likely 2-line plate: {is_likely_two_line}")
        
        # Try multiple OCR configurations
        ocr_results = []
        
        # Configuration 1: Single line (PSM 7)
        config_single = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        try:
            text_single = pytesseract.image_to_string(plate_img, config=config_single).strip()
            if text_single:
                ocr_results.append((text_single.replace(' ', ''), 'single_line', len(text_single.replace(' ', ''))))
                logger.info(f"Single line OCR result: '{text_single}'")
        except Exception as e:
            logger.warning(f"Single line OCR failed: {e}")
        
        # Configuration 2: Multiple lines (PSM 6) - for 2-line plates
        config_multi = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        try:
            text_multi = pytesseract.image_to_string(plate_img, config=config_multi).strip()
            if text_multi:
                # Process multi-line result
                lines = [line.strip() for line in text_multi.split('\n') if line.strip()]
                if len(lines) >= 2:
                    # Combine lines for 2-line motorcycle plate
                    combined_text = ''.join(lines).replace(' ', '')
                    ocr_results.append((combined_text, 'multi_line', len(combined_text)))
                    logger.info(f"Multi-line OCR result: {lines} -> '{combined_text}'")
                elif len(lines) == 1:
                    single_line_text = lines[0].replace(' ', '')
                    ocr_results.append((single_line_text, 'multi_as_single', len(single_line_text)))
        except Exception as e:
            logger.warning(f"Multi-line OCR failed: {e}")
        
        # Configuration 3: Block of text (PSM 8) - alternative approach
        if is_likely_two_line:
            config_block = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            try:
                text_block = pytesseract.image_to_string(plate_img, config=config_block).strip()
                if text_block:
                    # Process block result
                    lines = [line.strip() for line in text_block.split('\n') if line.strip()]
                    if len(lines) >= 2:
                        combined_text = ''.join(lines).replace(' ', '')
                        ocr_results.append((combined_text, 'block_multi', len(combined_text)))
                        logger.info(f"Block OCR result: {lines} -> '{combined_text}'")
            except Exception as e:
                logger.warning(f"Block OCR failed: {e}")
        
        # Log processing time
        elapsed_time = time.time() - start_time
        logger.info(f"Enhanced OCR completed in {elapsed_time:.2f} seconds")

        # Log memory usage after OCR
        mem_after = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory usage after OCR: {mem_after:.2f} MB (Change: {mem_after - mem_before:.2f} MB)")

        if not ocr_results:
            logger.warning("All OCR configurations returned no results")
            return None, None
        
        # Select best result based on length and format
        best_result = select_best_ocr_result(ocr_results, is_likely_two_line)
        
        if best_result:
            text, method, length = best_result
            logger.info(f"Selected best OCR result: '{text}' (method: {method}, length: {length})")
            return text, 1.0  # Return text with confidence 1.0
        else:
            logger.warning("No valid OCR result found")
            return None, None

    except Exception as ex:
        elapsed_time = time.time() - start_time
        logger.error(f"Error in enhanced plate recognition after {elapsed_time:.2f} seconds: {ex}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error in plate recognition: {ex}")
        return None, None


def select_best_ocr_result(ocr_results, is_likely_two_line):
    """
    Select the best OCR result from multiple attempts.
    Prioritizes results that match expected Indian license plate formats.
    """
    if not ocr_results:
        return None
    
    # Score each result
    scored_results = []
    
    for text, method, length in ocr_results:
        score = 0
        
        # Length scoring (Indian plates are typically 7-10 characters)
        if 7 <= length <= 10:
            score += 100
        elif 6 <= length <= 11:
            score += 80
        elif length >= 5:
            score += 60
        else:
            score += 20
        
        # Format pattern scoring
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$', text):  # Standard format: DL01AB1234
            score += 200
        elif re.match(r'^\d{2}BH\d{4}[A-Z]{1,3}$', text):     # Bharat series: 22BH1234AB
            score += 200
        elif re.match(r'^[A-Z]{2}VA[A-Z]{0,3}\d{4}$', text):   # Vintage series: DLVA1234
            score += 200
        elif re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$', text):  # Partial match
            score += 150
        elif re.match(r'^[A-Z]+\d+$', text) or re.match(r'^\d+[A-Z]+$', text):  # Has both letters and numbers
            score += 100
        
        # Method preference
        if is_likely_two_line:
            if method in ['multi_line', 'block_multi']:
                score += 50  # Prefer multi-line methods for 2-line plates
            elif method == 'single_line':
                score += 30  # Still valid but less preferred
        else:
            if method == 'single_line':
                score += 50  # Prefer single-line for regular plates
            elif method in ['multi_line', 'block_multi']:
                score += 40
        
        # Character validity (only alphanumeric)
        if text.isalnum():
            score += 30
        
        scored_results.append((score, text, method, length))
        logger.debug(f"OCR result '{text}' scored {score} points (method: {method})")
    
    # Sort by score (highest first)
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    if scored_results:
        best_score, best_text, best_method, best_length = scored_results[0]
        logger.info(f"Best OCR result: '{best_text}' with score {best_score} (method: {best_method})")
        return (best_text, best_method, best_length)
    
    return None


def save_image(directory, filename, image):
    ensure_dir(directory)
    save_to_file = os.path.join(directory, filename)
    cv2.imwrite(str(save_to_file), image)
    logger.debug(f"Saved image: {os.path.join(directory, filename)}")


def get_plate(photo, obj_id):
    timestamp = datetime.now().strftime(TIME_FORMAT)
    _, plates_dir, _ = get_output_dirs()
    detections = plate_model(source=photo, conf=PLATE_CONF_MIN, show=SHOW_LIVE)
    for detection in detections:
        if detection.boxes.data.numel() > 0:
            for xx, yx, xy, yy, detected_conf, _ in detection.boxes.data.tolist():
                plate = photo[int(yx):int(yy), int(xx):int(xy)]
                save_image(os.path.join(plates_dir, str(obj_id)), f"{timestamp}.jpg", plate)
                logger.info(f"Plate detected for object {obj_id} at {timestamp}")
                return timestamp
    logger.debug(f"No plate detected for object {obj_id}")
    return timestamp


def process_frame(frame, result):
    _, plates_dir, frames_dir = get_output_dirs()
    for obj in result.boxes.data.tolist():
        try:
            x1, y1, x2, y2, obj_id, conf, obj_class = map(float, obj)
            if int(obj_class) in VEHICLE_CLASSES:
                vehicle = frame[int(y1):int(y2), int(x1):int(x2)]
                timestamp = get_plate(vehicle, int(obj_id))
                save_image(os.path.join(frames_dir, str(int(obj_id))), f"{timestamp}.jpg", vehicle)
                logger.info(f"Processed frame for object {int(obj_id)} at {timestamp}")
        except ValueError:
            logger.warning(f"Skipping object due to unexpected data format: {obj}")
            continue


def vehicle_detection(frame_queue, result_queue):
    logger.info("Starting vehicle detection process")
    for result in vehicle_model.track(source=VIDEO_SOURCE, conf=VEHICLE_CONF_MIN, stream=True, show=SHOW_LIVE):
        if result is not None:
            frame_queue.put((result.orig_img, result))
        if not result_queue.empty():
            result_queue.get()


def plate_detection(frame_queue, result_queue):
    logger.info("Starting plate detection process")
    while True:
        frame, result = frame_queue.get()
        if frame is None:
            break
        process_frame(frame, result)
        result_queue.put(result)


def start_vehicle_process(frame_queue, result_queue):
    p = mp.Process(target=vehicle_detection, args=(frame_queue, result_queue))
    p.daemon = True
    p.start()
    logger.info(f"Vehicle detection process started with PID {p.pid}")
    return p


def start_plate_process(frame_queue, result_queue):
    p = mp.Process(target=plate_detection, args=(frame_queue, result_queue))
    p.daemon = True
    p.start()
    logger.info(f"Plate detection process started with PID {p.pid}")
    return p


def create_html_table(data, output_file, t_detect, t_read):
    serial = 1
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Detection Results</title>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Vehicle Detection Results</h1>"""
    html_content+= (f"<h2>Total Detections:{t_detect - 1}, Total OCR Output:{t_read - 1}</h2>"
                    f"<table>"
                    f"<tr>")
    for header in HTML_HEADERS:
        html_content += f"<th>{header}</th>"
    html_content += "</tr>"
    for row in data:

        html_content += "<tr>"
        html_content += f"<td>{serial}</td>"
        for i, cell in enumerate(row):
            # Check if this is the column containing the image path
            if i == 3:  # Assuming the image path is in the 4th column (index 3)
                if cell != "":
                    html_content += f'<td><img src="{cell[36:]}" alt="Image"></td>'
                else:
                    html_content += f"<td>{cell}</td>"
            elif i == 4:
                _, img_encoded = cv2.imencode('.jpg', cell)
                img_bytes = img_encoded.tobytes()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                html_content += f'<td><img src="data:image/jpeg;base64,{img_base64}" alt="Enhanced Image"></td>'
            else:
                html_content += f"<td>{cell}</td>"
        html_content += "</tr>"

        serial += 1
    html_content += """
        </table>
    </body>
    </html>
    """
    with open(output_file, 'w') as f:
        f.write(html_content)
    logger.info(f"HTML file created: {output_file}")
    print(f"HTML file created: {output_file}")


def send_email_with_attachment(configration, filename):
    sender_email = configration.get('Email', 'SENDER_EMAIL')
    to_emails = configration.get('Email', 'TO_EMAILS').split(',')
    cc_emails = configration.get('Email', 'CC_EMAILS').split(',') if configration.get('Email', 'CC_EMAILS') else []
    password = configration.get('Email', 'PASSWORD')
    subject = configration.get('Email', 'SUBJECT')
    body1 = configration.get('Email', 'BODY1')
    body2 = configration.get('Email', 'BODY2')
    smtp_server = configration.get('SMTP', 'HOST')
    smtp_port = configration.getint('SMTP', 'PORT')

    logger.info('Preparing to send email...')
    logger.info('Sender: %s', sender_email)
    logger.info('To: %s', ', '.join(to_emails))
    logger.info('Cc: %s', ', '.join(cc_emails))
    logger.info('Subject: %s', subject)
    # logger.info('Attachment: %s', filename)

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(to_emails)
    message["Cc"] = ", ".join(cc_emails)
    message["Subject"] = subject
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    body = body1 + yesterday + " " + body2

    message.attach(MIMEText(body, "plain"))
    logger.debug('Email body attached')

    text = message.as_string()
    all_recipients = to_emails + cc_emails
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            logger.info('Connecting to SMTP server...')
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender_email, password)
            logger.info('Logged in successfully')
            server.sendmail(sender_email, all_recipients, text)
            logger.info('Email sent successfully!')
    except smtplib.SMTPException as ex:
        logger.error('An error occurred while sending the email: %s', ex)
        raise

    logger.info('Email sending process completed')
    print('Email sending process completed')


def run_ocr_and_save_to_html(date):
    logger.info(f"Starting OCR process for {date}")

    # Input validation
    if not date:
        logger.error("Date parameter is required")
        return

    input_dir = os.path.join(OUTPUT_DIR, date)
    plates_dir = os.path.join(input_dir, "plates")
    output_file = os.path.join(input_dir, "index.html")

    # Validate directories exist
    if not os.path.exists(plates_dir):
        logger.error(f"Plates directory not found: {plates_dir}")
        return
    os.rename(plates_dir, f"{plates_dir}_org")
    crop_images_in_folder(f"{plates_dir}_org", plates_dir)

    data = []
    total_runs = 0
    total_not_read = 0
    total_detections = len(os.listdir(plates_dir))

    try:
        for obj_id in os.listdir(plates_dir):
            logger.info(f"Object processed by OCR: {total_runs}/{total_detections} out of which {total_not_read} are unable to read by OCR.")
            print(f"Object processed by OCR: {total_runs}/{total_detections} out of which {total_not_read} are unable to read by OCR.")
            total_runs += 1
            obj_dir = os.path.join(plates_dir, obj_id)
            if not os.path.isdir(obj_dir):
                continue

            results = []
            result_appended = False
            try:
                image_files = sorted(os.listdir(obj_dir))
            except OSError:
                logger.error(f"Error reading directory: {obj_dir}")
                continue

            if not image_files:
                logger.error(f'Object folder is empty: {obj_dir}')
                continue

            if len(image_files) > 1:
                indices_to_process = list(range(len(image_files)))
                first_half = indices_to_process[:len(indices_to_process) // 2]
                second_half = indices_to_process[len(indices_to_process) // 2:]
                final_indices = second_half + first_half[::-1]

                image_file = ""

                for index in final_indices:
                    try:
                        image_file = image_files[index]
                        image_path = os.path.join(obj_dir, image_file)
                        plate_img = cv2.imread(image_path)

                        if plate_img is None:
                            logger.error(f"Failed to load image: {image_path}")
                            continue

                        plate_img = enhance_plate(plate_img)
                        text, confidence = recognize_plate(plate_img)

                        # Remove spaces from the plate number and change to all CAPS
                        text = text.replace(" ", "").upper()

                        corrected_plate, is_valid, message = validate_and_format_plate(text)

                        logger.info(f"Corrected result: Text='{corrected_plate}'")
                        text = corrected_plate
                        logger.info(f"File Processed for OCR: Text: {text}, Conf: {confidence}, File: {image_path}, "
                                    f"Status:{is_valid}, MSG:{message}")
                        if text is not None:
                            if is_valid:
                                final_image = resize_plate(plate_img, 1 / RESIZE_FACTOR)
                                results.append((text, confidence, image_file, image_path, final_image))
                                result_appended = True

                    except Exception as ex:
                        logger.error(f"Error processing image: {str(ex)}")
                        continue
                if not result_appended:
                    img_file = image_files[second_half[0]]
                    img_path = os.path.join(obj_dir, str(img_file))
                    enhanced_image = enhance_plate(cv2.imread(img_path))
                    final_image = resize_plate(enhanced_image, 1 / RESIZE_FACTOR)
                    results.append(("", 0, img_file, img_path, final_image))

                    # results.append(("", 0, image_file, os.path.join(obj_dir, image_files[second_half[0]],
                    #                 enhance_plate(cv2.imread(os.path.join(obj_dir, image_files[second_half[0]]))))))
                    logger.error(f"Unable to run OCR on object: Object ID {obj_id}")
                    total_not_read += 1

            # Process results

            if results:

                conf_array = [row[1] for row in results]
                float_array = [float(x) for x in conf_array]
                index_max = float_array.index(max(float_array))

                text_captured, confidence_captured, image_file_captured, image_file_path, processed_img = results[index_max]

                confidence_captured = round(confidence_captured, 2)

                tStamp = get_timestamp_from_filename(image_file_captured)
                row = [
                    obj_id,
                    tStamp,
                    "Yes" if text_captured else "No",
                    image_file_path if image_file_path is not None else "",
                    processed_img,
                    confidence_captured if confidence_captured is not None else "",
                    text_captured if text_captured is not None else ""
                ]
                data.append(row)
            else:
                logger.error(f'No valid plate found for object: {obj_id}')

        sorted_data = sorted(data, key=lambda x: int(x[0]))

        # Save to HTML (missing implementation)
        create_html_table(sorted_data, output_file, total_detections, total_detections - total_not_read)
        send_email_with_attachment(config, output_file)
        print(f"OCR process completed and results saved to HTML for {date}")
        logger.info(f"OCR process completed and results saved to HTML for {date}")

    except Exception as ex:
        logger.error(f"Error in OCR process: {str(ex)}")
        raise


def get_timestamp_from_filename(filename):
    if filename != "":
        t_stamp = filename[0:13]
        pos_date = [4, 6]
        pos_date.sort()
        for i, pos in enumerate(pos_date):
            t_stamp = t_stamp[:pos + i] + '/' + t_stamp[pos + i:]
        t_stamp = t_stamp.replace("_", " ")
        t_stamp = t_stamp[:13] + ':' + t_stamp[13:]
        return t_stamp
    else:
        return ""


def scheduled_job():
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    ocr_process = mp.Process(target=run_ocr_and_save_to_html, args=(yesterday,))
    ocr_process.start()
    ocr_process.join()


def main():
    logger.info("Starting main process")
    ensure_dir(OUTPUT_DIR)
    frame_queue = mp.Queue()
    result_queue = mp.Queue()

    # Initial process start
    vehicle_process = start_vehicle_process(frame_queue, result_queue)
    plate_process = start_plate_process(frame_queue, result_queue)

    time_stamp = config.get('Schedule', 'JOB_TIME')

    # Schedule the OCR job to run daily
    schedule.every().day.at(time_stamp).do(scheduled_job)

    # Main loop with watchdog
    while True:
        schedule.run_pending()

        if not vehicle_process.is_alive():
            logger.error("Vehicle detection process died. Restarting...")
            vehicle_process = start_vehicle_process(frame_queue, result_queue)

        if not plate_process.is_alive():
            logger.error("Plate detection process died. Restarting...")
            plate_process = start_plate_process(frame_queue, result_queue)

        time.sleep(WATCHDOG_INTERVAL)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
