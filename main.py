import base64
import os

import cv2
from datetime import datetime, timedelta
import multiprocessing as mp
import logging
import sys

import psutil
from paddleocr import PaddleOCR
import schedule
from dotenv import load_dotenv
import time
import shutil
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from ultralytics import YOLO

from process_image import enhance_plate, resize_plate, prepare_plate_for_ocr
from validate_number import validate_and_format_plate
from crop_images import crop_images_in_folder

import re


def _parse_ocr_lines(result):
    """Return list of (text, confidence, box, x_center) from PaddleOCR result.

    This parser handles multiple shapes returned by different OCR APIs:
    - Paddle/PaddleX style dict with 'rec_texts' and 'rec_scores'
    - list-of-lines where each line is [box, [text, conf]]
    - list-of-dicts where each dict contains 'text'/'score' and optional box

    The function returns an empty list on any unexpected structure.
    """
    lines = []
    try:
        if not result:
            return lines

        # Normalize to 'first' item when result is a list
        first = result[0] if isinstance(result, list) and len(result) > 0 else result

        # Case 1: Paddle/PaddleX document style: first is dict with rec_texts/rec_scores
        if isinstance(first, dict):
            # direct rec_texts + rec_scores
            if 'rec_texts' in first and 'rec_scores' in first:
                for t, s in zip(first.get('rec_texts', []), first.get('rec_scores', [])):
                    text = t or ''
                    conf = s
                    box = None
                    x_center = None
                    lines.append((text, conf, box, x_center))
                return lines

            # Try common keys that hold line-level results
            candidate = None
            for key in ('rec_res', 'predictions', 'lines'):
                if key in first and isinstance(first[key], list):
                    candidate = first[key]
                    break

            # If still nothing, try to find any list-like value that looks like lines
            if candidate is None:
                for v in first.values():
                    if isinstance(v, list) and len(v) > 0:
                        candidate = v
                        break

            if candidate is not None:
                for item in candidate:
                    # item can be [box, [text, conf]]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        box = item[0]
                        rec = item[1]
                        if isinstance(rec, (list, tuple)):
                            text = rec[0] if len(rec) > 0 else ''
                            conf = rec[1] if len(rec) > 1 else None
                        elif isinstance(rec, dict):
                            text = rec.get('text', '')
                            conf = rec.get('score') or rec.get('confidence')
                        else:
                            text = str(rec)
                            conf = None
                        x_center = None
                        try:
                            if box and hasattr(box, '__iter__'):
                                xs = [float(pt[0]) for pt in box]
                                x_center = sum(xs) / len(xs)
                        except Exception:
                            x_center = None
                        lines.append((text, conf, box, x_center))
                    elif isinstance(item, dict):
                        text = item.get('text', '') or item.get('rec_text', '')
                        conf = item.get('score') or item.get('confidence')
                        box = item.get('box') or item.get('bbox') or item.get('points')
                        x_center = None
                        try:
                            if box and hasattr(box, '__iter__'):
                                xs = [float(pt[0]) for pt in box]
                                x_center = sum(xs) / len(xs)
                        except Exception:
                            x_center = None
                        lines.append((text, conf, box, x_center))
                    elif isinstance(item, str):
                        lines.append((item, None, None, None))
                return lines

        # Case 2: result is directly a list of lines ([box, [text, conf]] or dicts)
        if isinstance(result, list):
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    box = item[0]
                    rec = item[1]
                    if isinstance(rec, (list, tuple)):
                        text = rec[0] if len(rec) > 0 else ''
                        conf = rec[1] if len(rec) > 1 else None
                    elif isinstance(rec, dict):
                        text = rec.get('text', '')
                        conf = rec.get('score') or rec.get('confidence')
                    else:
                        text = str(rec)
                        conf = None
                    x_center = None
                    try:
                        if box and hasattr(box, '__iter__'):
                            xs = [float(pt[0]) for pt in box]
                            x_center = sum(xs) / len(xs)
                    except Exception:
                        x_center = None
                    lines.append((text, conf, box, x_center))
                elif isinstance(item, dict):
                    text = item.get('text', '') or item.get('rec_text', '')
                    conf = item.get('score') or item.get('confidence')
                    box = item.get('box') or item.get('bbox') or item.get('points')
                    x_center = None
                    try:
                        if box and hasattr(box, '__iter__'):
                            xs = [float(pt[0]) for pt in box]
                            x_center = sum(xs) / len(xs)
                    except Exception:
                        x_center = None
                    lines.append((text, conf, box, x_center))
                elif isinstance(item, str):
                    lines.append((item, None, None, None))
        return lines
    except Exception:
        return []

# Set up logging: write to both a file (persisted to /app/logs) and stdout so
# Docker's logs capture the messages (visible with `docker compose logs`).
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Ensure logs directory exists (inside container this is a mounted volume)
LOG_DIR = os.getenv('LOG_DIR', '/app/logs')
formatter = logging.Formatter(LOG_FORMAT)

# Add stream handler first so we always have console logs even if file logging fails
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'lpr_dev.log'), mode='a')
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as fh_ex:
        # If file handler fails, keep running with stream logs only
        logger.warning('Failed to create file handler for logs (%s). Continuing with stdout only.', fh_ex)
        LOG_DIR = '.'
except Exception as dir_ex:
    # If we cannot create the directory (e.g., permission), fallback to stdout only
    logger.warning('Failed to create log directory %s (%s). Continuing with stdout only.', LOG_DIR, dir_ex)
    LOG_DIR = '.'


load_dotenv()

# Load configuration
SHOW_LIVE = os.getenv("SHOW_LIVE") == "True"
PLATE_CONF_MIN = float(os.getenv("PLATE_CONF_MIN"))
VEHICLE_CONF_MIN = float(os.getenv("VEHICLE_CONF_MIN"))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE")
RESIZE_FACTOR = float(os.getenv("RESIZE_FACTOR"))
TIME_FORMAT = os.getenv("TIME_FORMAT")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
VEHICLE_CLASSES = [int(cls) for cls in os.getenv("VEHICLE_CLASSES").split(",")]
HTML_HEADERS = os.getenv("HTML_HEADERS").split(",")
WATCHDOG_INTERVAL = int(os.getenv("WATCHDOG_INTERVAL"))

# Initialize models
try:
    vehicle_model = YOLO(os.getenv('VEHICLE_MODEL_PATH'))
    plate_model = YOLO(os.getenv('PLATE_MODEL_PATH'))
    # Initialize PaddleOCR
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
except Exception as e:
    logger.error(f"Error loading models: {e}")
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
    """Recognize text from license plate image using PaddleOCR"""
    
    logger.info("Starting license plate recognition with PaddleOCR")

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
        # Ensure we have a 3-channel image. Some preprocessing paths return a
        # single-channel (grayscale) image which breaks downstream code that
        # expects img.shape[2]. Convert to BGR if needed.
        if len(plate_img.shape) == 2 or (len(plate_img.shape) == 3 and plate_img.shape[2] == 1):
            logger.debug('Converting single-channel plate image to BGR for OCR')
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

        # Prepare a cleaner image for OCR and call PaddleOCR
        ocr_input = prepare_plate_for_ocr(plate_img)
        logger.info("Calling PaddleOCR")
        result = ocr_model.ocr(ocr_input)

        # Log raw OCR output at INFO so it's visible in standard logs (before filtering)
        try:
            logger.info('Raw PaddleOCR output (primary): %s', result)
        except Exception:
            logger.info('Raw PaddleOCR output could not be stringified')

        # Also log each detected line (text + confidence + box) using the
        # robust parser so we handle both list and dict-shaped PaddleOCR
        # outputs uniformly and avoid index/type errors.
        try:
            parsed_preview = _parse_ocr_lines(result)
            if parsed_preview:
                for i, (text, conf, box, xc) in enumerate(parsed_preview):
                    logger.info("PaddleOCR line %d: text='%s', conf=%s, box=%s", i, text, conf, box)
        except Exception:
            logger.debug('No readable OCR lines to iterate')
        
        # Log processing time
        elapsed_time = time.time() - start_time
        logger.info(f"OCR completed in {elapsed_time:.2f} seconds")

        # Log memory usage after OCR
        mem_after = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory usage after OCR: {mem_after:.2f} MB (Change: {mem_after - mem_before:.2f} MB)")

        if result and result[0]:
            # Parse OCR lines robustly and assemble candidate text by X position
            parsed = _parse_ocr_lines(result)

            def assemble_from_parsed(parsed_lines):
                parts = []
                confs = []
                # sort by x_center (None -> end)
                parsed_sorted = sorted(parsed_lines, key=lambda x: (x[3] is None, x[3]))
                for text, conf, box, xc in parsed_sorted:
                    if not text:
                        continue
                    filtered = ''.join(ch for ch in text if ch.isalnum())
                    if filtered:
                        parts.append(filtered)
                        try:
                            confs.append(float(conf) if conf is not None else 0.0)
                        except Exception:
                            confs.append(0.0)
                if parts:
                    combined = ''.join(parts)
                    avg_conf = sum(confs) / len(confs) if confs else 0.0
                    return combined, avg_conf
                return None, None

            combined_text, avg_confidence = assemble_from_parsed(parsed)
            if combined_text:
                logger.info(f"PaddleOCR result: Text='{combined_text}', Confidence={avg_confidence:.3f}")
                return combined_text, avg_confidence

            # No alphanumeric from primary result. Try several fallback strategies:
            logger.warning("PaddleOCR found text but no valid alphanumeric characters")
            try:
                logger.info('Raw PaddleOCR output (primary) for failure case: %s', result)
            except Exception:
                logger.info('Raw PaddleOCR output could not be stringified for failure case')

            candidates = []

            # 1) Try OCR on upscaled original if small
            try:
                h, w = plate_img.shape[:2]
                scale = 1.0
                min_h = 128
                min_w = 320
                if h < min_h or w < min_w:
                    scale = max(min_h / max(1, h), min_w / max(1, w))
                    up = cv2.resize(plate_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                    logger.info('Attempting OCR on upscaled image (scale=%0.2f)', scale)
                    up_res = ocr_model.ocr(up)
                    parsed_up = _parse_ocr_lines(up_res)
                    assembled = assemble_from_parsed(parsed_up)
                    if assembled[0]:
                        candidates.append((assembled[0], assembled[1], 'upscaled'))
            except Exception as up_ex:
                logger.debug('Upscale OCR attempt failed: %s', up_ex)

            # 2) If PaddleOCR provided a rotated image (rot_img), try OCR on it
            try:
                rot_img = None
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    rot_img = result[0].get('rot_img')
                if rot_img is not None:
                    logger.info('Attempting OCR on rotated image provided by PaddleOCR')
                    rot_res = ocr_model.ocr(rot_img)
                    parsed_rot = _parse_ocr_lines(rot_res)
                    assembled = assemble_from_parsed(parsed_rot)
                    if assembled[0]:
                        candidates.append((assembled[0], assembled[1], 'rotated'))
            except Exception as rot_ex:
                logger.debug('Rotated OCR attempt failed: %s', rot_ex)

            # 3) Try OCR on original (less processed)
            try:
                logger.info('Attempting fallback OCR on original plate image (less processed)')
                orig_res = ocr_model.ocr(plate_img)
                parsed_orig = _parse_ocr_lines(orig_res)
                assembled = assemble_from_parsed(parsed_orig)
                if assembled[0]:
                    candidates.append((assembled[0], assembled[1], 'original'))
            except Exception as orig_ex:
                logger.debug('Original OCR attempt failed: %s', orig_ex)

            # Choose best candidate by highest confidence (and prefer non-empty)
            if candidates:
                # sort by confidence descending
                candidates.sort(key=lambda x: x[1] if x[1] is not None else 0.0, reverse=True)
                best = candidates[0]
                logger.info("PaddleOCR fallback chosen (%s): Text='%s', Confidence=%s", best[2], best[0], best[1])
                return best[0], best[1]

            # No candidates found - save artifacts for debugging
            logger.warning('Fallback OCR also returned no valid alphanumeric characters')
            try:
                fail_dir = os.path.join(LOG_DIR, 'ocr_failures')
                os.makedirs(fail_dir, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                proc_path = os.path.join(fail_dir, f'{ts}_processed.jpg')
                orig_path = os.path.join(fail_dir, f'{ts}_original.jpg')
                cv2.imwrite(proc_path, ocr_input)
                cv2.imwrite(orig_path, plate_img)
                try:
                    import json
                    raw_path = os.path.join(fail_dir, f'{ts}_raw.json')
                    with open(raw_path, 'w') as rf:
                        json.dump(result, rf, default=str)
                    logger.info('Saved OCR failure artifacts to %s', fail_dir)
                except Exception:
                    logger.warning('Failed to write raw OCR JSON for diagnostics')
            except Exception as save_ex:
                logger.warning('Failed to save OCR diagnostic artifacts: %s', save_ex)

            return None, None
        else:
            logger.warning("PaddleOCR returned no results")
            return None, None

    except Exception as ex:
        elapsed_time = time.time() - start_time
        logger.error(f"Error in plate recognition after {elapsed_time:.2f} seconds: {ex}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error in plate recognition: {ex}")
        return None, None


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


def send_email_with_attachment(filename):
    sender_email = os.getenv('SENDER_EMAIL')
    to_emails = os.getenv('TO_EMAILS').split(',')
    cc_emails = os.getenv('CC_EMAILS').split(',') if os.getenv('CC_EMAILS') else []
    password = os.getenv('PASSWORD')
    subject = os.getenv('SUBJECT')
    body1 = os.getenv('BODY1')
    body2 = os.getenv('BODY2')
    smtp_server = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT'))

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
            try:
                server.login(sender_email, password)
            except smtplib.SMTPAuthenticationError as auth_ex:
                # Log the authentication error and continue without crashing
                logger.error('SMTP authentication failed: %s', auth_ex)
                print('SMTP authentication failed; email will not be sent')
                return
            logger.info('Logged in successfully')
            server.sendmail(sender_email, all_recipients, text)
            logger.info('Email sent successfully!')
    except smtplib.SMTPException as ex:
        logger.error('An error occurred while sending the email: %s', ex)
        # Do not re-raise to avoid killing the worker process; caller will
        # log the failure and continue.
        return

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
    plates_new_dir = os.path.join(input_dir, "plates_new")
    output_file = os.path.join(input_dir, "index.html")

    # Validate original plates directory exists
    if not os.path.exists(plates_dir):
        logger.error(f"Plates directory not found: {plates_dir}")
        return

    # Prepare cropped images into `plates_new` (do a fresh crop)
    # If a previous `plates_new` exists, remove it to avoid stale files.
    if os.path.exists(plates_new_dir):
        try:
            shutil.rmtree(plates_new_dir)
        except Exception as e:
            logger.warning('Failed to remove existing plates_new directory %s: %s', plates_new_dir, e)

    crop_images_in_folder(plates_dir, plates_new_dir)

    data = []
    total_runs = 0
    total_not_read = 0
    total_detections = len(os.listdir(plates_new_dir))

    try:
        for obj_id in os.listdir(plates_new_dir):
            logger.info(f"Object processed by OCR: {total_runs}/{total_detections} out of which {total_not_read} are unable to read by OCR.")
            print(f"Object processed by OCR: {total_runs}/{total_detections} out of which {total_not_read} are unable to read by OCR.")
            total_runs += 1
            obj_dir = os.path.join(plates_new_dir, obj_id)
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

                        # Remove spaces from the plate number and change to all CAPS (only if text is not None)
                        if text:
                            text = text.replace(" ", "").upper()
                        else:
                            text = ""

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
        send_email_with_attachment(output_file)
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
    logger.info("scheduled_job Start")
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

    time_stamp = os.getenv('JOB_TIME')

    
    logger.info("Setting up scheduled job at %s", time_stamp)
    # Schedule the OCR job to run daily
    schedule.every().day.at(time_stamp).do(scheduled_job)
    #schedule.every(5).minutes.do(scheduled_job)

    # Main loop with watchdog
    while True:
        schedule.run_pending()

        if not vehicle_process.is_alive():
            logger.error("Vehicle detection process died. Restarting...")
            vehicle_process = start_vehicle_process(frame_queue, result_queue)

        if not plate_process.is_alive():
            logger.error("Plate detection process died. Restarting...")
            plate_process = start_plate_process(frame_queue, result_queue)

        time.sleep(1)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f'Multiprocessing start method already set: {e}')
    main()
