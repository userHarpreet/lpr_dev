import os
import cv2
import easyocr
from datetime import datetime, timedelta
import multiprocessing as mp
import logging
import schedule
import configparser
import time
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from ultralytics import YOLO
from validate_number import validate_hsrp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='vehicle_detection.log', filemode='w')
logger = logging.getLogger(__name__)


def read_config(config_path='requirements/config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found!")
        raise FileNotFoundError(f"Configuration file {config_path} not found!")
    config.read(config_path)
    return config


config = read_config()
SHOW_LIVE = config.getboolean('General', 'SHOW_LIVE')
PLATE_CONF_MIN = config.getfloat('General', 'PLATE_CONF_MIN')
VEHICLE_CONF_MIN = config.getfloat('General', 'VEHICLE_CONF_MIN')
VIDEO_SOURCE = config.get('General', 'VIDEO_SOURCE')
RESIZE_FACTOR = config.getint('General', 'RESIZE_FACTOR')
TIME_FORMAT = config.get('General', 'TIME_FORMAT')
OUTPUT_DIR = config.get('General', 'OUTPUT_DIR')
VEHICLE_CLASSES = [int(cls) for cls in config.get('General', 'VEHICLE_CLASSES').split(',')]
HTML_HEADERS = config.get('HTML', 'HEADERS').split(',')

# Initialize models
try:
    vehicle_model = YOLO(config.get('Models', 'VEHICLE_MODEL_PATH'))
    plate_model = YOLO(config.get('Models', 'PLATE_MODEL_PATH'))
except Exception as e:
    logger.error(f"Error loading YOLO models: {e}")
    raise

logger.info("Models loaded successfully")

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    logger.error(f"Error initializing EasyOCR: {e}")
    raise


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
    try:
        ocr_result = reader.readtext(plate_img)
        if ocr_result:
            return ocr_result[0][1], ocr_result[0][2]  # text and confidence
        return None, None
    except Exception as e:
        logger.error(f"Error in plate recognition: {e}")
        return None, None


def save_image(directory, filename, image):
    ensure_dir(directory)
    cv2.imwrite(os.path.join(directory, filename), image)
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
    timestamp = datetime.now().strftime(TIME_FORMAT)
    _, plates_dir, frames_dir = get_output_dirs()
    for obj in result.boxes.data.tolist():
        try:
            x1, y1, x2, y2, obj_id, conf, obj_class = map(float, obj)
            if int(obj_class) in VEHICLE_CLASSES:
                vehicle = frame[int(y1):int(y2), int(x1):int(x2)]
                timestamp = get_plate(vehicle, int(obj_id))
                save_image(os.path.join(frames_dir, str(int(obj_id))), f"{timestamp}.jpg", vehicle)
                logger.info(f"Processed frame for object {int(obj_id)} at {timestamp}")
        except ValueError as e:
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


def create_html_table(data, output_file):
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
        <h1>Vehicle Detection Results</h1>
        <table>
            <tr>
    """
    for header in HTML_HEADERS:
        html_content += f"<th>{header}</th>"
    html_content += "</tr>"
    for row in data:
        html_content += "<tr>"
        for cell in row:
            html_content += f"<td>{cell}</td>"
        html_content += "</tr>"
    html_content += """
        </table>
    </body>
    </html>
    """
    with open(output_file, 'w') as f:
        f.write(html_content)
    logger.info(f"HTML file created: {output_file}")


def send_email_with_attachment(config, filename):
    sender_email = config.get('Email', 'SENDER_EMAIL')
    to_emails = config.get('Email', 'TO_EMAILS').split(',')
    cc_emails = config.get('Email', 'CC_EMAILS').split(',') if config.get('Email', 'CC_EMAILS') else []
    password = config.get('Email', 'PASSWORD')
    subject = config.get('Email', 'SUBJECT')
    body = config.get('Email', 'BODY')
    smtp_server = config.get('SMTP', 'HOST')
    smtp_port = config.getint('SMTP', 'PORT')

    logger.info('Preparing to send email...')
    logger.info('Sender: %s', sender_email)
    logger.info('To: %s', ', '.join(to_emails))
    logger.info('Cc: %s', ', '.join(cc_emails))
    logger.info('Subject: %s', subject)
    logger.info('Attachment: %s', filename)

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(to_emails)
    message["Cc"] = ", ".join(cc_emails)
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    logger.debug('Email body attached')

    try:
        with open(filename, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            logger.debug('File %s read successfully', filename)
    except IOError as e:
        logger.error('Failed to read attachment file: %s', e)
        raise

    encoders.encode_base64(part)
    logger.debug('File encoded successfully')
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )
    message.attach(part)
    logger.debug('Attachment added to message')

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
    except smtplib.SMTPException as e:
        logger.error('An error occurred while sending the email: %s', e)
        raise

    logger.info('Email sending process completed')


def run_ocr_and_save_to_html(date):
    logger.info(f"Starting OCR process for {date}")
    input_dir = os.path.join(OUTPUT_DIR, date)
    plates_dir = os.path.join(input_dir, "plates")
    output_file = os.path.join(input_dir, f"index.html")
    data = []
    serial = 1

    for obj_id in os.listdir(plates_dir):
        obj_dir = os.path.join(plates_dir, obj_id)
        image_files = sorted(os.listdir(obj_dir))
        indices_to_process = []
        if len(image_files) > 1:
            middle_index = len(image_files) // 2
            second_last_index = len(image_files) - 2
            indices_to_process = [middle_index, second_last_index]

        results = []
        for index in indices_to_process:
            if index < len(image_files):
                image_file = image_files[index]
                image_path = os.path.join(obj_dir, image_file)
                plate_img = cv2.imread(image_path)
                text, confidence = recognize_plate(plate_img)
                if validate_hsrp(text) is True:
                    results.append((text, confidence, image_file))
                else:
                    logger.error('Not a valid HSRP Number.')

        if results:
            text1, confidence1, image_file1 = results[0] if len(results) > 0 else (None, None, None)
            text2, confidence2, image_file2 = results[1] if len(results) > 1 else (None, None, None)
            tStamp = get_timestamp_from_filename(image_file1)
            row = [
                serial,
                obj_id,
                tStamp,
                "Yes" if text1 or text2 else "No",
                confidence1 if confidence1 is not None else "",
                confidence2 if confidence2 is not None else "",
                text1 if text1 is not None else "",
                text2 if text2 is not None else ""
            ]
            data.append(row)
            serial += 1

    create_html_table(data, output_file)
    send_email_with_attachment(config, output_file)
    logger.info(f"OCR process completed and results saved to HTML for {date}")


def get_timestamp_from_filename(filename):
    tStamp = filename[0:13]
    pos_date = [4, 6]
    pos_date.sort()
    for i, pos in enumerate(pos_date):
        tStamp = tStamp[:pos + i] + '/' + tStamp[pos + i:]
    tStamp = tStamp.replace("_", " ")
    tStamp = tStamp[:13] + ':' + tStamp[13:]
    return tStamp


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
    vehicle_process = mp.Process(target=vehicle_detection, args=(frame_queue, result_queue))
    plate_process = mp.Process(target=plate_detection, args=(frame_queue, result_queue))
    vehicle_process.start()
    plate_process.start()

    TIME_STAMP = config.get('Schedule', 'JOB_TIME')

    # Schedule the OCR job to run daily at 19:08 PM
    schedule.every().day.at(TIME_STAMP).do(scheduled_job)

    # Run the scheduled jobs
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
