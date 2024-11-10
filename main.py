import cv2
from ultralytics import YOLO
import easyocr
import csv
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(filename='anpr_log.txt', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize models
try:
    vehicle_model = YOLO('yolo11n.pt')  # Pre-trained YOLOv11n
    license_plate_model = YOLO('best28.pt')  # Your custom-trained model
except Exception as e:
    logging.error(f"Error loading YOLO models: {e}")
    raise

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)  # Disable GPU for OCR
except Exception as e:
    logging.error(f"Error initializing EasyOCR: {e}")
    raise

# Create output directories
output_dir = 'anpr_output'
frames_dir = os.path.join(output_dir, 'frames')
plates_dir = os.path.join(output_dir, 'plates')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(plates_dir, exist_ok=True)

# CSV file setup
csv_file = os.path.join(output_dir, 'anpr_data.csv')
csv_headers = ['object_id', 'plate_number', 'timestamp', 'location', 'confidence', 'frame_path', 'plate_path']

# Create CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)


def detect_vehicles(frame):
    try:
        results = vehicle_model(frame)
        vehicles = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, confidence in zip(boxes, confidences):
                if confidence > 0.5:  # Confidence threshold
                    vehicles.append(box)
        return vehicles
    except Exception as e:
        logging.error(f"Error in vehicle detection: {e}")
        return []


def detect_license_plates(frame, vehicle_boxes):
    try:
        license_plates = []
        for box in vehicle_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            vehicle_crop = frame[y1:y2, x1:x2]
            results = license_plate_model(vehicle_crop)
            for result in results:
                plate_boxes = result.boxes.xyxy.cpu().numpy()
                plate_confidences = result.boxes.conf.cpu().numpy()
                for plate_box, plate_confidence in zip(plate_boxes, plate_confidences):
                    if plate_confidence > 0.5:  # Confidence threshold
                        px1, py1, px2, py2 = map(int, plate_box[:4])
                        license_plates.append((x1 + px1, y1 + py1, x1 + px2, y1 + py2))
        return license_plates
    except Exception as e:
        logging.error(f"Error in license plate detection: {e}")
        return []


def recognize_plate(plate_img):
    try:
        ocr_result = reader.readtext(plate_img)
        if ocr_result:
            return ocr_result[0][1], ocr_result[0][2]  # text and confidence
        return None, None
    except Exception as e:
        logging.error(f"Error in plate recognition: {e}")
        return None, None


def store_plate(object_id, plate_number, confidence, location, frame_path, plate_path):
    try:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([object_id, plate_number, datetime.now(), location, confidence, frame_path, plate_path])
    except Exception as e:
        logging.error(f"Error storing plate: {e}")


def main():
    cap = cv2.VideoCapture("input_5.mkv")  # Use 0 for webcam or provide video file path
    location = "Gate 6 In Camera"

    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return

    object_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            break

        vehicle_boxes = detect_vehicles(frame)
        license_plate_boxes = detect_license_plates(frame, vehicle_boxes)

        for box in vehicle_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for box in license_plate_boxes:
            x1, y1, x2, y2 = box
            plate_img = frame[y1:y2, x1:x2]
            plate_text, confidence = recognize_plate(plate_img)
            if plate_text:
                object_id += 1
                frame_path = os.path.join(frames_dir, f'frame_{object_id}.jpg')
                plate_path = os.path.join(plates_dir, f'plate_{object_id}.jpg')

                cv2.imwrite(frame_path, frame)
                cv2.imwrite(plate_path, plate_img)

                store_plate(object_id, plate_text, confidence, location, frame_path, plate_path)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('ANPR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main function error: {e}")