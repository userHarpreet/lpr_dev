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
        results = vehicle_model.track(frame, persist=True, tracker="botsort.yaml")
        vehicles = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                id = int(box.id[0].item())
                if conf > 0.5:  # Confidence threshold
                    vehicles.append((id, (int(x1), int(y1), int(x2), int(y2))))
        return vehicles
    except Exception as e:
        logging.error(f"Error in vehicle detection: {e}")
        return []


def detect_license_plates(frame, vehicle_boxes):
    try:
        license_plates = []
        for vehicle_id, box in vehicle_boxes:
            x1, y1, x2, y2 = box
            vehicle_crop = frame[y1:y2, x1:x2]
            results = license_plate_model.track(frame, persist=True, tracker="botsort.yaml")
            for r in results:
                plate_boxes = r.boxes
                for plate_box in plate_boxes:
                    if plate_box.id is None:
                        continue
                    px1, py1, px2, py2 = plate_box.xyxy[0].tolist()
                    plate_conf = plate_box.conf[0].item()
                    plate_id = int(plate_box.id[0].item())
                    if plate_conf > 0.5:  # Confidence threshold
                        license_plates.append(
                            (vehicle_id, plate_id, (int(x1 + px1), int(y1 + py1), int(x1 + px2), int(y1 + py2))))
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


def store_plate(vehicle_id, plate_id, plate_number, confidence, location, frame_path, plate_path):
    try:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{vehicle_id}_{plate_id}", plate_number, datetime.now(), location, confidence, frame_path,
                             plate_path])
    except Exception as e:
        logging.error(f"Error storing plate: {e}")


def main():
    cap = cv2.VideoCapture("input_5.mkv")  # Use 0 for webcam or provide video file path
    location = "Gate 6 In Camera"

    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            break

        frame_count += 1

        vehicle_boxes = detect_vehicles(frame)
        license_plate_boxes = detect_license_plates(frame, vehicle_boxes)

        for _, box in vehicle_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for vehicle_id, plate_id, box in license_plate_boxes:
            x1, y1, x2, y2 = box
            plate_img = frame[y1:y2, x1:x2]
            plate_text, confidence = recognize_plate(plate_img)
            if plate_text:
                frame_path = os.path.join(frames_dir, f'frame_{vehicle_id}_{plate_id}_{frame_count}.jpg')
                plate_path = os.path.join(plates_dir, f'plate_{vehicle_id}_{plate_id}_{frame_count}.jpg')

                cv2.imwrite(frame_path, frame)
                cv2.imwrite(plate_path, plate_img)

                store_plate(vehicle_id, plate_id, plate_text, confidence, location, frame_path, plate_path)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{plate_text} (V:{vehicle_id} P:{plate_id})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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