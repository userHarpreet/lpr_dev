import cv2
from ultralytics import YOLO
import easyocr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import threading
import queue

# Set up logging
logging.basicConfig(filename='anpr_log.txt', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize YOLOv10 model
model = YOLO('yolov10n.pt')  # Use the smallest model for speed

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Use GPU for OCR

# Initialize database
Base = declarative_base()
engine = create_engine('sqlite:///anpr_database.db')
Session = sessionmaker(bind=engine)


class PlateRecord(Base):
    __tablename__ = 'plate_records'
    id = Column(Integer, primary_key=True)
    plate_number = Column(String)
    timestamp = Column(DateTime)
    location = Column(String)
    confidence = Column(Float)


Base.metadata.create_all(engine)


def detect_and_recognize_plate(frame):
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):
            if confidence > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box[:4])
                plate_img = frame[y1:y2, x1:x2]

                # Perform OCR on the cropped image
                ocr_result = reader.readtext(plate_img)

                if ocr_result:
                    plate_text = ocr_result[0][1]
                    return plate_text, confidence, (x1, y1, x2, y2)

    return None, None, None


def store_plate(plate_number, confidence, location):
    session = Session()
    new_record = PlateRecord(plate_number=plate_number,
                             timestamp=datetime.now(),
                             location=location,
                             confidence=confidence)
    session.add(new_record)
    session.commit()
    session.close()


def process_frame(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        plate_text, confidence, bbox = detect_and_recognize_plate(frame)
        result_queue.put((plate_text, confidence, bbox))


def main():
    cap = cv2.VideoCapture("input.mkv")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    location = "Main Street Camera"

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    # Start processing thread
    processing_thread = threading.Thread(target=process_frame, args=(frame_queue, result_queue))
    processing_thread.start()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        if frame_queue.empty():
            frame_queue.put(frame)

        if not result_queue.empty():
            plate_text, confidence, bbox = result_queue.get()
            if plate_text:
                store_plate(plate_text, confidence, location)
                logging.info(f"Detected plate: {plate_text}, Confidence: {confidence}")

                # Draw bounding box and text on frame
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('ANPR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_queue.put(None)  # Signal the processing thread to stop
    processing_thread.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()