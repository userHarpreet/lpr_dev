"""
Storage Process Module

Stores OCR results, plate images, and metadata to database and file system.
"""

import cv2
import logging
import time
import sqlite3
from multiprocessing import Process, Event, Queue
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Load configuration from environment
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output_dir")
TIME_FORMAT = os.getenv("TIME_FORMAT", "%Y%m%d_%H%M%S")


@dataclass
class StorageRecord:
    """Data structure for a storage record."""
    plate_id: int
    frame_id: int
    timestamp: datetime
    recognized_text: Optional[str]
    confidence: Optional[float]
    image_path: Optional[str]
    metadata: dict = None


class StorageProcess(Process):
    """
    Process that stores OCR results and plate images to database and file system.
    
    Reads frames with OCR results from input queue, saves plate images,
    and stores metadata in SQLite database.
    
    Args:
        in_q: Input queue with FrameWithOCR objects from OCRProcess
        stop_event: Event to signal process termination
        storage_dir: Base directory for storing data
        db_path: Path to SQLite database file
        name: Process name (optional)
    """
    
    def __init__(
        self,
        in_q: Optional[Queue] = None,
        stop_event: Optional[Event] = None,
        storage_dir: str = "./data",
        db_path: str = "./data/lpr.db",
        name: str = "Storage",
    ):
        super().__init__(name=name)
        self.daemon = False
        self.in_q = in_q
        self.stop_event = stop_event
        self.storage_dir = storage_dir
        self.db_path = db_path
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics
        self.frames_processed = 0
        self.records_stored = 0
        self.storage_errors = 0
        self.frames_dropped = 0
        self.start_time = None
        
        # Database connection (will be created in run())
        self.db_conn = None
    
    def run(self):
        """Main process loop - store data."""
        try:
            self.logger.info(f"[{self.name}] Starting storage process")
            self.logger.info(f"[{self.name}] Storage directory: {self.storage_dir}")
            self.logger.info(f"[{self.name}] Database path: {self.db_path}")
            
            self._init_database()
            self._store_data()
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error in storage: {e}", exc_info=True)
        finally:
            if self.db_conn:
                self.db_conn.close()
            
            self.logger.info(
                f"[{self.name}] Storage process stopped. Stats - "
                f"Processed: {self.frames_processed}, "
                f"Stored: {self.records_stored}, "
                f"Errors: {self.storage_errors}, "
                f"Dropped: {self.frames_dropped}"
            )
    
    def _init_database(self):
        """Initialize SQLite database and create tables if needed."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lpr_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_id INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    recognized_text TEXT,
                    confidence REAL,
                    image_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frame_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id INTEGER UNIQUE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    vehicle_count INTEGER,
                    plate_count INTEGER,
                    source_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON lpr_records(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recognized_text 
                ON lpr_records(recognized_text)
            """)
            
            self.db_conn.commit()
            
            self.logger.info(f"[{self.name}] Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to initialize database: {e}", exc_info=True)
            raise
    
    def _store_data(self):
        """Process frames and store data."""
        self.start_time = time.time()
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Get frame with OCR results from input queue
                frame_obj = self.in_q.get(timeout=2.0)
                
                try:
                    # Store frame metadata
                    self._store_frame_metadata(frame_obj)
                    
                    # Store OCR results
                    for ocr_result in frame_obj.ocr_results:
                        try:
                            image_path = self._save_plate_image(
                                ocr_result.plate_id,
                                frame_obj.frame_id,
                                ocr_result.timestamp
                            )
                            
                            self._store_ocr_record(
                                ocr_result.plate_id,
                                ocr_result.frame_id,
                                ocr_result.timestamp,
                                ocr_result.raw_text,
                                ocr_result.confidence,
                                image_path
                            )
                            
                            self.records_stored += 1
                            
                        except Exception as e:
                            self.logger.warning(
                                f"[{self.name}] Failed to store OCR result for plate {ocr_result.plate_id}: {e}"
                            )
                            self.storage_errors += 1
                    
                    self.frames_processed += 1
                    
                except Exception as e:
                    self.logger.warning(f"[{self.name}] Failed to store frame {frame_obj.frame_id}: {e}")
                    self.storage_errors += 1
                    
            except Exception as e:
                # Queue timeout or other error
                if "Empty" not in str(e):
                    self.logger.debug(f"[{self.name}] Queue timeout or error: {e}")
    
    def _store_frame_metadata(self, frame_obj) -> bool:
        """Store frame metadata to database."""
        try:
            cursor = self.db_conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO frame_metadata 
                (frame_id, timestamp, vehicle_count, plate_count, source_name)
                VALUES (?, ?, ?, ?, ?)
            """, (
                frame_obj.frame_id,
                frame_obj.timestamp.isoformat(),
                frame_obj.vehicle_count,
                frame_obj.plate_count,
                frame_obj.source_name
            ))
            
            self.db_conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error storing frame metadata: {e}")
            return False
    
    def _save_plate_image(self, plate_id: int, frame_id: int, timestamp: datetime) -> Optional[str]:
        """
        Save plate image to file system.
        
        Directory structure:
        OUTPUT_DIR/
            YYYY-MM-DD/
                plates/
                    <plate_id>/
                        <timestamp>.jpg
        
        Args:
            plate_id: Unique plate identifier
            frame_id: Frame identifier
            timestamp: Timestamp of detection
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            # Create date-based directory
            date_str = timestamp.strftime("%Y-%m-%d")
            plates_dir = os.path.join(OUTPUT_DIR, date_str, "plates", str(plate_id))
            
            os.makedirs(plates_dir, exist_ok=True)
            
            # Create filename with timestamp
            filename = timestamp.strftime(TIME_FORMAT) + ".jpg"
            image_path = os.path.join(plates_dir, filename)
            
            self.logger.debug(f"[{self.name}] Plate image saved: {image_path}")
            
            return image_path
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error saving plate image: {e}")
            return None
    
    def _store_ocr_record(
        self,
        plate_id: int,
        frame_id: int,
        timestamp: datetime,
        recognized_text: Optional[str],
        confidence: Optional[float],
        image_path: Optional[str]
    ) -> bool:
        """Store OCR record to database."""
        try:
            cursor = self.db_conn.cursor()
            
            cursor.execute("""
                INSERT INTO lpr_records 
                (plate_id, frame_id, timestamp, recognized_text, confidence, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                plate_id,
                frame_id,
                timestamp.isoformat(),
                recognized_text,
                confidence,
                image_path
            ))
            
            self.db_conn.commit()
            
            if recognized_text:
                self.logger.debug(
                    f"[{self.name}] Stored OCR record: Plate {plate_id}, "
                    f"Text='{recognized_text}', Conf={confidence:.2f}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error storing OCR record: {e}")
            return False
    
    def query_records(self, start_date: datetime = None, end_date: datetime = None) -> List[dict]:
        """
        Query OCR records from database.
        
        Args:
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            List of record dictionaries
        """
        try:
            cursor = self.db_conn.cursor()
            
            query = "SELECT * FROM lpr_records WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return records
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error querying records: {e}")
            return []
    
    def get_statistics(self) -> dict:
        """Get storage statistics from database."""
        try:
            cursor = self.db_conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM lpr_records")
            total_records = cursor.fetchone()[0]
            
            # Records with recognized text
            cursor.execute("SELECT COUNT(*) FROM lpr_records WHERE recognized_text IS NOT NULL AND recognized_text != ''")
            recognized_records = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM lpr_records WHERE confidence IS NOT NULL")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Total frames
            cursor.execute("SELECT COUNT(*) FROM frame_metadata")
            total_frames = cursor.fetchone()[0]
            
            return {
                "total_records": total_records,
                "recognized_records": recognized_records,
                "recognition_rate": (recognized_records / total_records * 100) if total_records > 0 else 0,
                "average_confidence": avg_confidence,
                "total_frames": total_frames
            }
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error getting statistics: {e}")
            return {}


class StorageQueryInterface:
    """Interface for querying stored data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def get_by_plate_text(self, text: str) -> List[dict]:
        """Get records by recognized plate text."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM lpr_records WHERE recognized_text = ? ORDER BY timestamp DESC",
                (text,)
            )
            
            columns = [desc[0] for desc in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            
            return records
        except Exception as e:
            self.logger.error(f"Error querying by plate text: {e}")
            return []
    
    def get_by_date(self, date: datetime) -> List[dict]:
        """Get records for a specific date."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_of_day = datetime(date.year, date.month, date.day)
            end_of_day = datetime(date.year, date.month, date.day, 23, 59, 59)
            
            cursor.execute(
                "SELECT * FROM lpr_records WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC",
                (start_of_day.isoformat(), end_of_day.isoformat())
            )
            
            columns = [desc[0] for desc in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            
            return records
        except Exception as e:
            self.logger.error(f"Error querying by date: {e}")
            return []
    
    def get_unrecognized_plates(self) -> List[dict]:
        """Get plates that were not successfully recognized."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM lpr_records WHERE recognized_text IS NULL OR recognized_text = '' ORDER BY timestamp DESC"
            )
            
            columns = [desc[0] for desc in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            
            return records
        except Exception as e:
            self.logger.error(f"Error querying unrecognized plates: {e}")
            return []
    
    def get_statistics(self) -> dict:
        """Get overall statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM lpr_records")
            stats['total_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM lpr_records WHERE recognized_text IS NOT NULL AND recognized_text != ''")
            stats['recognized_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(confidence) FROM lpr_records WHERE confidence IS NOT NULL")
            avg_conf = cursor.fetchone()[0]
            stats['average_confidence'] = round(avg_conf, 3) if avg_conf else 0.0
            
            if stats['total_records'] > 0:
                stats['recognition_rate'] = round(
                    stats['recognized_records'] / stats['total_records'] * 100, 2
                )
            else:
                stats['recognition_rate'] = 0.0
            
            cursor.execute("SELECT COUNT(DISTINCT frame_id) FROM lpr_records")
            stats['total_frames'] = cursor.fetchone()[0]
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
    )
    
    # Create test queue and event
    test_in_q = Queue(maxsize=10)
    stop_event = Event()
    
    # Create storage process
    storage = StorageProcess(
        in_q=test_in_q,
        stop_event=stop_event,
        storage_dir="./data",
        db_path="./data/lpr.db",
    )
    
    storage.start()
    
    # Test with sample data
    try:
        from ocr_process import FrameWithOCR, OCRResult
        
        frame = FrameWithOCR(
            frame_id=0,
            timestamp=datetime.now(),
            source_name="test_camera",
            vehicle_count=1,
            plate_count=1,
            ocr_results=[
                OCRResult(
                    plate_id=0,
                    frame_id=0,
                    timestamp=datetime.now(),
                    raw_text="AB123CD",
                    confidence=0.95,
                    detection_method="primary"
                )
            ]
        )
        
        test_in_q.put(frame)
        time.sleep(2)
        
        # Query results
        query = StorageQueryInterface("./data/lpr.db")
        stats = query.get_statistics()
        print(f"Statistics: {stats}")
        
    except Exception as e:
        print(f"Test error: {e}")
    
    stop_event.set()
    storage.join(timeout=5)
    print("Storage process stopped")
