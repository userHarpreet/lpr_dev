import os
import sys
import time  
import logging
from dotenv import load_dotenv
from threading import Thread, Lock
from collections import defaultdict
from multiprocessing import Event, Queue, set_start_method, Process

from ocr_process import OCRProcess
from storage_process import StorageProcess
from plate_editor import PlateEditorProcess
from video_reader import VideoReaderProcess
from plate_detector import PlateDetectorProcess
from vehicle_detector import VehicleDetectorProcess
# -------------------------
# Load configuration
# -------------------------

load_dotenv()
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

# Pipeline configuration
MAX_VEHICLE_WORKERS = int(os.getenv("MAX_VEHICLE_WORKERS", "2"))
MAX_PLATE_WORKERS = int(os.getenv("MAX_PLATE_WORKERS", "3"))
MAX_EDIT_WORKERS = int(os.getenv("MAX_EDIT_WORKERS", "2"))
MAX_OCR_WORKERS = int(os.getenv("MAX_OCR_WORKERS", "2"))
WORKER_IDLE_TIMEOUT = int(os.getenv("WORKER_IDLE_TIMEOUT", "10"))  # seconds


# -------------------------
# Utility functions
# -------------------------

def ensure_dirs(directory):
    os.makedirs(directory, exist_ok=True)
    logging.debug(f"Ensured directory exists: {directory}")

def init_db(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS lpr_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    logging.debug(f"Initialized database at: {db_path}")


class ProcessPool:
    """Manages a pool of worker processes for a stage."""
    
    def __init__(self, process_class, max_workers, name, input_queue, output_queue, stop_event, **kwargs):
        self.process_class = process_class
        self.max_workers = max_workers
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
        self.workers = []
        self.lock = Lock()
    
    def start_worker(self):
        """Start a new worker process."""
        with self.lock:
            if len(self.workers) >= self.max_workers:
                return False
            
            worker = self.process_class(
                in_q=self.input_queue,
                out_q=self.output_queue,
                stop_event=self.stop_event,
                name=f"{self.name}-Worker-{len(self.workers) + 1}",
                **self.kwargs
            )
            worker.start()
            self.workers.append(worker)
            self.logger.info(f"[{self.name}] Started worker: {worker.name} (Total: {len(self.workers)}/{self.max_workers})")
            return True
    
    def stop_idle_workers(self):
        """Stop workers that have been idle."""
        with self.lock:
            still_alive = [w for w in self.workers if w.is_alive()]
            
            if len(still_alive) > 1:  # Keep at least one worker
                worker = still_alive[0]
                worker.terminate()
                worker.join(timeout=5)
                self.workers.remove(worker)
                self.logger.info(f"[{self.name}] Stopped idle worker: {worker.name} (Total: {len(self.workers)}/{self.max_workers})")
    
    def stop_all_workers(self):
        """Stop all worker processes."""
        with self.lock:
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            
            self.logger.info(f"[{self.name}] All workers stopped")
            self.workers.clear()
    
    def get_active_workers(self):
        """Get number of active workers."""
        with self.lock:
            return len([w for w in self.workers if w.is_alive()])


# -------------------------
# Main orchestration
# -------------------------
def main():
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "[%(asctime)s] %(processName)s/%(levelname)s: %(message)s"

    # Configure Root Logger to ensure all modules (video_reader, etc.) are captured
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Clear existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add stream handler properly
    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(LOG_LEVEL)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Use root logger for this module too
    logger = logging.getLogger(__name__)

    # Use 'spawn' on platforms like Windows to be safe
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    db_path = os.path.join(storage_dir, "lpr_v2.db")
    LOG_DIR = os.path.join(storage_dir, "logs")

    ensure_dirs(storage_dir)
    init_db(db_path)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        try:
            file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'lpr_dev.log'), mode='a')
            file_handler.setLevel(LOG_LEVEL)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")
    except Exception as e:
        logger.error(f"Failed to create log directory {LOG_DIR}: {e}")


    logger.info("=" * 60)
    logger.info("Starting LPR Pipeline")
    logger.info("=" * 60)

    # Global stop event
    stop_event = Event()

    # Queues between stages
    q_frames = Queue(maxsize=20)
    q_vehicles = Queue(maxsize=30)
    q_plates = Queue(maxsize=30)
    q_edited = Queue(maxsize=30)
    q_ocr = Queue(maxsize=30)

    try:
        # Start Video Reader Process (reads frames from source)
        logger.info("Starting Video Reader...")
        video_reader = VideoReaderProcess(
            source=VIDEO_SOURCE,
            out_q=q_frames,
            stop_event=stop_event
        )
        video_reader.start()

        # Create worker pools for each stage
        logger.info(f"Creating worker pools (Vehicle: {MAX_VEHICLE_WORKERS}, Plate: {MAX_PLATE_WORKERS}, "
                    f"Edit: {MAX_EDIT_WORKERS}, OCR: {MAX_OCR_WORKERS})")
        
        vehicle_pool = ProcessPool(
            VehicleDetectorProcess,
            MAX_VEHICLE_WORKERS,
            "VehicleDetector",
            q_frames,
            q_vehicles,
            stop_event
        )

        plate_pool = ProcessPool(
            PlateDetectorProcess,
            MAX_PLATE_WORKERS,
            "PlateDetector",
            q_vehicles,
            q_plates,
            stop_event
        )

        edit_pool = ProcessPool(
            PlateEditorProcess,
            MAX_EDIT_WORKERS,
            "PlateEditor",
            q_plates,
            q_edited,
            stop_event
        )

        ocr_pool = ProcessPool(
            OCRProcess,
            MAX_OCR_WORKERS,
            "OCR",
            q_edited,
            q_ocr,
            stop_event
        )

        # Start Storage Process (runs continuously)
        logger.info("Starting Storage Process...")
        storage_proc = StorageProcess(
            in_q=q_ocr,
            stop_event=stop_event,
            storage_dir=storage_dir,
            db_path=db_path
        )
        storage_proc.start()

        # Start monitor thread to manage worker pools
        logger.info("Starting pipeline monitor thread...")
        monitor_thread = Thread(
            target=_monitor_pipeline,
            args=(
                vehicle_pool, plate_pool, edit_pool, ocr_pool,
                q_frames, q_vehicles, q_plates, q_edited, q_ocr,
                video_reader, stop_event, logger
            ),
            daemon=False
        )
        monitor_thread.start()

        logger.info("Pipeline started. Monitoring processes...")

        # Wait for video reader to finish
        video_reader.join()
        logger.info("Video reader finished")

        # Wait for all queues to be processed
        logger.info("Waiting for pipeline to complete...")
        _wait_for_pipeline_completion(
            q_frames, q_vehicles, q_plates, q_edited, q_ocr,
            vehicle_pool, plate_pool, edit_pool, ocr_pool,
            logger
        )

        # Signal all processes to stop
        logger.info("Signaling processes to stop...")
        stop_event.set()

        # Stop all worker pools
        vehicle_pool.stop_all_workers()
        plate_pool.stop_all_workers()
        edit_pool.stop_all_workers()
        ocr_pool.stop_all_workers()

        # Wait for storage process
        storage_proc.join(timeout=30)
        if storage_proc.is_alive():
            storage_proc.terminate()
            logger.warning("Storage process did not terminate gracefully, forced termination")

        monitor_thread.join(timeout=10)

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
        stop_event.set()
        video_reader.terminate()
        vehicle_pool.stop_all_workers()
        plate_pool.stop_all_workers()
        edit_pool.stop_all_workers()
        ocr_pool.stop_all_workers()
        storage_proc.terminate()
        logger.info("Pipeline shutdown complete")

    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        stop_event.set()
        raise


def _monitor_pipeline(vehicle_pool, plate_pool, edit_pool, ocr_pool,
                     q_frames, q_vehicles, q_plates, q_edited, q_ocr,
                     video_reader, stop_event, logger):
    """
    Monitor pipeline and dynamically start/stop workers based on queue sizes.
    """
    idle_counters = defaultdict(int)
    worker_start_times = {
        'vehicle': time.time(),
        'plate': time.time(),
        'edit': time.time(),
        'ocr': time.time()
    }

    while not stop_event.is_set() and video_reader.is_alive():
        time.sleep(2)  # Check every 2 seconds

        try:
            # Get queue sizes
            frames_size = q_frames.qsize()
            vehicles_size = q_vehicles.qsize()
            plates_size = q_plates.qsize()
            edited_size = q_edited.qsize()
            ocr_size = q_ocr.qsize()

            # Log queue status periodically
            logger.debug(
                f"Queue sizes - Frames: {frames_size}, Vehicles: {vehicles_size}, "
                f"Plates: {plates_size}, Edited: {edited_size}, OCR: {ocr_size}"
            )

            # Vehicle Detection scaling
            if frames_size > vehicle_pool.get_active_workers() * 5 and vehicle_pool.get_active_workers() < MAX_VEHICLE_WORKERS:
                vehicle_pool.start_worker()
                idle_counters['vehicle'] = 0
                worker_start_times['vehicle'] = time.time()
            elif frames_size == 0 and vehicles_size < 5:
                idle_counters['vehicle'] += 1
                if idle_counters['vehicle'] > WORKER_IDLE_TIMEOUT:
                    vehicle_pool.stop_idle_workers()
                    idle_counters['vehicle'] = 0
            else:
                idle_counters['vehicle'] = 0

            # Plate Detection scaling
            if vehicles_size > plate_pool.get_active_workers() * 3 and plate_pool.get_active_workers() < MAX_PLATE_WORKERS:
                plate_pool.start_worker()
                idle_counters['plate'] = 0
                worker_start_times['plate'] = time.time()
            elif vehicles_size == 0 and plates_size < 5:
                idle_counters['plate'] += 1
                if idle_counters['plate'] > WORKER_IDLE_TIMEOUT:
                    plate_pool.stop_idle_workers()
                    idle_counters['plate'] = 0
            else:
                idle_counters['plate'] = 0

            # Plate Editing scaling
            if plates_size > edit_pool.get_active_workers() * 3 and edit_pool.get_active_workers() < MAX_EDIT_WORKERS:
                edit_pool.start_worker()
                idle_counters['edit'] = 0
                worker_start_times['edit'] = time.time()
            elif plates_size == 0 and edited_size < 5:
                idle_counters['edit'] += 1
                if idle_counters['edit'] > WORKER_IDLE_TIMEOUT:
                    edit_pool.stop_idle_workers()
                    idle_counters['edit'] = 0
            else:
                idle_counters['edit'] = 0

            # OCR scaling
            if edited_size > ocr_pool.get_active_workers() * 2 and ocr_pool.get_active_workers() < MAX_OCR_WORKERS:
                ocr_pool.start_worker()
                idle_counters['ocr'] = 0
                worker_start_times['ocr'] = time.time()
            elif edited_size == 0 and ocr_size < 5:
                idle_counters['ocr'] += 1
                if idle_counters['ocr'] > WORKER_IDLE_TIMEOUT:
                    ocr_pool.stop_idle_workers()
                    idle_counters['ocr'] = 0
            else:
                idle_counters['ocr'] = 0

            logger.debug(
                f"Active workers - Vehicle: {vehicle_pool.get_active_workers()}/{MAX_VEHICLE_WORKERS}, "
                f"Plate: {plate_pool.get_active_workers()}/{MAX_PLATE_WORKERS}, "
                f"Edit: {edit_pool.get_active_workers()}/{MAX_EDIT_WORKERS}, "
                f"OCR: {ocr_pool.get_active_workers()}/{MAX_OCR_WORKERS}"
            )

        except Exception as e:
            logger.error(f"Error in monitor thread: {e}", exc_info=True)


def _wait_for_pipeline_completion(q_frames, q_vehicles, q_plates, q_edited, q_ocr,
                                  vehicle_pool, plate_pool, edit_pool, ocr_pool, logger):
    """
    Wait for all queues to be processed and workers to complete.
    """
    timeout = 300  # 5 minutes max wait
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(2)

        # Check if all queues are empty and no workers are active
        all_queues_empty = (
            q_frames.empty() and
            q_vehicles.empty() and
            q_plates.empty() and
            q_edited.empty() and
            q_ocr.empty()
        )

        active_workers = (
            vehicle_pool.get_active_workers() +
            plate_pool.get_active_workers() +
            edit_pool.get_active_workers() +
            ocr_pool.get_active_workers()
        )

        logger.info(
            f"Pipeline status - Queues empty: {all_queues_empty}, "
            f"Active workers: {active_workers}, "
            f"Elapsed: {int(time.time() - start_time)}s"
        )

        if all_queues_empty and active_workers == 0:
            logger.info("Pipeline processing complete!")
            break

    if time.time() - start_time >= timeout:
        logger.warning(f"Pipeline completion timeout after {timeout}s")


if __name__ == "__main__":
    main()