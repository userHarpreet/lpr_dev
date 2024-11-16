import queue
import threading
import logging


class ThreadedLogger:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.logger = logging.getLogger('ThreadedLogger')
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('vehicle_detection.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.thread = threading.Thread(target=self._logger_thread, daemon=True)
        self.thread.start()

    def _logger_thread(self):
        while True:
            record = self.log_queue.get()
            if record is None:
                break
            self.logger.handle(record)

    def log(self, level, message):
        self.log_queue.put(logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname='',
            lineno=0,
            msg=message,
            args=None,
            exc_info=None
        ))

    def debug(self, message):
        self.log(logging.DEBUG, message)

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message)

    def critical(self, message):
        self.log(logging.CRITICAL, message)

    def shutdown(self):
        self.log_queue.put(None)
        self.thread.join()
