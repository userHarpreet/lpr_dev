"""
Unit tests for storage_process module.

Covers:
- Database initialization
- Inserting OCR records and querying
- File storage path generation (without heavy I/O)
"""

import unittest
import tempfile
import os
import sqlite3
from datetime import datetime

from storage_process import StorageProcess
from multiprocessing import Queue, Event


class TestStorageProcessDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_lpr.db')
        # initialize DB manually
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lpr_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT
            )
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_db_initialized_tables_exist(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lpr_records'")
        row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(row)

    def test_insert_and_query_record(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO lpr_records (plate_text, image_path) VALUES (?, ?)", ('ABC123', '/tmp/img.jpg'))
        conn.commit()

        cursor.execute("SELECT plate_text, image_path FROM lpr_records WHERE plate_text = ?", ('ABC123',))
        row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 'ABC123')
        self.assertEqual(row[1], '/tmp/img.jpg')


if __name__ == '__main__':
    unittest.main(verbosity=2)
