import json
import sqlite3
import os
from typing import List, Dict, Any, Optional
from database.base import DatabaseAdapter

class SQLiteAdapter(DatabaseAdapter):
    def __init__(self, connection_url: str):
        # connection_url should be the path to the .db file
        self.connection_url = connection_url
        if not self.connection_url.endswith(".db"):
            self.connection_url = os.path.join(self.connection_url, "attendance.db")
        self.conn = None

    def connect(self) -> None:
        # Create directory if it doesn't exist
        db_dir = os.path.dirname(self.connection_url)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self.conn = sqlite3.connect(self.connection_url, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize schema
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                embedding TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date_str TEXT NOT NULL,
                time_str TEXT NOT NULL,
                status TEXT NOT NULL
            )
        ''')
        
        self.conn.commit()

    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()

    def test_connection(self) -> bool:
        if not self.conn:
            return False
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def create_student(self, student_id: str, name: str) -> bool:
        try:
            self.conn.execute(
                "INSERT INTO students (student_id, name) VALUES (?, ?)", 
                (student_id, name)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT student_id, name FROM students WHERE student_id = ?", 
            (student_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
            
        # Get embeddings
        emb_cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE student_id = ?", 
            (student_id,)
        )
        embeddings = [json.loads(r["embedding"]) for r in emb_cursor.fetchall()]
        
        return {
            "student_id": row["student_id"],
            "name": row["name"],
            "embeddings": embeddings
        }

    def get_students(self) -> Dict[str, Dict[str, Any]]:
        cursor = self.conn.execute("SELECT student_id, name FROM students")
        students = {}
        for row in cursor.fetchall():
            sid = row["student_id"]
            students[sid] = {
                "student_id": sid,
                "name": row["name"],
                "embeddings": []
            }
            
        emb_cursor = self.conn.execute("SELECT student_id, embedding FROM embeddings")
        for row in emb_cursor.fetchall():
            sid = row["student_id"]
            if sid in students:
                students[sid]["embeddings"].append(json.loads(row["embedding"]))
                
        return students

    def update_student(self, student_id: str, name: str) -> bool:
        cursor = self.conn.execute(
            "UPDATE students SET name = ? WHERE student_id = ?", 
            (name, student_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_student(self, student_id: str) -> bool:
        self.conn.execute("PRAGMA foreign_keys = ON")
        cursor = self.conn.execute(
            "DELETE FROM students WHERE student_id = ?", 
            (student_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def student_exists(self, student_id: str) -> bool:
        cursor = self.conn.execute(
            "SELECT 1 FROM students WHERE student_id = ?", 
            (student_id,)
        )
        return cursor.fetchone() is not None

    def add_embedding(self, student_id: str, embedding: List[float]) -> bool:
        if not self.student_exists(student_id):
            return False
        self.conn.execute(
            "INSERT INTO embeddings (student_id, embedding) VALUES (?, ?)", 
            (student_id, json.dumps(embedding))
        )
        self.conn.commit()
        return True

    def get_embeddings(self) -> Dict[str, List[List[float]]]:
        emb_cursor = self.conn.execute("SELECT student_id, embedding FROM embeddings")
        embeddings = {}
        for row in emb_cursor.fetchall():
            sid = row["student_id"]
            if sid not in embeddings:
                embeddings[sid] = []
            embeddings[sid].append(json.loads(row["embedding"]))
        return embeddings

    def get_student_embeddings(self, student_id: str) -> List[List[float]]:
        emb_cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE student_id = ?", 
            (student_id,)
        )
        return [json.loads(r["embedding"]) for r in emb_cursor.fetchall()]

    def delete_embeddings(self, student_id: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM embeddings WHERE student_id = ?", 
            (student_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def mark_attendance(self, student_id: str, name: str, date_str: str, time_str: str, status: str) -> Dict[str, Any]:
        # Check duplicate
        cursor = self.conn.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND date_str = ?", 
            (student_id, date_str)
        )
        if cursor.fetchone():
            return {"success": False, "message": "Attendance already marked", "record": {
                "student_id": student_id, "name": name, "date": date_str
            }}

        self.conn.execute(
            "INSERT INTO attendance (student_id, name, date_str, time_str, status) VALUES (?, ?, ?, ?, ?)",
            (student_id, name, date_str, time_str, status)
        )
        self.conn.commit()
        
        record = {
            "student_id": student_id,
            "name": name,
            "date": date_str,
            "time": time_str,
            "status": status
        }
        return {"success": True, "message": "Attendance marked", "record": record}

    def get_attendance(self) -> List[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT student_id, name, date_str as date, time_str as time, status FROM attendance")
        return [dict(row) for row in cursor.fetchall()]

    def get_today_attendance(self, today_str: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT student_id, name, date_str as date, time_str as time, status FROM attendance WHERE date_str = ?",
            (today_str,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_by_date(self, target_date: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT student_id, name, date_str as date, time_str as time, status FROM attendance WHERE date_str = ?",
            (target_date,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT student_id, name, date_str as date, time_str as time, status FROM attendance WHERE student_id = ?",
            (student_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
