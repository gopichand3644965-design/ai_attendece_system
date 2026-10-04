import functools
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from database.base import DatabaseAdapter


def _with_reconnect(method):
    """Decorator that auto-reconnects on database connection failure.
    
    Catches psycopg2.OperationalError (server unavailable, connection dropped)
    and psycopg2.InterfaceError (connection already closed), reconnects,
    and retries the operation once.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"[PostgresAdapter] Connection lost ({e}), reconnecting...")
            self.connect()
            return method(self, *args, **kwargs)
    return wrapper


class PostgresAdapter(DatabaseAdapter):
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.conn = None

    def connect(self) -> None:
        self.conn = psycopg2.connect(self.connection_url, cursor_factory=RealDictCursor)
        self.conn.autocommit = True
        
        # Initialize schema
        with self.conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    student_id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                    embedding REAL[] NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    date_str VARCHAR NOT NULL,
                    time_str VARCHAR NOT NULL,
                    status VARCHAR NOT NULL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    key VARCHAR(255) PRIMARY KEY,
                    value VARCHAR(1000) NOT NULL
                )
            ''')

    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()

    def test_connection(self) -> bool:
        if not self.conn or self.conn.closed:
            return False
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    @_with_reconnect
    def create_student(self, student_id: str, name: str) -> bool:
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO students (student_id, name) VALUES (%s, %s)", 
                    (student_id, name)
                )
            return True
        except psycopg2.IntegrityError:
            return False

    @_with_reconnect
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT student_id, name FROM students WHERE student_id = %s", (student_id,))
            row = cursor.fetchone()
            if not row:
                return None
                
            cursor.execute("SELECT embedding FROM embeddings WHERE student_id = %s", (student_id,))
            embeddings = [r["embedding"] for r in cursor.fetchall()]
            
        return {
            "student_id": row["student_id"],
            "name": row["name"],
            "embeddings": embeddings
        }

    @_with_reconnect
    def get_students(self) -> Dict[str, Dict[str, Any]]:
        students = {}
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT student_id, name FROM students")
            for row in cursor.fetchall():
                sid = row["student_id"]
                students[sid] = {
                    "student_id": sid,
                    "name": row["name"],
                    "embeddings": []
                }
                
            cursor.execute("SELECT student_id, embedding FROM embeddings")
            for row in cursor.fetchall():
                sid = row["student_id"]
                if sid in students:
                    students[sid]["embeddings"].append(row["embedding"])
                    
        return students

    @_with_reconnect
    def update_student(self, student_id: str, name: str) -> bool:
        with self.conn.cursor() as cursor:
            cursor.execute("UPDATE students SET name = %s WHERE student_id = %s", (name, student_id))
            return cursor.rowcount > 0

    @_with_reconnect
    def delete_student(self, student_id: str) -> bool:
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
            return cursor.rowcount > 0

    @_with_reconnect
    def student_exists(self, student_id: str) -> bool:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM students WHERE student_id = %s", (student_id,))
            return cursor.fetchone() is not None

    @_with_reconnect
    def add_embedding(self, student_id: str, embedding: List[float]) -> bool:
        if not self.student_exists(student_id):
            return False
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO embeddings (student_id, embedding) VALUES (%s, %s)", 
                (student_id, embedding)
            )
        return True

    @_with_reconnect
    def get_embeddings(self) -> Dict[str, List[List[float]]]:
        embeddings = {}
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT student_id, embedding FROM embeddings")
            for row in cursor.fetchall():
                sid = row["student_id"]
                if sid not in embeddings:
                    embeddings[sid] = []
                embeddings[sid].append(row["embedding"])
        return embeddings

    @_with_reconnect
    def get_student_embeddings(self, student_id: str) -> List[List[float]]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT embedding FROM embeddings WHERE student_id = %s", (student_id,))
            return [r["embedding"] for r in cursor.fetchall()]

    @_with_reconnect
    def delete_embeddings(self, student_id: str) -> bool:
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM embeddings WHERE student_id = %s", (student_id,))
            return cursor.rowcount > 0

    @_with_reconnect
    def mark_attendance(self, student_id: str, name: str, date_str: str, time_str: str, status: str) -> Dict[str, Any]:
        with self.conn.cursor() as cursor:
            # Check duplicate
            cursor.execute(
                "SELECT 1 FROM attendance WHERE student_id = %s AND date_str = %s", 
                (student_id, date_str)
            )
            if cursor.fetchone():
                return {"success": False, "message": "Attendance already marked", "record": {
                    "student_id": student_id, "name": name, "date": date_str
                }}

            cursor.execute(
                "INSERT INTO attendance (student_id, name, date_str, time_str, status) VALUES (%s, %s, %s, %s, %s)",
                (student_id, name, date_str, time_str, status)
            )
            
        record = {
            "student_id": student_id,
            "name": name,
            "date": date_str,
            "time": time_str,
            "status": status
        }
        return {"success": True, "message": "Attendance marked", "record": record}

    @_with_reconnect
    def get_attendance(self) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT student_id, name, date_str as date, time_str as time, status FROM attendance")
            return [dict(row) for row in cursor.fetchall()]

    @_with_reconnect
    def get_today_attendance(self, today_str: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT student_id, name, date_str as date, time_str as time, status FROM attendance WHERE date_str = %s",
                (today_str,)
            )
            return [dict(row) for row in cursor.fetchall()]

    @_with_reconnect
    def get_attendance_by_date(self, target_date: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT student_id, name, date_str as date, time_str as time, status FROM attendance WHERE date_str = %s",
                (target_date,)
            )
            return [dict(row) for row in cursor.fetchall()]

    @_with_reconnect
    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT student_id, name, date_str as date, time_str as time, status FROM attendance WHERE student_id = %s",
                (student_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # -- Settings Operations --

    @_with_reconnect
    def get_settings(self) -> dict:
        defaults = {
            "attendance_start_time": "08:00",
            "attendance_end_time": "15:00"
        }
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT key, value FROM app_settings")
            rows = cursor.fetchall()
        
        if rows:
            for row in rows:
                defaults[row["key"]] = row["value"]
        
        return defaults

    @_with_reconnect
    def save_settings(self, settings: dict) -> bool:
        with self.conn.cursor() as cursor:
            for key, value in settings.items():
                cursor.execute(
                    """INSERT INTO app_settings (key, value) VALUES (%s, %s)
                       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                    (key, str(value))
                )
        return True
