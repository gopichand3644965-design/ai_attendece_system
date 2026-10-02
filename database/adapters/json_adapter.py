import json
import os
import threading
from typing import List, Dict, Any, Optional
from database.base import DatabaseAdapter

class JSONAdapter(DatabaseAdapter):
    def __init__(self, connection_url: str):
        # connection_url can be a directory path or 'local'
        if connection_url == "local" or not connection_url:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = base_dir
        else:
            self.data_dir = connection_url

        self.students_file = os.path.join(self.data_dir, "students.json")
        self.attendance_file = os.path.join(self.data_dir, "attendance.json")
        
        self.students = {}
        self.attendance = []
        
        self._lock = threading.Lock()

    def connect(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        self.students = self._load_json(self.students_file, {})
        self.attendance = self._load_json(self.attendance_file, [])

    def disconnect(self) -> None:
        pass

    def test_connection(self) -> bool:
        return os.path.exists(self.data_dir)

    def _load_json(self, file_path: str, default_val: Any) -> Any:
        if not os.path.exists(file_path):
            self._save_json(file_path, default_val)
            return default_val
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                if type(data) != type(default_val):
                    return default_val
                return data
        except (json.JSONDecodeError, ValueError):
            self._save_json(file_path, default_val)
            return default_val

    def _save_json(self, file_path: str, data: Any):
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def _save_students(self):
        self._save_json(self.students_file, self.students)

    def _save_attendance(self):
        self._save_json(self.attendance_file, self.attendance)

    # -- Student Ops --

    def create_student(self, student_id: str, name: str) -> bool:
        with self._lock:
            if student_id in self.students:
                return False
            self.students[student_id] = {
                "student_id": student_id,
                "name": name,
                "embeddings": []
            }
            self._save_students()
            return True

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        return self.students.get(student_id)

    def get_students(self) -> Dict[str, Dict[str, Any]]:
        return self.students

    def update_student(self, student_id: str, name: str) -> bool:
        with self._lock:
            if student_id not in self.students:
                return False
            self.students[student_id]["name"] = name
            self._save_students()
            return True

    def delete_student(self, student_id: str) -> bool:
        with self._lock:
            if student_id in self.students:
                del self.students[student_id]
                self._save_students()
                return True
            return False

    def student_exists(self, student_id: str) -> bool:
        return student_id in self.students

    # -- Embedding Ops --

    def add_embedding(self, student_id: str, embedding: List[float]) -> bool:
        with self._lock:
            if student_id not in self.students:
                return False
            self.students[student_id]["embeddings"].append(embedding)
            self._save_students()
            return True

    def get_embeddings(self) -> Dict[str, List[List[float]]]:
        return {sid: s.get("embeddings", []) for sid, s in self.students.items()}

    def get_student_embeddings(self, student_id: str) -> List[List[float]]:
        student = self.students.get(student_id)
        if not student:
            return []
        return student.get("embeddings", [])

    def delete_embeddings(self, student_id: str) -> bool:
        with self._lock:
            if student_id in self.students:
                self.students[student_id]["embeddings"] = []
                self._save_students()
                return True
            return False

    # -- Attendance Ops --

    def mark_attendance(self, student_id: str, name: str, date_str: str, time_str: str, status: str) -> Dict[str, Any]:
        with self._lock:
            # Check duplicate
            for record in self.attendance:
                if record.get("student_id") == student_id and record.get("date") == date_str:
                    return {"success": False, "message": "Attendance already marked", "record": record}

            record = {
                "student_id": student_id,
                "name": name,
                "date": date_str,
                "time": time_str,
                "status": status
            }
            self.attendance.append(record)
            self._save_attendance()
            return {"success": True, "message": "Attendance marked", "record": record}

    def get_attendance(self) -> List[Dict[str, Any]]:
        return self.attendance

    def get_today_attendance(self, today_str: str) -> List[Dict[str, Any]]:
        return [r for r in self.attendance if r.get("date") == today_str]

    def get_attendance_by_date(self, target_date: str) -> List[Dict[str, Any]]:
        return [r for r in self.attendance if r.get("date") == target_date]

    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.attendance if r.get("student_id") == student_id]
