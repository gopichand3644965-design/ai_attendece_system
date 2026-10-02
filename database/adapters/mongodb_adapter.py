import pymongo
from typing import List, Dict, Any, Optional
from database.base import DatabaseAdapter

class MongoDBAdapter(DatabaseAdapter):
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.client = None
        self.db = None
        self.students = None
        self.attendance = None

    def connect(self) -> None:
        self.client = pymongo.MongoClient(self.connection_url)
        self.db = self.client.get_database("attendance_db") if "attendance_db" not in self.connection_url else self.client.get_default_database()
        
        self.students = self.db["students"]
        self.attendance = self.db["attendance"]

        # Ensure index for student_id lookup
        self.students.create_index("student_id", unique=True)
        # Unique attendance per student per day
        self.attendance.create_index([("student_id", pymongo.ASCENDING), ("date", pymongo.ASCENDING)], unique=True)

    def disconnect(self) -> None:
        if self.client:
            self.client.close()

    def test_connection(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    def create_student(self, student_id: str, name: str) -> bool:
        try:
            self.students.insert_one({
                "student_id": student_id,
                "name": name,
                "embeddings": []
            })
            return True
        except pymongo.errors.DuplicateKeyError:
            return False

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        doc = self.students.find_one({"student_id": student_id}, {"_id": 0})
        if not doc:
            return None
        return doc

    def get_students(self) -> Dict[str, Dict[str, Any]]:
        students = {}
        for doc in self.students.find({}, {"_id": 0}):
            students[doc["student_id"]] = doc
        return students

    def update_student(self, student_id: str, name: str) -> bool:
        result = self.students.update_one(
            {"student_id": student_id},
            {"$set": {"name": name}}
        )
        return result.modified_count > 0

    def delete_student(self, student_id: str) -> bool:
        result = self.students.delete_one({"student_id": student_id})
        return result.deleted_count > 0

    def student_exists(self, student_id: str) -> bool:
        return self.students.count_documents({"student_id": student_id}, limit=1) > 0

    def add_embedding(self, student_id: str, embedding: List[float]) -> bool:
        result = self.students.update_one(
            {"student_id": student_id},
            {"$push": {"embeddings": embedding}}
        )
        return result.modified_count > 0

    def get_embeddings(self) -> Dict[str, List[List[float]]]:
        embeddings = {}
        for doc in self.students.find({}, {"_id": 0, "student_id": 1, "embeddings": 1}):
            embeddings[doc["student_id"]] = doc.get("embeddings", [])
        return embeddings

    def get_student_embeddings(self, student_id: str) -> List[List[float]]:
        doc = self.students.find_one({"student_id": student_id}, {"_id": 0, "embeddings": 1})
        if not doc:
            return []
        return doc.get("embeddings", [])

    def delete_embeddings(self, student_id: str) -> bool:
        result = self.students.update_one(
            {"student_id": student_id},
            {"$set": {"embeddings": []}}
        )
        return result.modified_count > 0

    def mark_attendance(self, student_id: str, name: str, date_str: str, time_str: str, status: str) -> Dict[str, Any]:
        record = {
            "student_id": student_id,
            "name": name,
            "date": date_str,
            "time": time_str,
            "status": status
        }
        try:
            self.attendance.insert_one(record.copy())
            record.pop("_id", None)
            return {"success": True, "message": "Attendance marked", "record": record}
        except pymongo.errors.DuplicateKeyError:
            return {"success": False, "message": "Attendance already marked", "record": {
                "student_id": student_id, "name": name, "date": date_str
            }}

    def get_attendance(self) -> List[Dict[str, Any]]:
        return list(self.attendance.find({}, {"_id": 0}))

    def get_today_attendance(self, today_str: str) -> List[Dict[str, Any]]:
        return list(self.attendance.find({"date": today_str}, {"_id": 0}))

    def get_attendance_by_date(self, target_date: str) -> List[Dict[str, Any]]:
        return list(self.attendance.find({"date": target_date}, {"_id": 0}))

    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        return list(self.attendance.find({"student_id": student_id}, {"_id": 0}))
