"""
Tests for services/attendance_service.py — Attendance marking, persistence, filtering.
"""
import json
import os
from datetime import date, datetime

import pytest

from database.manager import DatabaseManager
from services.attendance_service import AttendanceService

@pytest.fixture
def tmp_svc(tmp_path):
    """Create an AttendanceService backed by a temp directory via DatabaseManager."""
    db = DatabaseManager()
    db.connect(db_type="json", db_url=str(tmp_path))
    return AttendanceService(db)

# ── Mark Attendance ──────────────────────────────────────

class TestMarkAttendance:

    def test_mark_success(self, tmp_svc):
        result = tmp_svc.mark_attendance("S001", "Alice")

        assert result["success"] is True
        assert result["message"] == "Attendance marked"
        assert result["record"]["student_id"] == "S001"
        assert result["record"]["name"] == "Alice"
        assert result["record"]["date"] == str(date.today())
        assert result["record"]["status"] == "Present"

    def test_mark_includes_time(self, tmp_svc):
        result = tmp_svc.mark_attendance("S001", "Alice")

        assert "time" in result["record"]
        # Time should be in HH:MM:SS format
        time_str = result["record"]["time"]
        parts = time_str.split(":")
        assert len(parts) == 3

    def test_duplicate_prevention(self, tmp_svc):
        tmp_svc.mark_attendance("S001", "Alice")
        result = tmp_svc.mark_attendance("S001", "Alice")

        assert result["success"] is False
        assert "already marked" in result["message"]

    def test_different_students_same_day(self, tmp_svc):
        r1 = tmp_svc.mark_attendance("S001", "Alice")
        r2 = tmp_svc.mark_attendance("S002", "Bob")

        assert r1["success"] is True
        assert r2["success"] is True

    def test_multiple_students(self, tmp_svc):
        for i in range(5):
            result = tmp_svc.mark_attendance(f"S{i:03d}", f"Student_{i}")
            assert result["success"] is True

        assert len(tmp_svc.get_attendance()) == 5


# ── Persistence ──────────────────────────────────────────

class TestAttendancePersistence:

    def test_survives_reload(self, tmp_path):
        db1 = DatabaseManager()
        db1.connect(db_type="json", db_url=str(tmp_path))
        svc1 = AttendanceService(db1)
        svc1.mark_attendance("S001", "Alice")
        svc1.mark_attendance("S002", "Bob")

        # Simulate server restart
        db2 = DatabaseManager()
        db2.connect(db_type="json", db_url=str(tmp_path))
        svc2 = AttendanceService(db2)
        records = svc2.get_attendance()

        assert len(records) == 2
        assert records[0]["student_id"] == "S001"
        assert records[1]["student_id"] == "S002"

    def test_json_file_created(self, tmp_path):
        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        svc = AttendanceService(db)
        svc.mark_attendance("S001", "Alice")

        att_file = os.path.join(str(tmp_path), "attendance.json")
        assert os.path.exists(att_file)
        with open(att_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1


# ── Error Recovery ───────────────────────────────────────

class TestAttendanceErrorRecovery:

    def test_empty_file(self, tmp_path):
        att_file = os.path.join(str(tmp_path), "attendance.json")
        open(att_file, "w").close()

        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        svc = AttendanceService(db)
        assert svc.get_attendance() == []

    def test_corrupted_json(self, tmp_path):
        att_file = os.path.join(str(tmp_path), "attendance.json")
        with open(att_file, "w") as f:
            f.write("not json!")

        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        svc = AttendanceService(db)
        assert svc.get_attendance() == []

    def test_wrong_type_json(self, tmp_path):
        att_file = os.path.join(str(tmp_path), "attendance.json")
        with open(att_file, "w") as f:
            json.dump({"key": "value"}, f)

        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        svc = AttendanceService(db)
        assert svc.get_attendance() == []


# ── Filtering ────────────────────────────────────────────

class TestAttendanceFiltering:

    def test_get_today(self, tmp_svc):
        tmp_svc.mark_attendance("S001", "Alice")

        records = tmp_svc.get_today_attendance()
        assert len(records) == 1
        assert records[0]["date"] == str(date.today())

    def test_get_by_date(self, tmp_svc):
        tmp_svc.mark_attendance("S001", "Alice")
        today = str(date.today())

        records = tmp_svc.get_attendance_by_date(today)
        assert len(records) == 1

        records = tmp_svc.get_attendance_by_date("1999-01-01")
        assert len(records) == 0

    def test_get_by_student(self, tmp_svc):
        tmp_svc.mark_attendance("S001", "Alice")
        tmp_svc.mark_attendance("S002", "Bob")

        records = tmp_svc.get_attendance_by_student("S001")
        assert len(records) == 1
        assert records[0]["name"] == "Alice"

        records = tmp_svc.get_attendance_by_student("FAKE")
        assert len(records) == 0

    def test_get_all(self, tmp_svc):
        tmp_svc.mark_attendance("S001", "Alice")
        tmp_svc.mark_attendance("S002", "Bob")
        tmp_svc.mark_attendance("S003", "Charlie")

        records = tmp_svc.get_attendance()
        assert len(records) == 3
