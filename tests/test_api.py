"""
Integration tests for FastAPI API endpoints.
Uses TestClient (no real server needed), but requires InsightFace model to be available.
These tests use mock services to avoid GPU/model dependency.
"""
import json
import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def make_embedding():
    def _make(seed):
        rng = np.random.RandomState(seed)
        vec = rng.randn(512).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return _make


@pytest.fixture
def test_app(tmp_path, make_embedding):
    """Create a FastAPI app with mock services (no InsightFace needed)."""
    from fastapi import FastAPI

    from database.manager import DatabaseManager
    from services.student_service import StudentService
    from services.attendance_service import AttendanceService
    from services.attendance_processor import AttendanceProcessor

    from api.students import router as students_router
    from api.attendance import router as attendance_router
    from api.reports import router as reports_router
    from api.database import router as database_router

    class MockFaceModel:
        def __init__(self):
            self.threshold = 0.39

        def get_embedding(self, image):
            # Return a deterministic embedding based on image hash
            if image is None:
                return None, "No face detected"
            # Use first few pixel values as seed
            seed = int(np.sum(image[:5, :5])) % 10000
            rng = np.random.RandomState(seed)
            vec = rng.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec, "success"

        def get_all_embeddings(self, image):
            emb, status = self.get_embedding(image)
            if emb is None:
                return []
            return [{"embedding": emb, "bbox": [0, 0, 100, 100], "det_score": 0.99}]

    app = FastAPI()

    # Create services with temp storage
    face_model = MockFaceModel()
    database = DatabaseManager()
    database.connect(db_type="json", db_url=str(tmp_path))
    
    student_service = StudentService(face_model, database)
    attendance_service = AttendanceService(database)
    attendance_processor = AttendanceProcessor(face_model, student_service, attendance_service)

    # Monkey-patch the module-level imports used by routers
    import app.main as main_module
    main_module.database = database
    main_module.student_service = student_service
    main_module.attendance_service = attendance_service
    main_module.attendance_processor = attendance_processor

    app.include_router(students_router)
    app.include_router(attendance_router)
    app.include_router(reports_router)
    app.include_router(database_router)

    @app.get("/")
    def home():
        return {"status": "running"}

    @app.get("/health")
    def health():
        return {"status": "healthy", "registered_students": len(database.get_students())}

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def _create_test_image():
    """Create a simple test JPEG in memory."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [255, 200, 150]  # colored rectangle
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── System Endpoints ─────────────────────────────────────

class TestSystemEndpoints:

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        assert r.json()["registered_students"] == 0


# ── Student Endpoints ────────────────────────────────────

class TestStudentEndpoints:

    def test_list_empty(self, client):
        r = client.get("/students/")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_register_student(self, client):
        img = _create_test_image()

        r = client.post(
            "/students/register",
            data={"student_id": "S001", "name": "Alice"},
            files=[("files", ("face1.jpg", img, "image/jpeg"))]
        )

        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["student_id"] == "S001"
        assert body["embeddings_count"] >= 1

    def test_register_and_list(self, client):
        img = _create_test_image()

        client.post(
            "/students/register",
            data={"student_id": "S001", "name": "Alice"},
            files=[("files", ("face.jpg", img, "image/jpeg"))]
        )

        r = client.get("/students/")
        assert r.json()["count"] == 1
        assert r.json()["students"][0]["name"] == "Alice"

    def test_get_student(self, client):
        img = _create_test_image()
        client.post(
            "/students/register",
            data={"student_id": "S001", "name": "Alice"},
            files=[("files", ("face.jpg", img, "image/jpeg"))]
        )

        r = client.get("/students/S001")
        assert r.status_code == 200
        assert r.json()["student"]["name"] == "Alice"

    def test_get_student_not_found(self, client):
        r = client.get("/students/FAKE")
        assert r.status_code == 404

    def test_delete_student(self, client):
        img = _create_test_image()
        client.post(
            "/students/register",
            data={"student_id": "S001", "name": "Alice"},
            files=[("files", ("face.jpg", img, "image/jpeg"))]
        )

        r = client.delete("/students/S001")
        assert r.status_code == 200
        assert r.json()["success"] is True

        r = client.get("/students/")
        assert r.json()["count"] == 0

    def test_delete_not_found(self, client):
        r = client.delete("/students/FAKE")
        assert r.status_code == 404

    def test_duplicate_registration(self, client):
        img = _create_test_image()

        client.post(
            "/students/register",
            data={"student_id": "S001", "name": "Alice"},
            files=[("files", ("face.jpg", img, "image/jpeg"))]
        )

        r = client.post(
            "/students/register",
            data={"student_id": "S001", "name": "Alice Again"},
            files=[("files", ("face.jpg", img, "image/jpeg"))]
        )

        assert r.json()["success"] is False
        assert "already exists" in r.json()["message"]


# ── Attendance Endpoints ─────────────────────────────────

class TestAttendanceEndpoints:

    def test_attendance_empty(self, client):
        r = client.get("/attendance/")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_today_empty(self, client):
        r = client.get("/attendance/today")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_attendance_by_date(self, client):
        r = client.get("/attendance/date/2026-01-01")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_attendance_by_student(self, client):
        r = client.get("/attendance/student/S001")
        assert r.status_code == 200
        assert r.json()["count"] == 0


# ── Report Endpoints ─────────────────────────────────────

class TestReportEndpoints:

    def test_report_empty(self, client):
        r = client.get("/reports/attendance")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_report_with_filters(self, client):
        r = client.get("/reports/attendance?start_date=2026-01-01&end_date=2026-12-31")
        assert r.status_code == 200

    def test_csv_export_empty(self, client):
        r = client.get("/reports/export")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        # Should have header row at minimum
        assert "Student ID" in r.text

    def test_csv_export_header(self, client):
        r = client.get("/reports/export")
        lines = r.text.strip().split("\n")
        assert len(lines) >= 1
        header = lines[0]
        assert "Student ID" in header
        assert "Name" in header
        assert "Date" in header
        assert "Time" in header
        assert "Status" in header

# ── Database Endpoints ───────────────────────────────────

class TestDatabaseEndpoints:

    def test_status(self, client):
        r = client.get("/database/status")
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is True
        assert body["database_type"] == "json"
        assert body["status"] == "healthy"

    def test_connect_sqlite(self, client, tmp_path):
        sqlite_file = str(tmp_path / "test.db")
        r = client.post("/database/connect", json={
            "database_type": "sqlite",
            "connection_url": sqlite_file
        })
        assert r.status_code == 200
        
        # Verify status changed
        r = client.get("/database/status")
        assert r.json()["database_type"] == "sqlite"
