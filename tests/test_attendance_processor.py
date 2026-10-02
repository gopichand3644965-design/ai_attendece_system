"""
Tests for services/attendance_processor.py — Pipeline orchestration.
Uses mock models to avoid requiring InsightFace.
"""
import numpy as np
import pytest

from database.manager import DatabaseManager
from services.student_service import StudentService
from services.attendance_service import AttendanceService
from services.attendance_processor import AttendanceProcessor


class MockFaceModel:
    """Mock face model for processor tests."""

    def __init__(self, threshold=0.39):
        self.threshold = threshold

    def get_embedding(self, image):
        """Return the image itself as embedding (for single-face tests)."""
        if image is None:
            return None, "No face detected"
        return image, "success"

    def get_all_embeddings(self, image):
        """Return pre-set face data for group tests."""
        if hasattr(self, "_group_data"):
            return self._group_data
        return []

    def set_group_data(self, data):
        self._group_data = data


@pytest.fixture
def make_embedding():
    def _make(seed):
        rng = np.random.RandomState(seed)
        vec = rng.randn(512).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return _make


@pytest.fixture
def setup(tmp_path, make_embedding):
    """Set up all services with mock model."""
    db = DatabaseManager()
    db.connect(db_type="json", db_url=str(tmp_path))

    model = MockFaceModel(threshold=0.39)
    student_svc = StudentService(model, db)
    att_svc = AttendanceService(db)
    processor = AttendanceProcessor(model, student_svc, att_svc)

    # Register a known student
    emb = make_embedding(42)
    db.create_student("S001", "Alice")
    db.add_embedding("S001", emb.tolist())

    return {
        "processor": processor,
        "model": model,
        "db": db,
        "att_svc": att_svc,
        "student_svc": student_svc,
        "alice_emb": emb,
        "make_embedding": make_embedding,
    }


# ── Single Face ──────────────────────────────────────────

class TestProcessSingle:

    def test_recognize_known_student(self, setup):
        proc = setup["processor"]
        emb = setup["alice_emb"]

        result = proc.process_single(emb)  # pass embedding as "image"

        assert result["success"] is True
        assert result["recognized"] is True
        assert result["student_id"] == "S001"
        assert result["name"] == "Alice"
        assert result["similarity"] >= 0.99

    def test_unknown_person(self, setup):
        proc = setup["processor"]
        unknown_emb = setup["make_embedding"](999)

        result = proc.process_single(unknown_emb)

        assert result["success"] is False
        assert result["recognized"] is False
        assert result["name"] == "Unknown"

    def test_no_face_detected(self, setup):
        proc = setup["processor"]

        result = proc.process_single(None)  # triggers "No face detected"

        assert result["success"] is False
        assert result["recognized"] is False
        assert result["message"] == "No face detected"

    def test_duplicate_attendance_prevention(self, setup):
        proc = setup["processor"]
        emb = setup["alice_emb"]

        r1 = proc.process_single(emb)
        r2 = proc.process_single(emb)

        assert r1["attendance"]["success"] is True
        assert r2["attendance"]["success"] is False
        assert "already marked" in r2["attendance"]["message"]


# ── Group Processing ─────────────────────────────────────

class TestProcessGroup:

    def test_group_with_known_faces(self, setup):
        proc = setup["processor"]
        model = setup["model"]
        db = setup["db"]
        mk = setup["make_embedding"]

        emb_bob = mk(100)
        db.create_student("S002", "Bob")
        db.add_embedding("S002", emb_bob.tolist())

        # Set up group data — two known faces
        model.set_group_data([
            {"embedding": setup["alice_emb"], "bbox": [10, 10, 100, 100], "det_score": 0.99},
            {"embedding": emb_bob, "bbox": [200, 10, 300, 100], "det_score": 0.98},
        ])

        result = proc.process_group(None)  # image doesn't matter for mock

        assert result["success"] is True
        assert result["total_faces"] == 2
        assert result["recognized_count"] == 2

    def test_group_with_unknown_faces(self, setup):
        proc = setup["processor"]
        model = setup["model"]
        mk = setup["make_embedding"]

        unknown = mk(888)
        model.set_group_data([
            {"embedding": unknown, "bbox": [10, 10, 100, 100], "det_score": 0.95},
        ])

        result = proc.process_group(None)

        assert result["success"] is True
        assert result["total_faces"] == 1
        assert result["recognized_count"] == 0
        assert result["results"][0]["recognized"] is False

    def test_group_mixed(self, setup):
        proc = setup["processor"]
        model = setup["model"]
        mk = setup["make_embedding"]

        unknown = mk(777)
        model.set_group_data([
            {"embedding": setup["alice_emb"], "bbox": [10, 10, 100, 100], "det_score": 0.99},
            {"embedding": unknown, "bbox": [200, 10, 300, 100], "det_score": 0.90},
        ])

        result = proc.process_group(None)

        assert result["total_faces"] == 2
        assert result["recognized_count"] == 1

    def test_group_no_faces(self, setup):
        proc = setup["processor"]
        model = setup["model"]
        model.set_group_data([])

        result = proc.process_group(None)

        assert result["success"] is False
        assert result["total_faces"] == 0
