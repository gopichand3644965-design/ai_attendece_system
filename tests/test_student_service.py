"""
Tests for services/student_service.py — Registration, matching logic, and CRUD.
Uses a mock model to avoid requiring InsightFace/GPU during testing.
"""
import numpy as np
import pytest

from database.manager import DatabaseManager
from services.student_service import StudentService

class MockFaceModel:
    """Mock FaceRecognitionService that returns predictable embeddings."""

    def __init__(self, threshold=0.39):
        self.threshold = threshold
        self._embeddings = {}  # path -> embedding

    def set_embedding(self, path, embedding):
        """Pre-configure what embedding to return for a given image path."""
        self._embeddings[path] = embedding

    def get_embedding(self, image):
        # For tests that pass a numpy array, return a fixed embedding
        if isinstance(image, np.ndarray):
            return image, "success"  # treat the image itself as the embedding
        return None, "No face detected"

class MockFaceModelForRegistration:
    """Mock that reads images from paths and returns preset embeddings."""

    def __init__(self, threshold=0.39):
        self.threshold = threshold
        self._call_count = 0
        self._embeddings = []

    def set_embeddings(self, embeddings):
        self._embeddings = embeddings
        self._call_count = 0

    def get_embedding(self, image):
        if self._call_count < len(self._embeddings):
            emb = self._embeddings[self._call_count]
            self._call_count += 1
            if emb is None:
                return None, "No face detected"
            return emb, "success"
        return None, "No face detected"

@pytest.fixture
def make_embedding():
    def _make(seed):
        rng = np.random.RandomState(seed)
        vec = rng.randn(512).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return _make

@pytest.fixture
def tmp_db(tmp_path):
    db = DatabaseManager()
    db.connect(db_type="json", db_url=str(tmp_path))
    return db


# ── Registration ─────────────────────────────────────────

class TestRegisterStudent:

    def test_register_success(self, tmp_path, tmp_db, make_embedding):
        emb1 = make_embedding(1)
        emb2 = make_embedding(2)

        model = MockFaceModelForRegistration()
        model.set_embeddings([emb1, emb2])

        svc = StudentService(model, tmp_db)

        # Create dummy image files
        import cv2
        p1 = str(tmp_path / "face1.jpg")
        p2 = str(tmp_path / "face2.jpg")
        cv2.imwrite(p1, np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.imwrite(p2, np.zeros((100, 100, 3), dtype=np.uint8))

        result = svc.register_student("S001", "Alice", [p1, p2])

        assert result["success"] is True
        assert result["embeddings_count"] == 2

        student = tmp_db.get_student("S001")
        assert student["name"] == "Alice"
        assert len(student["embeddings"]) == 2

    def test_register_duplicate_rejected(self, tmp_path, tmp_db, make_embedding):
        emb = make_embedding(1)
        model = MockFaceModelForRegistration()
        model.set_embeddings([emb])
        svc = StudentService(model, tmp_db)

        import cv2
        p = str(tmp_path / "face.jpg")
        cv2.imwrite(p, np.zeros((100, 100, 3), dtype=np.uint8))

        svc.register_student("S001", "Alice", [p])

        # Try to register again with same ID
        model.set_embeddings([emb])
        result = svc.register_student("S001", "Alice Again", [p])

        assert result["success"] is False
        assert "already exists" in result["message"]

    def test_register_no_faces_found(self, tmp_path, tmp_db):
        model = MockFaceModelForRegistration()
        model.set_embeddings([None])  # Simulates "No face detected"
        svc = StudentService(model, tmp_db)

        import cv2
        p = str(tmp_path / "face.jpg")
        cv2.imwrite(p, np.zeros((100, 100, 3), dtype=np.uint8))

        result = svc.register_student("S001", "Alice", [p])

        assert result["success"] is False
        assert "No valid face found" in result["message"]
        assert tmp_db.get_student("S001") is None


# ── Recognition ──────────────────────────────────────────

class TestRecognizeStudent:

    def test_recognize_known_student(self, tmp_db, make_embedding):
        emb1 = make_embedding(1)
        tmp_db.create_student("S001", "Alice")
        tmp_db.add_embedding("S001", emb1.tolist())

        svc = StudentService(MockFaceModel(0.39), tmp_db)
        # Passing emb1 directly (MockFaceModel treats input as embedding)
        result = svc.recognize_student(emb1)

        assert result["recognized"] is True
        assert result["student_id"] == "S001"
        assert result["name"] == "Alice"
        assert result["similarity"] >= 0.99

    def test_recognize_unknown_person(self, tmp_db, make_embedding):
        emb1 = make_embedding(1)
        tmp_db.create_student("S001", "Alice")
        tmp_db.add_embedding("S001", emb1.tolist())

        svc = StudentService(MockFaceModel(0.39), tmp_db)

        # Different embedding entirely
        emb2 = make_embedding(999)
        result = svc.recognize_student(emb2)

        assert result["recognized"] is False
        assert result["name"] == "Unknown"
        assert result.get("similarity", 0) < 0.39

    def test_recognize_empty_database(self, tmp_db, make_embedding):
        svc = StudentService(MockFaceModel(0.39), tmp_db)
        emb = make_embedding(1)
        result = svc.recognize_student(emb)
        assert result["recognized"] is False

    def test_recognize_best_match_wins(self, tmp_db, make_embedding):
        emb1 = make_embedding(1)
        emb2 = make_embedding(2)

        tmp_db.create_student("S001", "Alice")
        tmp_db.add_embedding("S001", emb1.tolist())
        tmp_db.create_student("S002", "Bob")
        tmp_db.add_embedding("S002", emb2.tolist())

        svc = StudentService(MockFaceModel(0.39), tmp_db)
        result = svc.recognize_student(emb2)

        assert result["recognized"] is True
        assert result["student_id"] == "S002"
        assert result["name"] == "Bob"


# ── CRUD Methods ─────────────────────────────────────────

class TestStudentCRUD:

    def test_list_students(self, tmp_db):
        tmp_db.create_student("S001", "Alice")
        tmp_db.create_student("S002", "Bob")
        svc = StudentService(MockFaceModel(), tmp_db)

        students = svc.list_students()
        assert len(students) == 2
        # Ensure embeddings are NOT returned in list
        assert "embeddings" not in students[0]
        assert "embeddings" not in students[1]

    def test_list_students_empty(self, tmp_db):
        svc = StudentService(MockFaceModel(), tmp_db)
        assert svc.list_students() == []

    def test_get_student(self, tmp_db):
        tmp_db.create_student("S001", "Alice")
        svc = StudentService(MockFaceModel(), tmp_db)

        student = svc.get_student("S001")
        assert student["name"] == "Alice"
        assert "embeddings" not in student

    def test_get_student_not_found(self, tmp_db):
        svc = StudentService(MockFaceModel(), tmp_db)
        assert svc.get_student("FAKE") is None

    def test_delete_student(self, tmp_db):
        tmp_db.create_student("S001", "Alice")
        svc = StudentService(MockFaceModel(), tmp_db)

        result = svc.delete_student("S001")
        assert result is True
        assert tmp_db.get_student("S001") is None

    def test_delete_nonexistent(self, tmp_db):
        svc = StudentService(MockFaceModel(), tmp_db)
        assert svc.delete_student("FAKE") is False
