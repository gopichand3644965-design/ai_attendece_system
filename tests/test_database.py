"""
Tests for database/adapters/json_adapter.py via DatabaseManager.
Tests CRUD operations, persistence, error recovery, and thread safety.
"""
import json
import os
import threading

import numpy as np
import pytest

from database.manager import DatabaseManager


@pytest.fixture
def tmp_db(tmp_path):
    """Create a DatabaseManager backed by a temp directory."""
    db = DatabaseManager()
    db.connect(db_type="json", db_url=str(tmp_path))
    return db


@pytest.fixture
def sample_embeddings():
    """Return two fake 512-d normalized embeddings."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(2, 512).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return list(vecs)


# ── Basic CRUD ───────────────────────────────────────────

class TestAddStudent:

    def test_add_student_stores_data(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        for emb in sample_embeddings:
            tmp_db.add_embedding("S001", emb.tolist())

        student = tmp_db.get_student("S001")
        assert student is not None
        assert student["student_id"] == "S001"
        assert student["name"] == "Alice"
        assert len(student["embeddings"]) == 2

    def test_add_student_converts_numpy_to_list(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        for emb in sample_embeddings:
            tmp_db.add_embedding("S001", emb.tolist())

        student = tmp_db.get_student("S001")
        # Embeddings should be plain Python lists (not numpy arrays)
        assert isinstance(student["embeddings"][0], list)
        assert len(student["embeddings"][0]) == 512

    def test_update_existing(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        tmp_db.update_student("S001", "Alice Updated")

        student = tmp_db.get_student("S001")
        assert student["name"] == "Alice Updated"


class TestGetStudents:

    def test_get_students_empty(self, tmp_db):
        assert tmp_db.get_students() == {}

    def test_get_students_multiple(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        tmp_db.create_student("S002", "Bob")

        students = tmp_db.get_students()
        assert len(students) == 2
        assert "S001" in students
        assert "S002" in students


class TestGetStudent:

    def test_get_existing_student(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        assert tmp_db.get_student("S001") is not None

    def test_get_nonexistent_student(self, tmp_db):
        assert tmp_db.get_student("FAKE") is None


class TestStudentExists:

    def test_exists_true(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        assert tmp_db.student_exists("S001") is True

    def test_exists_false(self, tmp_db):
        assert tmp_db.student_exists("FAKE") is False


class TestDeleteStudent:

    def test_delete_existing(self, tmp_db, sample_embeddings):
        tmp_db.create_student("S001", "Alice")
        result = tmp_db.delete_student("S001")

        assert result is True
        assert tmp_db.get_student("S001") is None

    def test_delete_nonexistent(self, tmp_db):
        result = tmp_db.delete_student("FAKE")
        assert result is False


# ── Persistence ──────────────────────────────────────────

class TestPersistence:

    def test_data_survives_reload(self, tmp_path, sample_embeddings):
        db1 = DatabaseManager()
        db1.connect(db_type="json", db_url=str(tmp_path))
        db1.create_student("S001", "Alice")
        for emb in sample_embeddings:
            db1.add_embedding("S001", emb.tolist())

        # Simulate server restart — create a new DatabaseManager instance
        db2 = DatabaseManager()
        db2.connect(db_type="json", db_url=str(tmp_path))
        student = db2.get_student("S001")

        assert student is not None
        assert student["name"] == "Alice"
        assert len(student["embeddings"]) == 2

    def test_delete_persists_across_reload(self, tmp_path, sample_embeddings):
        db1 = DatabaseManager()
        db1.connect(db_type="json", db_url=str(tmp_path))
        db1.create_student("S001", "Alice")
        db1.delete_student("S001")

        db2 = DatabaseManager()
        db2.connect(db_type="json", db_url=str(tmp_path))
        assert db2.get_student("S001") is None


# ── Error Recovery ───────────────────────────────────────

class TestErrorRecovery:

    def test_empty_file_recovery(self, tmp_path):
        db_file = str(tmp_path / "students.json")
        open(db_file, "w").close()

        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        assert db.get_students() == {}

    def test_corrupted_json_recovery(self, tmp_path):
        db_file = str(tmp_path / "students.json")
        with open(db_file, "w") as f:
            f.write("{invalid json!!")

        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        assert db.get_students() == {}

    def test_nondict_json_recovery(self, tmp_path):
        db_file = str(tmp_path / "students.json")
        with open(db_file, "w") as f:
            json.dump([1, 2, 3], f)

        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))
        assert db.get_students() == {}


# ── Thread Safety ────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_writes(self, tmp_path, sample_embeddings):
        db = DatabaseManager()
        db.connect(db_type="json", db_url=str(tmp_path))

        errors = []

        def add_student(sid):
            try:
                db.create_student(sid, f"Student_{sid}")
                db.add_embedding(sid, sample_embeddings[0].tolist())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_student, args=(f"S{i:03d}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(db.get_students()) == 20
