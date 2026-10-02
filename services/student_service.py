import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class StudentService:

    def __init__(self, face_model, database):

        self.face_model = face_model
        self.database = database

    def register_student(self, student_id, name, image_paths):
        """
        Register a student by extracting face embeddings from provided images.
        Stores embeddings in the database. Original images are NOT kept.
        """

        # Check for duplicate student ID
        if self.database.student_exists(student_id):
            return {
                "success": False,
                "message": f"Student ID '{student_id}' already exists. Use a different ID or delete the existing student first."
            }

        embeddings = []

        for image_path in image_paths:

            image = cv2.imread(image_path)

            if image is None:
                continue

            embedding, status = self.face_model.get_embedding(image)

            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) == 0:
            return {
                "success": False,
                "message": "No valid face found in any of the provided images"
            }

        # Create the student record
        created = self.database.create_student(student_id, name)
        if not created:
            return {
                "success": False,
                "message": f"Failed to create student '{student_id}'"
            }

        # Add all extracted embeddings
        for embedding in embeddings:
            self.database.add_embedding(student_id, embedding.tolist())

        return {
            "success": True,
            "student_id": student_id,
            "name": name,
            "embeddings_count": len(embeddings),
            "message": f"Student '{name}' registered with {len(embeddings)} face embedding(s)"
        }

    def recognize_student(self, embedding):
        """
        Match a single embedding against all registered students.
        Uses vectorized dot product for performance (embeddings are L2-normalized,
        so dot product == cosine similarity).

        Args:
            embedding: normalized face embedding vector

        Returns:
            dict with recognition result
        """

        students = self.database.get_students()

        if not students:
            return {
                "recognized": False,
                "student_id": None,
                "name": "Unknown",
                "similarity": 0.0,
                "message": "No students registered in database"
            }

        best_student = None
        best_similarity = -1.0

        for student_id, student in students.items():

            stored_embeddings = student["embeddings"]

            if not stored_embeddings:
                continue

            # Vectorized: compute dot product of query against all stored embeddings at once
            stored_matrix = np.array(stored_embeddings)
            similarities = np.dot(stored_matrix, embedding)

            max_sim = float(np.max(similarities))

            if max_sim > best_similarity:
                best_similarity = max_sim
                best_student = student

        if best_student is not None and best_similarity >= self.face_model.threshold:

            return {
                "recognized": True,
                "student_id": best_student["student_id"],
                "name": best_student["name"],
                "similarity": round(best_similarity, 4),
                "message": "Student recognized"
            }

        return {
            "recognized": False,
            "student_id": None,
            "name": "Unknown",
            "similarity": round(best_similarity, 4) if best_similarity > -1.0 else 0.0,
            "message": "Unknown person"
        }

    def list_students(self):
        """Return a list of all registered students (without embedding data)."""

        students = self.database.get_students()

        result = []
        for student_id, student in students.items():
            result.append({
                "student_id": student["student_id"],
                "name": student["name"],
                "embeddings_count": len(student.get("embeddings", []))
            })

        return result

    def get_student(self, student_id):
        """Return info for a single student (without embedding data)."""

        student = self.database.get_student(student_id)

        if student is None:
            return None

        return {
            "student_id": student["student_id"],
            "name": student["name"],
            "embeddings_count": len(student.get("embeddings", []))
        }

    def delete_student(self, student_id):
        """Delete a student by ID."""

        return self.database.delete_student(student_id)