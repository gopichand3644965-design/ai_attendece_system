import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionService:

    def __init__(self, threshold=0.39):

        self.threshold = threshold

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_embedding(self, image):
        """
        Extract face embedding from an image with exactly ONE face.
        Used for student registration (one face per image).

        Returns:
            (embedding, "success") if exactly one face found
            (None, error_message) otherwise
        """

        faces = self.app.get(image)

        if len(faces) == 0:
            return None, "No face detected"

        if len(faces) > 1:
            return None, "Multiple faces detected. Please use a single-face image."

        embedding = faces[0].embedding

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding, "success"

    def get_all_embeddings(self, image):
        """
        Extract face embeddings from ALL faces in an image.
        Used for group attendance (multiple faces per image).

        Returns:
            list of dicts with keys:
                - embedding: normalized embedding vector
                - bbox: [x1, y1, x2, y2] bounding box
                - det_score: detection confidence
            Empty list if no faces detected.
        """

        faces = self.app.get(image)

        if len(faces) == 0:
            return []

        results = []

        for face in faces:

            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            results.append({
                "embedding": embedding,
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score)
            })

        return results