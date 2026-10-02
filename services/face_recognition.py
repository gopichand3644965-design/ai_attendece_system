import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionService:

    def __init__(self, threshold=0.39):

        self.threshold = threshold
        self._app = None  # Lazy-loaded

        # Use environment variables for deployment flexibility
        self._model_name = os.environ.get("FACE_MODEL", "buffalo_s")
        self._det_size = int(os.environ.get("DET_SIZE", "320"))

    @property
    def app(self):
        """Lazy-load the model on first use to reduce startup memory spike."""
        if self._app is None:
            # Limit ONNX threads to reduce memory
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"

            import onnxruntime
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1

            self._app = FaceAnalysis(
                name=self._model_name,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"]
            )

            self._app.prepare(
                ctx_id=-1,
                det_size=(self._det_size, self._det_size)
            )
        return self._app

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