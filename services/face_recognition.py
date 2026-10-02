import os
import gc
import cv2
import numpy as np


class FaceRecognitionService:

    def __init__(self, threshold=0.39):

        self.threshold = threshold

        # Lazy loading: model is loaded on first use, not at startup
        self._app = None
        self._model_name = os.environ.get("FACE_MODEL", "buffalo_s")
        self._det_size = int(os.environ.get("DET_SIZE", "320"))

    def _load_model(self):
        """Load the InsightFace model lazily on first use."""
        if self._app is not None:
            return

        # Limit ONNX Runtime threads to reduce memory
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import onnxruntime
        onnxruntime.set_default_logger_severity(3)  # Suppress warnings

        # Set ONNX session options for low memory
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        from insightface.app import FaceAnalysis

        print(f"Loading face model: {self._model_name} (det_size={self._det_size})")

        self._app = FaceAnalysis(
            name=self._model_name,
            # Only load detection + recognition (skip age/gender/landmark to save memory)
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )

        self._app.prepare(
            ctx_id=-1,  # CPU only
            det_size=(self._det_size, self._det_size)
        )

        # Force garbage collection after model loading
        gc.collect()

        print(f"Face model loaded successfully (model={self._model_name})")

    @property
    def app(self):
        """Access the model, loading it lazily if needed."""
        self._load_model()
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