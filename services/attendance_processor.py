class AttendanceProcessor:

    def __init__(self, face_model, student_service, attendance_service):

        self.face_model = face_model
        self.student_service = student_service
        self.attendance_service = attendance_service

    def process_single(self, image):
        """
        Process a single-face image for attendance.
        Detects face → extracts embedding → matches student → marks attendance.
        """

        # Step 1: Detect and extract embedding
        embedding, status = self.face_model.get_embedding(image)

        if embedding is None:
            return {
                "success": False,
                "recognized": False,
                "student_id": None,
                "name": None,
                "similarity": 0.0,
                "message": status  # Preserves actual error: "No face detected" or "Multiple faces detected"
            }

        # Step 2: Match against registered students
        result = self.student_service.recognize_student(embedding)

        if not result["recognized"]:
            return {
                "success": False,
                "recognized": False,
                "student_id": None,
                "name": "Unknown",
                "similarity": result["similarity"],
                "message": "Unknown person — not registered"
            }

        # Step 3: Mark attendance
        attendance = self.attendance_service.mark_attendance(
            result["student_id"],
            result["name"]
        )

        return {
            "success": True,
            "recognized": True,
            "student_id": result["student_id"],
            "name": result["name"],
            "similarity": result["similarity"],
            "attendance": attendance
        }

    def process_group(self, image):
        """
        Process a group image with multiple faces.
        Detects all faces → matches each → marks attendance for recognized students.
        """

        face_data = self.face_model.get_all_embeddings(image)

        if not face_data:
            return {
                "success": False,
                "message": "No faces detected in the image",
                "results": [],
                "total_faces": 0,
                "recognized_count": 0
            }

        results = []
        recognized_count = 0

        for face in face_data:

            embedding = face["embedding"]
            match = self.student_service.recognize_student(embedding)

            face_result = {
                "bbox": face["bbox"],
                "det_score": face["det_score"],
                "recognized": match["recognized"],
                "student_id": match["student_id"],
                "name": match["name"],
                "similarity": match["similarity"]
            }

            if match["recognized"]:
                attendance = self.attendance_service.mark_attendance(
                    match["student_id"],
                    match["name"]
                )
                face_result["attendance"] = attendance
                recognized_count += 1

            results.append(face_result)

        return {
            "success": True,
            "message": f"Processed {len(face_data)} face(s), recognized {recognized_count}",
            "results": results,
            "total_faces": len(face_data),
            "recognized_count": recognized_count
        }