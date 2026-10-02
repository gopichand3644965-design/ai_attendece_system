import os
import tempfile

import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter(prefix="/students", tags=["Students"])


def get_student_service():
    """Get student_service from app state. Set during app startup."""
    from app.main import student_service
    return student_service


@router.post("/register")
async def register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Register a new student with multiple face images (1-10 images).
    Extracts embeddings and stores them. Original images are NOT permanently stored.
    """

    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least 1 face image is required")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per registration")

    image_paths = []

    try:
        for file in files:
            suffix = os.path.splitext(file.filename or ".jpg")[1]

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            content = await file.read()
            temp_file.write(content)
            temp_file.close()

            image_paths.append(temp_file.name)

        svc = get_student_service()
        result = svc.register_student(student_id, name, image_paths)

        return result

    finally:
        # Always clean up temp files, even on error
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)


@router.get("/")
def list_students():
    """List all registered students (without embedding data)."""

    svc = get_student_service()
    students = svc.list_students()

    return {
        "success": True,
        "count": len(students),
        "students": students
    }


@router.get("/{student_id}")
def get_student(student_id: str):
    """Get details of a specific student."""

    svc = get_student_service()
    student = svc.get_student(student_id)

    if student is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    return {
        "success": True,
        "student": student
    }


@router.delete("/{student_id}")
def delete_student(student_id: str):
    """Delete a registered student and their embeddings."""

    svc = get_student_service()
    deleted = svc.delete_student(student_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    return {
        "success": True,
        "message": f"Student '{student_id}' deleted"
    }
