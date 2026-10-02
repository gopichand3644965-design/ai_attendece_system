import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/attendance", tags=["Attendance"])


def get_attendance_processor():
    from app.main import attendance_processor
    return attendance_processor


def get_attendance_service():
    from app.main import attendance_service
    return attendance_service


def _decode_image(contents: bytes):
    """Decode uploaded image bytes into OpenCV image array."""
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return image


@router.post("/recognize")
async def recognize_attendance(file: UploadFile = File(...)):
    """
    Upload a single-face image to recognize and mark attendance.
    Returns recognition result and attendance status.
    """
    contents = await file.read()
    image = _decode_image(contents)

    processor = get_attendance_processor()
    result = processor.process_single(image)

    return result


@router.post("/recognize-group")
async def recognize_group_attendance(file: UploadFile = File(...)):
    """
    Upload a group photo with multiple faces.
    Recognizes each face and marks attendance for known students.
    """
    contents = await file.read()
    image = _decode_image(contents)

    processor = get_attendance_processor()
    result = processor.process_group(image)

    return result


@router.get("/")
def get_all_attendance():
    """Get all attendance records."""
    svc = get_attendance_service()
    records = svc.get_attendance()

    return {
        "success": True,
        "count": len(records),
        "records": records
    }


@router.get("/today")
def get_today_attendance():
    """Get today's attendance records."""
    svc = get_attendance_service()
    records = svc.get_today_attendance()

    return {
        "success": True,
        "count": len(records),
        "records": records
    }


@router.get("/date/{target_date}")
def get_attendance_by_date(target_date: str):
    """Get attendance records for a specific date (YYYY-MM-DD)."""
    svc = get_attendance_service()
    records = svc.get_attendance_by_date(target_date)

    return {
        "success": True,
        "date": target_date,
        "count": len(records),
        "records": records
    }


@router.get("/student/{student_id}")
def get_attendance_by_student(student_id: str):
    """Get all attendance records for a specific student."""
    svc = get_attendance_service()
    records = svc.get_attendance_by_student(student_id)

    return {
        "success": True,
        "student_id": student_id,
        "count": len(records),
        "records": records
    }

@router.post("/close-day")
def close_day():
    """
    Manually trigger the end-of-day attendance processing.
    Marks all students as 'Absent' if they haven't been marked 'Present' today.
    """
    from app.main import student_service
    
    svc = get_attendance_service()
    all_students = student_service.list_students()
    
    result = svc.close_day_attendance(all_students)
    return result
