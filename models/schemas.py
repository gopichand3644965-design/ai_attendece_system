from pydantic import BaseModel
from typing import Optional


class StudentResponse(BaseModel):
    student_id: str
    name: str
    embeddings_count: int


class RecognitionResult(BaseModel):
    success: bool
    recognized: bool
    student_id: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    message: str = ""


class AttendanceRecord(BaseModel):
    student_id: str
    name: str
    date: str
    time: str
    status: str
