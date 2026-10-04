from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class DatabaseAdapter(ABC):
    """
    Abstract base class for all database adapters.
    Defines the standard interface the application uses.
    All adapters must implement these methods and return domain models (dicts).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish the database connection."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the connection is active and healthy."""
        pass

    # -- Student Operations --

    @abstractmethod
    def create_student(self, student_id: str, name: str) -> bool:
        """Create a new student record."""
        pass

    @abstractmethod
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get a student by ID. Returns a dict or None."""
        pass

    @abstractmethod
    def get_students(self) -> Dict[str, Dict[str, Any]]:
        """Get all students. Returns a dict mapping student_id -> student dict."""
        pass

    @abstractmethod
    def update_student(self, student_id: str, name: str) -> bool:
        """Update a student's information."""
        pass

    @abstractmethod
    def delete_student(self, student_id: str) -> bool:
        """Delete a student and their associated embeddings/attendance."""
        pass

    @abstractmethod
    def student_exists(self, student_id: str) -> bool:
        """Check if a student exists."""
        pass

    # -- Embedding Operations --

    @abstractmethod
    def add_embedding(self, student_id: str, embedding: List[float]) -> bool:
        """Add a face embedding for a student."""
        pass

    @abstractmethod
    def get_embeddings(self) -> Dict[str, List[List[float]]]:
        """Get all embeddings for all students. Returns student_id -> list of embeddings."""
        pass

    @abstractmethod
    def get_student_embeddings(self, student_id: str) -> List[List[float]]:
        """Get all embeddings for a specific student."""
        pass

    @abstractmethod
    def delete_embeddings(self, student_id: str) -> bool:
        """Delete all embeddings for a specific student."""
        pass

    # -- Attendance Operations --

    @abstractmethod
    def mark_attendance(self, student_id: str, name: str, date_str: str, time_str: str, status: str) -> Dict[str, Any]:
        """Mark attendance for a student. Returns the created record dict or error info."""
        pass

    @abstractmethod
    def get_attendance(self) -> List[Dict[str, Any]]:
        """Get all attendance records."""
        pass

    @abstractmethod
    def get_today_attendance(self, today_str: str) -> List[Dict[str, Any]]:
        """Get attendance records for a specific date (today)."""
        pass

    @abstractmethod
    def get_attendance_by_date(self, target_date: str) -> List[Dict[str, Any]]:
        """Get attendance records for a specific date."""
        pass

    @abstractmethod
    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        """Get attendance records for a specific student."""
        pass

    # -- Settings Operations --

    def get_settings(self) -> Dict[str, Any]:
        """Get application settings. Override for persistent storage."""
        return {
            "attendance_start_time": "08:00",
            "attendance_end_time": "15:00"
        }

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save application settings. Override for persistent storage."""
        return False
