from typing import Optional
from database.base import DatabaseAdapter
from database.config import DatabaseConfig
from database.factory import DatabaseFactory

class DatabaseManager:
    """
    Manages the active database adapter and proxies calls to it.
    Allows changing the database at runtime without altering business logic.
    """
    
    def __init__(self):
        self._adapter: Optional[DatabaseAdapter] = None
        self.db_type = ""
        self.db_url = ""

    def connect(self, db_type: Optional[str] = None, db_url: Optional[str] = None) -> bool:
        """Connect to the database, optionally overriding config."""
        
        new_type = db_type or DatabaseConfig.get_database_type()
        new_url = db_url or DatabaseConfig.get_database_url()
        
        try:
            # Create the new adapter
            new_adapter = DatabaseFactory.create_adapter(new_type, new_url)
            new_adapter.connect()
            
            # Verify connection works
            if not new_adapter.test_connection():
                raise ConnectionError("Database test_connection() failed.")
                
            # If we had an old adapter, disconnect it
            if self._adapter:
                try:
                    self._adapter.disconnect()
                except Exception:
                    pass
                    
            self._adapter = new_adapter
            self.db_type = new_type
            self.db_url = new_url
            
            # Persist configuration if it was explicitly provided
            if db_type or db_url:
                DatabaseConfig.set_config(self.db_type, self.db_url)
                
            return True
            
        except Exception as e:
            # If changing connection fails, log or raise, but keep old connection if exists
            print(f"Failed to connect to database: {e}")
            raise

    def get_status(self) -> dict:
        """Return the current database connection status."""
        healthy = False
        if self._adapter:
            try:
                healthy = self._adapter.test_connection()
            except Exception:
                pass
                
        return {
            "connected": self._adapter is not None,
            "database_type": self.db_type,
            "status": "healthy" if healthy else "unhealthy"
        }

    # --- Proxy Student Operations ---

    def create_student(self, student_id: str, name: str) -> bool:
        return self._adapter.create_student(student_id, name)

    def get_student(self, student_id: str) -> dict:
        return self._adapter.get_student(student_id)

    def get_students(self) -> dict:
        return self._adapter.get_students()

    def update_student(self, student_id: str, name: str) -> bool:
        return self._adapter.update_student(student_id, name)

    def delete_student(self, student_id: str) -> bool:
        return self._adapter.delete_student(student_id)

    def student_exists(self, student_id: str) -> bool:
        return self._adapter.student_exists(student_id)

    # --- Proxy Embedding Operations ---

    def add_embedding(self, student_id: str, embedding: list) -> bool:
        return self._adapter.add_embedding(student_id, embedding)

    def get_embeddings(self) -> dict:
        return self._adapter.get_embeddings()

    def get_student_embeddings(self, student_id: str) -> list:
        return self._adapter.get_student_embeddings(student_id)

    def delete_embeddings(self, student_id: str) -> bool:
        return self._adapter.delete_embeddings(student_id)

    # --- Proxy Attendance Operations ---

    def mark_attendance(self, student_id: str, name: str, date_str: str, time_str: str, status: str) -> dict:
        return self._adapter.mark_attendance(student_id, name, date_str, time_str, status)

    def get_attendance(self) -> list:
        return self._adapter.get_attendance()

    def get_today_attendance(self, today_str: str) -> list:
        return self._adapter.get_today_attendance(today_str)

    def get_attendance_by_date(self, target_date: str) -> list:
        return self._adapter.get_attendance_by_date(target_date)

    def get_student_attendance(self, student_id: str) -> list:
        return self._adapter.get_student_attendance(student_id)
