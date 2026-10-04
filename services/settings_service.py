import json
import os

SETTINGS_FILE = "settings.json"

class SettingsService:
    """
    Settings service that stores settings in the database (production)
    with fallback to local JSON file (local development).
    """

    @staticmethod
    def _get_db():
        """Get the database manager from the app. Returns None if unavailable."""
        try:
            from app.main import database
            if database and database._adapter:
                return database
        except (ImportError, AttributeError):
            pass
        return None

    @staticmethod
    def get_settings() -> dict:
        default_settings = {
            "attendance_start_time": "08:00",
            "attendance_end_time": "15:00"
        }

        # Try database first (production — persists across Render restarts)
        db = SettingsService._get_db()
        if db:
            try:
                return db.get_settings()
            except Exception as e:
                print(f"[SettingsService] Failed to read settings from database: {e}")

        # Fall back to local JSON file (local development)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    # Update defaults with loaded data
                    default_settings.update(data)
            except Exception:
                pass
                
        return default_settings

    @staticmethod
    def save_settings(settings: dict) -> bool:
        # Try database first (production)
        db = SettingsService._get_db()
        if db:
            try:
                return db.save_settings(settings)
            except Exception as e:
                print(f"[SettingsService] Failed to save settings to database: {e}")

        # Fall back to local JSON file
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception:
            return False
