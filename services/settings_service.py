import json
import os

SETTINGS_FILE = "settings.json"

class SettingsService:
    @staticmethod
    def get_settings() -> dict:
        default_settings = {
            "attendance_start_time": "08:00",
            "attendance_end_time": "15:00"
        }
        
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
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception:
            return False
