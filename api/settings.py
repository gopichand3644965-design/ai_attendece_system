from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.settings_service import SettingsService

router = APIRouter(prefix="/settings", tags=["Settings"])

class TimeSettings(BaseModel):
    attendance_start_time: str
    attendance_end_time: str

@router.get("/")
def get_settings():
    return SettingsService.get_settings()

@router.post("/")
def update_settings(settings: TimeSettings):
    success = SettingsService.save_settings(settings.dict())
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")
    
    # Notify scheduler to reload settings
    try:
        from app.main import schedule_auto_absent_job
        schedule_auto_absent_job()
    except Exception as e:
        print(f"Failed to reschedule job: {e}")
        
    return {"success": True, "message": "Settings updated successfully"}
