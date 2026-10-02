import os
import gc

# Set memory-limiting environment variables early, before any imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import traceback

from services.face_recognition import FaceRecognitionService
from database.manager import DatabaseManager
from services.student_service import StudentService
from services.attendance_service import AttendanceService
from services.attendance_processor import AttendanceProcessor

from api.students import router as students_router
from api.attendance import router as attendance_router
from api.reports import router as reports_router
from api.database import router as database_router
from api.settings import router as settings_router

from apscheduler.schedulers.background import BackgroundScheduler
from services.settings_service import SettingsService

scheduler = BackgroundScheduler()


# 🔧 App 🔧

app = FastAPI(
    title="AI Attendance System",
    description="AI-powered student attendance system using face recognition (InsightFace / ArcFace)",
    version="1.0.0"
)


# 🔧 Service Initialization 🔧
# Face model uses lazy loading — it won't consume memory until first face recognition request

face_model = FaceRecognitionService()

database = DatabaseManager()
database.connect()

student_service = StudentService(
    face_model,
    database
)

attendance_service = AttendanceService(database)

attendance_processor = AttendanceProcessor(
    face_model,
    student_service,
    attendance_service
)


# 🔧 Routers 🔧

app.include_router(students_router)
app.include_router(attendance_router)
app.include_router(reports_router)
app.include_router(database_router)
app.include_router(settings_router)


# 🔧 Static Files 🔧

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 🔧 Background Scheduler 🔧

def run_absent_job():
    print("Running auto-absent job...")
    all_students = student_service.list_students()
    attendance_service.close_day_attendance(all_students)
    print("Auto-absent job completed.")

def schedule_auto_absent_job():
    # Remove existing job if any
    if scheduler.get_job("auto_absent"):
        scheduler.remove_job("auto_absent")
        
    settings = SettingsService.get_settings()
    end_time = settings.get("attendance_end_time", "23:59")
    
    try:
        hour, minute = map(int, end_time.split(":"))
        scheduler.add_job(
            run_absent_job, 
            "cron", 
            hour=hour, 
            minute=minute, 
            id="auto_absent",
            replace_existing=True
        )
        print(f"Scheduled auto-absent job for {hour:02d}:{minute:02d} daily.")
    except Exception as e:
        print(f"Error scheduling job: {e}")

@app.on_event("startup")
def startup_event():
    schedule_auto_absent_job()
    scheduler.start()
    # Force garbage collection after startup
    gc.collect()
    print("App started. Face model will load lazily on first recognition request.")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled Exception: {str(exc)}\n{traceback.format_exc()}"
    print(error_msg)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error", "detail": error_msg}
    )

# 🔧 Root Endpoints 🔧

@app.get("/", tags=["System"])
def home():
    """Serve the Web Dashboard."""
    return FileResponse("static/index.html")

@app.get("/ping", tags=["System"])
def ping():
    """Simple health check endpoint."""
    return {
        "status": "running",
        "message": "AI Attendance API is working",
        "docs": "/docs"
    }


@app.get("/health", tags=["System"])
def health():

    student_count = len(database.get_students())

    return {
        "status": "healthy",
        "registered_students": student_count
    }


# 🔧 Entry Point 🔧

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)