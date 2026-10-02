from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/database", tags=["Database"])

class ConnectRequest(BaseModel):
    database_type: str
    connection_url: str

def get_database_manager():
    from app.main import database
    return database

@router.post("/connect")
def connect_database(request: ConnectRequest):
    """
    Connect to a new database at runtime without restarting the application.
    Supports switching between different database adapters (e.g., json, sqlite, postgresql).
    """
    db = get_database_manager()
    
    try:
        success = db.connect(db_type=request.database_type, db_url=request.connection_url)
        if success:
            return {
                "success": True,
                "database_type": request.database_type,
                "status": "connected"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect: {str(e)}")
        
    raise HTTPException(status_code=500, detail="Unknown error connecting to database.")

@router.get("/status")
def database_status():
    """
    Get the current active database configuration and connection health.
    Does not expose sensitive credentials.
    """
    db = get_database_manager()
    return db.get_status()
