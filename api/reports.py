import csv
import io
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/reports", tags=["Reports"])


def get_attendance_service():
    from app.main import attendance_service
    return attendance_service


@router.get("/attendance")
def get_attendance_report(
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    student_id: str = Query(None, description="Filter by student ID")
):
    """
    Get filtered attendance report.
    Can filter by date range and/or student ID.
    """

    svc = get_attendance_service()
    records = svc.get_attendance()

    # Apply filters
    if start_date:
        records = [r for r in records if r["date"] >= start_date]

    if end_date:
        records = [r for r in records if r["date"] <= end_date]

    if student_id:
        records = [r for r in records if r["student_id"] == student_id]

    return {
        "success": True,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "student_id": student_id
        },
        "count": len(records),
        "records": records
    }


@router.get("/export")
def export_attendance_csv(
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    student_id: str = Query(None, description="Filter by student ID")
):
    """
    Export attendance records as a downloadable CSV file.
    Supports same filters as the report endpoint.
    """

    svc = get_attendance_service()
    records = svc.get_attendance()

    # Apply filters
    if start_date:
        records = [r for r in records if r["date"] >= start_date]

    if end_date:
        records = [r for r in records if r["date"] <= end_date]

    if student_id:
        records = [r for r in records if r["student_id"] == student_id]

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Student ID", "Name", "Date", "Time", "Status"])

    # Data rows
    for record in records:
        writer.writerow([
            record.get("student_id", ""),
            record.get("name", ""),
            record.get("date", ""),
            record.get("time", ""),
            record.get("status", "")
        ])

    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=attendance_report.csv"
        }
    )
