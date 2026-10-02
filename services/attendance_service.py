from datetime import datetime, date

class AttendanceService:

    def __init__(self, database):
        self.database = database

    def mark_attendance(self, student_id, name, status="Present"):
        """
        Mark attendance for a student.
        Prevents duplicate attendance for the same student on the same date.
        """
        today = str(date.today())
        now = datetime.now().strftime("%H:%M:%S")

        return self.database.mark_attendance(student_id, name, today, now, status)

    def get_attendance(self):
        """Return all attendance records."""
        return self.database.get_attendance()

    def get_today_attendance(self):
        """Return attendance records for today."""
        today = str(date.today())
        return self.database.get_today_attendance(today)

    def close_day_attendance(self, all_students: list):
        """
        Check today's attendance against all registered students.
        Anyone not marked 'Present' will be marked as 'Absent'.
        """
        today = str(date.today())
        today_records = self.get_today_attendance()
        
        # Get set of student IDs who are present today
        present_ids = {record["student_id"] for record in today_records}
        
        absent_count = 0
        for student in all_students:
            sid = student["student_id"]
            if sid not in present_ids:
                # Mark them absent
                self.database.mark_attendance(sid, student["name"], today, "23:59:59", "Absent")
                absent_count += 1
                
        return {"success": True, "absent_marked": absent_count}

    def get_attendance_by_date(self, target_date):
        """Return attendance records for a specific date (YYYY-MM-DD)."""
        return self.database.get_attendance_by_date(target_date)

    def get_attendance_by_student(self, student_id):
        """Return all attendance records for a specific student."""
        return self.database.get_student_attendance(student_id)