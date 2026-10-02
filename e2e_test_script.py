import requests
import json
import os

base = "http://127.0.0.1:8000"

# Register multiple students
people = {
    "STU001": ("Ajay", "Ajay"),
    "STU002": ("Bablu", "Bablu"),
    "STU003": ("Heman", "Heman"),
}

for sid, (name, folder) in people.items():
    img_dir = os.path.join("recognition", "Saved Searches", folder)
    files = []
    for fname in os.listdir(img_dir):
        fpath = os.path.join(img_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(("files", (fname, open(fpath, "rb"), "image/jpeg")))

    r = requests.post(f"{base}/students/register", data={"student_id": sid, "name": name}, files=files)
    res = r.json()
    ec = res.get("embeddings_count", 0)
    print(f"Registered {name}: success={res['success']} ({ec} embeddings)")
    for _, (_, f, _) in files:
        f.close()

# List all students
r = requests.get(f"{base}/students/")
print(f"\nTotal students: {r.json()['count']}")

# Test recognition - Ajay image should match Ajay
img_path = os.path.join("recognition", "Saved Searches", "Ajay", "IMG_20250526_124234.jpg")
with open(img_path, "rb") as f:
    r = requests.post(f"{base}/attendance/recognize", files={"file": ("test.jpg", f)})
res = r.json()
print(f"\nRecognize Ajay image: recognized={res['recognized']}, name={res.get('name')}, sim={res.get('similarity')}")

# Test recognition - Bablu image should match Bablu
img_path = os.path.join("recognition", "Saved Searches", "Bablu", "IMG_20250326_175506.jpg")
with open(img_path, "rb") as f:
    r = requests.post(f"{base}/attendance/recognize", files={"file": ("test.jpg", f)})
res = r.json()
print(f"Recognize Bablu image: recognized={res['recognized']}, name={res.get('name')}, sim={res.get('similarity')}")

# Test recognition - Heman image should match Heman
img_path = os.path.join("recognition", "Saved Searches", "Heman", "IMG_20260726_121543.jpg")
with open(img_path, "rb") as f:
    r = requests.post(f"{base}/attendance/recognize", files={"file": ("test.jpg", f)})
res = r.json()
print(f"Recognize Heman image: recognized={res['recognized']}, name={res.get('name')}, sim={res.get('similarity')}")

# Test recognition - Sonu (NOT registered) should be Unknown
img_path = os.path.join("recognition", "Saved Searches", "Sonu", "IMG-20250317-WA0000.jpg")
with open(img_path, "rb") as f:
    r = requests.post(f"{base}/attendance/recognize", files={"file": ("test.jpg", f)})
res = r.json()
print(f"Recognize Sonu (not registered): recognized={res['recognized']}, name={res.get('name')}, sim={res.get('similarity')}")

# Today attendance
r = requests.get(f"{base}/attendance/today")
att = r.json()
print(f"\nToday attendance: {att['count']} records")
for rec in att["records"]:
    print(f"  {rec['student_id']} | {rec['name']} | {rec['time']} | {rec['status']}")
