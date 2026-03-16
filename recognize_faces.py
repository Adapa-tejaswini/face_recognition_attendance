import cv2
import face_recognition
import os
import numpy as np
import pickle
import datetime
import csv

# ---------------- CONFIGURATION ----------------
DATASET_DIR    = "dataset_face"
ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE      = 0.5
RESIZE_SCALE   = 0.5

# ---------------- LOAD KNOWN FACES ----------------
known_face_encodings = []
known_face_names     = []

print("=" * 50)
print("   FACE RECOGNITION ATTENDANCE SYSTEM")
print("=" * 50)
print("\n🔄 Loading face encodings...")
print("-" * 40)

def load_image_properly(image_path):
    """Load image using OpenCV -> proper RGB uint8 contiguous array"""
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    return img_rgb

# ---- Load from pickle ----
if os.path.exists(ENCODINGS_FILE):
    print(f"📂 Loading from '{ENCODINGS_FILE}'...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names     = data["names"]
    print(f"✅ Loaded {len(known_face_encodings)} encoding(s)")
else:
    print(f"❌ '{ENCODINGS_FILE}' not found! Run encode_faces.py first.")
    exit()

print(f"👥 Persons: {sorted(set(known_face_names))}")
print("-" * 40)

if len(known_face_encodings) == 0:
    print("❌ No encodings found!")
    exit()

# ---------------- ATTENDANCE LOGIC ----------------
attendance_marked    = set()  # Names to show on screen
already_marked_today = set()  # Internal tracker to prevent duplicates

def load_todays_attendance():
    """
    Reads the CSV file on startup to find who already attended TODAY.
    Automatically ignores records from previous days.
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.exists(ATTENDANCE_FILE):
        print(f"ℹ️  No attendance file found yet. Starting fresh.")
        return

    count = 0
    try:
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            
            for row in reader:
                # Row format: [Name, Date, Time, Status]
                if len(row) >= 2:
                    name = row[0]
                    date = row[1]
                    
                    # ONLY load if the date matches TODAY
                    if date == today:
                        unique_key = f"{name}_{today}"
                        already_marked_today.add(unique_key)
                        attendance_marked.add(name)
                        count += 1
    except Exception as e:
        print(f"⚠️ Error reading CSV: {e}")

    if count > 0:
        print(f"📋 Loaded {count} existing records for today ({today}):")
        for name in sorted(attendance_marked):
            print(f"   ✅ {name}")
    else:
        print(f"ℹ️  No attendance recorded for today yet.")

def mark_attendance(name):
    """Marks attendance ONLY if not already marked today"""
    if name == "Unknown":
        return

    now        = datetime.datetime.now()
    date_str   = now.strftime("%Y-%m-%d")
    time_str   = now.strftime("%H:%M:%S")
    unique_key = f"{name}_{date_str}"

    # Check if already marked
    if unique_key in already_marked_today:
        return

    # Write to CSV
    exists = os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Name", "Date", "Time", "Status"])
        writer.writerow([name, date_str, time_str, "Present"])

    # Update memory sets
    already_marked_today.add(unique_key)
    attendance_marked.add(name)
    
    print(f"✅ NEW Attendance: {name} at {time_str}")

def reset_attendance_for_today():
    """Removes today's entries from CSV and clears memory"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.exists(ATTENDANCE_FILE):
        print("⚠️  No file to reset.")
        return

    rows_to_keep = []
    removed_count = 0

    try:
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                rows_to_keep.append(header)
            
            for row in reader:
                if len(row) >= 2:
                    # Keep row ONLY if date is NOT today
                    if row[1] != today:
                        rows_to_keep.append(row)
                    else:
                        removed_count += 1
        
        # Write back the filtered data
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_keep)
        
        # Clear memory sets
        attendance_marked.clear()
        already_marked_today.clear()
        
        print(f"🗑️  RESET COMPLETE: Removed {removed_count} entries for today.")
        
    except Exception as e:
        print(f"❌ Error resetting: {e}")

# 👇 Load existing attendance on startup
load_todays_attendance()

# ---------------- CAMERA ----------------
print("\n📷 Starting camera...")
print("📌 Controls: [Q] Quit  |  [R] Reset Today's Attendance")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Camera ready!")
print("=" * 40)

frame_count = 0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed!")
        break

    frame_count += 1
    display_frame = frame.copy()

    if frame_count % 2 == 0:
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

        # Resize for speed
        h, w = rgb_frame.shape[:2]
        small_frame = cv2.resize(
            rgb_frame,
            (int(w * RESIZE_SCALE), int(h * RESIZE_SCALE))
        )
        small_frame = np.ascontiguousarray(small_frame, dtype=np.uint8)

        try:
            face_locations = face_recognition.face_locations(
                small_frame, model="hog"
            )
            face_encodings_list = face_recognition.face_encodings(
                small_frame, face_locations
            )
        except RuntimeError:
            continue

        for (top, right, bottom, left), face_enc in zip(
            face_locations, face_encodings_list
        ):
            # Scale back
            top    = int(top    / RESIZE_SCALE)
            right  = int(right  / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)
            left   = int(left   / RESIZE_SCALE)

            name       = "Unknown"
            confidence = ""
            color      = (0, 0, 255)  # Red

            matches        = face_recognition.compare_faces(
                known_face_encodings, face_enc, tolerance=TOLERANCE
            )
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_enc
            )

            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name       = known_face_names[best_idx]
                    color      = (0, 255, 0)  # Green
                    score      = (1 - face_distances[best_idx]) * 100
                    confidence = f"{score:.1f}%"
                    mark_attendance(name)

            # Draw Box
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Draw Label Background
            cv2.rectangle(
                display_frame,
                (left, bottom),
                (right, bottom + 40),
                color,
                cv2.FILLED
            )

            # Draw Name
            cv2.putText(
                display_frame, name,
                (left + 6, bottom + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

            # Draw Confidence
            if confidence:
                cv2.putText(
                    display_frame, confidence,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2
                )

    # ---------------- INFO PANEL ----------------
    total_persons = len(set(known_face_names))
    present_count = len(attendance_marked)

    # Dark background for info
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0,0), (320, 100 + present_count*30), (0,0,0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

    cv2.putText(
        display_frame,
        f"Present: {present_count}/{total_persons}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2
    )

    time_now = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(
        display_frame,
        f"Time: {time_now}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
    )

    # List of present people
    y = 90
    for person in sorted(attendance_marked):
        cv2.putText(
            display_frame, f"+ {person}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
        y += 30

    # Controls Hint
    cv2.putText(
        display_frame, "[Q] Quit  [R] Reset",
        (display_frame.shape[1] - 230, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )

    cv2.imshow("Face Recognition Attendance", display_frame)

    # ---------------- KEY CONTROLS ----------------
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to Quit
    if key == ord("q"):
        break
    
    # Press 'r' to Reset Today's Attendance
    if key == ord("r"):
        reset_attendance_for_today()

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()

# ---------------- FINAL SUMMARY ----------------
print("\n" + "=" * 40)
print("📊 FINAL ATTENDANCE SUMMARY")
print("=" * 40)

if attendance_marked:
    for i, p in enumerate(sorted(attendance_marked), 1):
        print(f"   {i}. {p} ✅")
else:
    print("   No attendance recorded.")

print(f"\n📁 Saved to: {ATTENDANCE_FILE}")
print("👋 Done!")