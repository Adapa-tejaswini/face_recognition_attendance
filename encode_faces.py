import face_recognition
import cv2
import os
import pickle
import numpy as np

# -------- CONFIGURATION --------
DATASET_DIR    = "dataset_face"
ENCODINGS_FILE = "encodings.pickle"

# -------- DELETE OLD PICKLE AUTOMATICALLY --------
if os.path.exists(ENCODINGS_FILE):
    os.remove(ENCODINGS_FILE)
    print(f"🗑️  Deleted old '{ENCODINGS_FILE}'")

known_encodings = []
known_names     = []

print("=" * 50)
print("       FACE ENCODING SYSTEM")
print("=" * 50)
print(f"\n📁 Dataset folder : {DATASET_DIR}")

# -------- CHECK FOLDER EXISTS --------
if not os.path.exists(DATASET_DIR):
    print(f"❌ Folder '{DATASET_DIR}' not found!")
    exit()

# -------- LIST ALL FILES FOUND --------
print("\n📂 Files found in dataset:")
print("-" * 50)
for person_name in sorted(os.listdir(DATASET_DIR)):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if os.path.isdir(person_folder):
        files = os.listdir(person_folder)
        print(f"  👤 {person_name}/")
        for f in files:
            print(f"      📸 {f}")
print("-" * 50)

# -------- PROCESS EACH PERSON --------
for person_name in sorted(os.listdir(DATASET_DIR)):
    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\n👤 Processing: {person_name}")

    image_files = [
        f for f in os.listdir(person_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"   ⚠️  No images found!")
        continue

    print(f"   📂 Found {len(image_files)} image(s)")
    success = 0
    failed  = 0

    for image_name in image_files:
        image_path = os.path.join(person_folder, image_name)
        print(f"\n   📸 Image: {image_name}")

        try:
            # -------- LOAD IMAGE --------
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if img_bgr is None:
                print(f"   ❌ Cannot read image!")
                failed += 1
                continue

            print(f"   📐 Original : {img_bgr.shape}")

            # -------- RESIZE IF TOO LARGE --------
            h, w = img_bgr.shape[:2]
            if max(h, w) > 800:
                scale   = 800 / max(h, w)
                img_bgr = cv2.resize(
                    img_bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA
                )
                print(f"   📏 Resized  : {img_bgr.shape}")

            # -------- CONVERT TO RGB --------
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = np.array(img_rgb, dtype=np.uint8, order='C')

            # -------- DETECT FACES --------
            face_locations = face_recognition.face_locations(
                img_rgb,
                number_of_times_to_upsample=2,
                model="hog"
            )
            print(f"   👁️  Faces    : {len(face_locations)}")

            if not face_locations:
                print(f"   ⚠️  No face detected!")
                failed += 1
                continue

            # -------- ENCODE FACES --------
            encodings = face_recognition.face_encodings(
                img_rgb,
                known_face_locations=face_locations,
                num_jitters=2
            )

            if not encodings:
                print(f"   ⚠️  Encoding failed!")
                failed += 1
                continue

            known_encodings.append(encodings[0])
            known_names.append(person_name)
            success += 1
            print(f"   ✅ Encoded successfully!")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1

    print(f"\n   📊 {person_name}: ✅{success} success | ❌{failed} failed")
    print("-" * 50)

# -------- RESULTS --------
print(f"\n{'=' * 50}")
print(f"  ENCODING RESULTS")
print(f"{'=' * 50}")
print(f"  Total encodings : {len(known_encodings)}")
print(f"  Total persons   : {len(set(known_names))}")
print(f"  Names           : {sorted(set(known_names))}")

if not known_encodings:
    print("\n❌ NO FACES ENCODED!")
    print("\n💡 Reasons & Solutions:")
    print("   1. Photo unclear  -> Use clear front face photo")
    print("   2. Face too small -> Face should fill 1/4 of image")
    print("   3. Bad lighting   -> Use well-lit photos")
    exit()

# -------- SAVE PICKLE --------
print(f"\n💾 Saving encodings...")

data = {
    "encodings" : known_encodings,
    "names"     : known_names
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

# -------- VERIFY --------
size   = os.path.getsize(ENCODINGS_FILE)
print(f"✅ Saved '{ENCODINGS_FILE}'")
print(f"📦 File size : {size} bytes")

with open(ENCODINGS_FILE, "rb") as f:
    verify = pickle.load(f)

print(f"✅ Verified  : {len(verify['encodings'])} encodings")
print(f"✅ Names     : {verify['names']}")
print(f"\n🎉 Done! Now run recognize_faces.py")