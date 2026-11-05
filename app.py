import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

st.set_page_config(page_title="Smart Attendance System", page_icon="📸", layout="centered")

DATA = Path("data")
FACES = DATA / "faces"
ATTENDANCE = DATA / "attendance.csv"
LABELS = DATA / "labels.json"
FACES.mkdir(parents=True, exist_ok=True)

st.title("📸 Smart Attendance System")
st.caption("A simple face-recognition attendance prototype")

def detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def get_face(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
    if len(faces) == 0:
        return image, None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return image, face

def load_model():
    names = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    samples, ids = [], []
    for sid, name in names.items():
        folder = FACES / sid
        if folder.exists():
            for path in folder.glob("*.png"):
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    samples.append(cv2.resize(img, (200, 200)))
                    ids.append(int(sid))
    if not samples:
        return None, names
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(samples, np.array(ids))
    return model, names

tab1, tab2, tab3 = st.tabs(["Register Student", "Mark Attendance", "Attendance Records"])

with tab1:
    st.subheader("Register a student")
    name = st.text_input("Student name")
    student_id = st.text_input("Student ID")
    photo = st.camera_input("Capture student's face")
    if st.button("Register Student", type="primary"):
        if not name.strip() or not student_id.strip() or photo is None:
            st.error("Enter the student's name and ID, then capture a photo.")
        elif not student_id.strip().isalnum():
            st.error("Use letters and numbers only for the Student ID.")
        else:
            _, face = get_face(photo.getvalue())
            if face is None:
                st.error("No face detected. Face the camera and try again.")
            else:
                sid = student_id.strip().upper()
                folder = FACES / sid
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.png"), face)
                names = json.loads(LABELS.read_text()) if LABELS.exists() else {}
                names[sid] = name.strip()
                LABELS.write_text(json.dumps(names, indent=2))
                st.success(f"Registered {name.strip()} ({sid}) successfully!")

with tab2:
    st.subheader("Mark attendance")
    shot = st.camera_input("Capture face to mark attendance", key="attendance_camera")
    if st.button("Recognise & Mark Attendance", type="primary"):
        if shot is None:
            st.error("Capture a face first.")
        else:
            model, names = load_model()
            _, face = get_face(shot.getvalue())
            if face is None:
                st.error("No face detected. Please try again.")
            elif model is None:
                st.warning("No students registered yet. Register a student first.")
            else:
                label, confidence = model.predict(face)
                # LBPH confidence is a distance; lower is a closer match.
                if label in [int(x) for x in names] and confidence < 75:
                    sid = str(label)
                    now = datetime.now()
                    row = {"Student ID": sid, "Name": names[sid],
                           "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}
                    df = pd.read_csv(ATTENDANCE) if ATTENDANCE.exists() else pd.DataFrame(columns=row.keys())
                    already = ((df["Student ID"] == sid) & (df["Date"] == row["Date"])).any() if not df.empty else False
                    if already:
                        st.info(f"Attendance already marked today for {names[sid]}.")
                    else:
                        pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(ATTENDANCE, index=False)
                        st.success(f"Attendance marked for {names[sid]}!")
                else:
                    st.error("Face not recognised. Register the student with a clear photo first.")

with tab3:
    st.subheader("Attendance history")
    if ATTENDANCE.exists():
        df = pd.read_csv(ATTENDANCE)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download attendance CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
    else:
        st.info("No attendance records yet.")

st.divider()
st.caption("Prototype note: LBPH face matching is a basic demo, not a secure identity-verification system. Use consent and test carefully.")
