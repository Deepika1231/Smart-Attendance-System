import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Smart Attendance System", page_icon="📸", layout="wide")
st.title("📸 Smart Attendance System")
st.caption("Face registration, recognition, and attendance tracking with a local SQLite database.")

DB_PATH = Path(__file__).parent / "attendance.db"
CASCADE_PATH = Path(__file__).parent / "haarcascade_frontalface_default.xml"

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        face_image BLOB NOT NULL,
        registered_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        name TEXT NOT NULL,
        attendance_date TEXT NOT NULL,
        attendance_time TEXT NOT NULL,
        UNIQUE(student_id, attendance_date),
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )""")
    conn.commit()
    return conn

def face_detector():
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError("Face detector file is missing. Keep haarcascade_frontalface_default.xml beside app.py.")
    return cascade

def extract_face(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return None, "Could not read the camera image."
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None, "No face detected. Face the camera and try again."
    if len(faces) > 1:
        return None, "More than one face detected. Please make sure only one person is in the frame."
    x, y, w, h = faces[0]
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return face, None

def registered_faces(conn):
    rows = conn.execute("SELECT student_id, name, face_image FROM students").fetchall()
    faces, labels, names = [], [], {}
    for sid, name, blob in rows:
        img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(cv2.resize(img, (200, 200)))
            labels.append(len(names))
            names[len(names)] = (sid, name)
    return faces, np.array(labels, dtype=np.int32), names

def build_recognizer(conn):
    if not hasattr(cv2, "face"):
        raise RuntimeError("OpenCV face module is unavailable. Check opencv-contrib-python-headless in requirements.txt.")
    faces, labels, names = registered_faces(conn)
    if len(faces) == 0:
        return None, names
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    return recognizer, names

conn = db()
register_tab, mark_tab, students_tab, records_tab = st.tabs([
    "Register Student", "Mark Attendance", "Registered Students", "Attendance Records"
])

with register_tab:
    st.subheader("Register a student")
    st.info("Ask the student for permission before capturing and storing their face image.")
    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input("Student name")
        student_id = st.text_input("Student ID").strip().upper()
    with c2:
        registration_photo = st.camera_input("Capture one clear face photo", key="register_camera")
    if st.button("Save Student", type="primary"):
        if not student_name.strip() or not student_id or registration_photo is None:
            st.error("Enter the name and ID, and capture a face photo.")
        elif not student_id.replace("-", "").replace("_", "").isalnum():
            st.error("Student ID can contain letters, numbers, hyphens, and underscores only.")
        else:
            face, error = extract_face(registration_photo.getvalue())
            if error:
                st.error(error)
            else:
                ok, encoded = cv2.imencode(".png", face)
                try:
                    conn.execute(
                        "INSERT INTO students(student_id, name, face_image, registered_at) VALUES (?, ?, ?, ?)",
                        (student_id, student_name.strip(), encoded.tobytes(), datetime.now().isoformat(timespec="seconds"))
                    )
                    conn.commit()
                    st.success(f"{student_name.strip()} ({student_id}) registered successfully!")
                except sqlite3.IntegrityError:
                    st.error("That Student ID is already registered. Use a unique ID.")

with mark_tab:
    st.subheader("Recognise face and mark attendance")
    attendance_photo = st.camera_input("Capture face for attendance", key="attendance_camera")
    if st.button("Recognise & Mark Attendance", type="primary"):
        if attendance_photo is None:
            st.error("Capture a face first.")
        else:
            try:
                face, error = extract_face(attendance_photo.getvalue())
                if error:
                    st.error(error)
                else:
                    recognizer, label_names = build_recognizer(conn)
                    if recognizer is None:
                        st.warning("No students are registered yet. Register a student first.")
                    else:
                        predicted_label, confidence = recognizer.predict(face)
                        # LBPH returns a distance; smaller means a closer match.
                        if predicted_label in label_names and confidence < 65:
                            sid, name = label_names[predicted_label]
                            now = datetime.now()
                            try:
                                conn.execute(
                                    "INSERT INTO attendance(student_id, name, attendance_date, attendance_time) VALUES (?, ?, ?, ?)",
                                    (sid, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                                )
                                conn.commit()
                                st.success(f"Face recognised: {name} ({sid}). Attendance marked!")
                            except sqlite3.IntegrityError:
                                st.info(f"{name}'s attendance is already marked today.")
                            st.caption(f"Face-match distance: {confidence:.1f} (lower is a closer match)")
                        else:
                            st.error("Face not recognised. Only registered faces can mark attendance.")
            except Exception as exc:
                st.error(f"Recognition error: {exc}")

with students_tab:
    st.subheader("Registered students database")
    student_rows = conn.execute(
        "SELECT student_id AS 'Student ID', name AS 'Name', registered_at AS 'Registered At' FROM students ORDER BY name"
    ).fetchall()
    if student_rows:
        st.dataframe(pd.DataFrame(student_rows, columns=["Student ID", "Name", "Registered At"]), use_container_width=True)
        st.caption(f"Total registered students: {len(student_rows)}")
        st.warning("Deleting a student also removes their attendance records.")
        remove_id = st.selectbox("Select student to remove", [r[0] for r in student_rows], format_func=lambda x: f"{x} — " + next(r[1] for r in student_rows if r[0] == x))
        if st.button("Delete selected student"):
            conn.execute("DELETE FROM attendance WHERE student_id = ?", (remove_id,))
            conn.execute("DELETE FROM students WHERE student_id = ?", (remove_id,))
            conn.commit()
            st.success("Student and related attendance records deleted.")
            st.rerun()
    else:
        st.info("No students registered yet.")

with records_tab:
    st.subheader("Attendance records")
    rows = conn.execute(
        "SELECT student_id AS 'Student ID', name AS 'Name', attendance_date AS 'Date', attendance_time AS 'Time' FROM attendance ORDER BY attendance_date DESC, attendance_time DESC"
    ).fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=["Student ID", "Name", "Date", "Time"])
        st.dataframe(df, use_container_width=True)
        st.download_button("Download attendance CSV", df.to_csv(index=False), "attendance_records.csv", "text/csv")
    else:
        st.info("No attendance records yet.")

conn.close()
st.divider()
st.caption(
    "Educational prototype only. Face recognition can make mistakes and is not secure identity verification. "
    "Obtain consent, protect stored biometric data, and test with multiple lighting conditions. "
    "The SQLite file on Streamlit Community Cloud may be reset when the app is redeployed or its instance restarts."
)
