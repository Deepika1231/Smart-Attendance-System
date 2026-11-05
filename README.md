# Smart Attendance System

A Python and Streamlit face-recognition attendance prototype using OpenCV LBPH and a lightweight SQLite database.

## Features
- Register a student with a unique ID, name, and webcam face image
- Store student details and face image in SQLite (`attendance.db`)
- Compare a captured face with registered student faces
- Mark attendance with date and time
- Prevent duplicate attendance for the same student on the same day
- View registered students and delete a student
- View and download attendance records as CSV

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `haarcascade_frontalface_default.xml` in the same folder as `app.py`.

## Database
The app creates `attendance.db` automatically. It contains:
- `students`: student ID, name, face image, registration timestamp
- `attendance`: student ID, name, date, time

## Important limitations
This is an educational prototype, not a secure or clinically/legally validated identity system. Face recognition can make mistakes. Obtain informed consent, restrict access to biometric data, and use secure storage and a tested liveness-detection process before any real deployment.

Streamlit Community Cloud uses an ephemeral local filesystem; the SQLite database may be reset when the app restarts or is redeployed. For reliable hosted persistence, connect a managed database such as PostgreSQL or another durable database service.
