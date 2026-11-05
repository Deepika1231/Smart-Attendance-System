# Smart Attendance System

🚀 **Live Demo:** https://smart-attendance-system-hbcntfy5bnppugjzzw7ax5.streamlit.app/

A Python-based face recognition attendance system using OpenCV and SQLite.

## Features
- Student registration
- Face detection and recognition
- Automatic attendance marking
- SQLite database for student records
- Duplicate attendance prevention
- Attendance history and CSV download

## Technologies Used
- Python
- Streamlit
- OpenCV
- SQLite
- Pandas

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
