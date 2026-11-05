# Smart Attendance System

A Python + Streamlit prototype that registers students using a webcam, recognises registered faces with OpenCV LBPH, marks attendance once per day, and exports attendance records as CSV.

## Features
- Register students with name, ID, and webcam photo
- Detect and match faces using OpenCV
- Mark attendance with date and time
- Prevent duplicate attendance on the same day
- View and download attendance as CSV

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push these files to a public GitHub repository, then deploy the repository on Streamlit Community Cloud.

**Important:** This is an educational prototype. It uses basic face matching and local file storage; it is not suitable for high-security attendance. Streamlit Cloud's local files may reset or be ephemeral, so use a database and proper consent/privacy controls for real deployment.
