
import cv2
import face_recognition

print("Starting Smart Attendance System...")

# Sample camera capture
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
if ret:
    print("Captured frame for face recognition")
video_capture.release()
