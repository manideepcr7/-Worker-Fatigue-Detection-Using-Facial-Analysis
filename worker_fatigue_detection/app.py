
import cv2
import numpy as np
import mediapipe as mp
import gradio as gr

mp_face_mesh = mp.solutions.face_mesh

def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_fatigue(video):
    cap = cv2.VideoCapture(video)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    fatigue_events = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                h, w, _ = frame.shape

                left_eye = np.array([
                    [landmarks[362].x * w, landmarks[362].y * h],
                    [landmarks[385].x * w, landmarks[385].y * h],
                    [landmarks[387].x * w, landmarks[387].y * h],
                    [landmarks[263].x * w, landmarks[263].y * h],
                    [landmarks[373].x * w, landmarks[373].y * h],
                    [landmarks[380].x * w, landmarks[380].y * h],
                ])

                right_eye = np.array([
                    [landmarks[33].x * w, landmarks[33].y * h],
                    [landmarks[160].x * w, landmarks[160].y * h],
                    [landmarks[158].x * w, landmarks[158].y * h],
                    [landmarks[133].x * w, landmarks[133].y * h],
                    [landmarks[153].x * w, landmarks[153].y * h],
                    [landmarks[144].x * w, landmarks[144].y * h],
                ])

                ear_left = calculate_EAR(left_eye)
                ear_right = calculate_EAR(right_eye)
                ear_avg = (ear_left + ear_right) / 2.0

                if ear_avg < 0.21:
                    fatigue_events += 1
        if frame_count >= 100:
            break
    cap.release()

    if fatigue_events > 15:
        return "Fatigue Detected!"
    else:
        return "No Fatigue Detected."

iface = gr.Interface(fn=detect_fatigue,
                     inputs=gr.Video(type="mp4"),
                     outputs="text",
                     title="Worker Fatigue Detection",
                     description="Analyzes uploaded video to detect signs of eye fatigue using facial landmark tracking.")

iface.launch()
