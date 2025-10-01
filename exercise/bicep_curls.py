import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from playsound import playsound

# --- Sound paths ---
correct_sound = "sounds/good.wav"
up_sound = "sounds/up.wav"
hold_sound = "sounds/hold.wav"
down_sound = "sounds/move.wav"
success_sound = "sounds/success.wav"

# --- Play sound in a separate thread ---
def play_sound(sound_path):
    threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

# --- "hold" then "down" sequence ---
def play_hold_then_down():
    play_sound(hold_sound)
    time.sleep(2)
    play_sound(down_sound)

# --- Bicep Curls Function ---
def run_bicep_curls(num_sets):
    st.subheader("Bicep Curls Counter 💪 (Form Checker)")

    FRAME_WINDOW = st.image([])
    progress_bar = st.progress(0)

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    reps = 0
    sets_done = 0
    down = False
    up = False
    set_goal_reps = 10  # reps per set

    last_warning_time = 0

    def get_elbow_angle(keypoints, side='left'):
        try:
            if side == 'left':
                shoulder, elbow, wrist = keypoints[5], keypoints[7], keypoints[9]
            else:
                shoulder, elbow, wrist = keypoints[6], keypoints[8], keypoints[10]

            a = np.array(shoulder)
            b = np.array(elbow)
            c = np.array(wrist)

            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return None

    while cap.isOpened():
        # --- Check if user pressed End Exercise (sidebar) ---
        if st.session_state.get("end_exercise", False):
            st.info("Exercise stopped 🚫")
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not available.")
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, save=False, conf=0.5)

        for result in results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy[0].cpu().numpy()
            left_angle = get_elbow_angle(keypoints, 'left')
            right_angle = get_elbow_angle(keypoints, 'right')

            if left_angle is None or right_angle is None:
                continue

            avg_angle = (left_angle + right_angle) / 2.0
            progress = np.clip((avg_angle - 30) / (150 - 30), 0, 1)
            progress_bar.progress(float(1 - progress) if 0 <= progress <= 1 else 0.0)

            feedback = ""

            if avg_angle < 50:
                down = True
                feedback = "Good Curl ✅"
                if up:
                    reps += 1
                    up = False
                    play_sound(correct_sound)
                    threading.Thread(target=play_hold_then_down, daemon=True).start()

            elif avg_angle > 140:
                if down:
                    up = True
                    down = False
                feedback = "Extend your arms fully! ❗"
                current_time = time.time()
                if current_time - last_warning_time > 4:
                    threading.Thread(target=play_hold_then_down, daemon=True).start()
                    last_warning_time = current_time

            annotated_frame = result.plot()
            cv2.putText(annotated_frame, f"Reps: {reps}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated_frame, feedback, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame)

        if reps >= set_goal_reps:
            sets_done += 1
            st.success(f"Set {sets_done} Completed! 🎉")
            play_sound(success_sound)
            reps = 0
            time.sleep(5)

            if sets_done >= num_sets:
                st.balloons()
                st.success("Bicep Curls Workout Completed! 🎯")
                break

    cap.release()
    FRAME_WINDOW.empty()
    progress_bar.empty()
