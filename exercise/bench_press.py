import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_bench_press(num_sets):
    st.subheader("Bench Press Counter 🏋️‍♀️ (Form Checker)")

    FRAME_WINDOW = st.image([])
    progress_bar = st.progress(0)

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    reps = 0
    sets_done = 0
    down = False
    up = False
    initial_shoulder_y = None
    set_goal_reps = 10  # reps per set

    while cap.isOpened():
        # --- If user pressed End Exercise (sidebar button) ---
        if st.session_state.get("end_exercise", False):
            st.info("Workout Stopped 🚫")
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("Camera error.")
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, save=False, conf=0.5)

        for result in results:
            keypoints = result.keypoints.xy[0].cpu().numpy()

            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

            if initial_shoulder_y is None:
                initial_shoulder_y = shoulder_y

            bar_depth = np.clip((shoulder_y - initial_shoulder_y) / 100, 0, 1)
            progress_bar.progress(bar_depth)

            feedback = ""

            if bar_depth > 0.5:
                down = True
                feedback = "Good Press ✅"

                if up:
                    reps += 1
                    up = False
            else:
                if down:
                    up = True
                    down = False
                feedback = "Bring bar lower ❗"

            annotated_frame = result.plot()
            cv2.putText(annotated_frame, f"Reps: {reps}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated_frame, feedback, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)

        # --- Set Complete ---
        if reps >= set_goal_reps:
            sets_done += 1
            st.success(f"Set {sets_done} Completed! 🎯")

            reps = 0
            time.sleep(5)

            if sets_done >= num_sets:
                st.balloons()
                st.success("Bench Press Workout Completed! ✅")
                break

    cap.release()
    FRAME_WINDOW.empty()
    progress_bar.empty()
