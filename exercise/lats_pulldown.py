import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_lats_pulldown(num_sets):
    st.subheader("Lats Pulldown Counter 🏋️‍♀️")

    # --- Session State ---
    if "lats_stop" not in st.session_state:
        st.session_state.lats_stop = False
    if "lats_reset" not in st.session_state:
        st.session_state.lats_reset = False

    # --- Buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🛑 Stop"):
            st.session_state.lats_stop = True
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.lats_reset = True
            st.experimental_rerun()

    FRAME_WINDOW = st.image([])
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    reps = 0
    sets_done = 0
    pulling = False
    set_goal_reps = 10

    def get_shoulder_movement(keypoints):
        try:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
            return avg_y
        except:
            return None

    while cap.isOpened() and not st.session_state.lats_stop:
        ret, frame = cap.read()
        if not ret:
            st.warning('Camera not available.')
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, save=False, conf=0.5)

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        shoulder_y = get_shoulder_movement(keypoints)

        if shoulder_y is None:
            continue

        feedback = ""
        if shoulder_y > 300:
            pulling = True
            feedback = "Good Pull! ✅"
        elif shoulder_y < 200:
            if pulling:
                reps += 1
                pulling = False
            feedback = "Pull Down Properly! ❗"

        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f'Reps: {reps}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated_frame, feedback, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)

        if reps >= set_goal_reps:
            sets_done += 1
            st.success(f"Set {sets_done} Completed! 🎉")
            reps = 0
            time.sleep(5)

            if sets_done >= num_sets:
                st.balloons()
                st.success("Workout Finished! 🎯")
                cap.release()
                break

    cap.release()
    FRAME_WINDOW.empty()

    if st.session_state.lats_stop:
        st.info("Exercise stopped by user 🚫")
