import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_legpress(num_sets):
    st.subheader("Leg Press Counter 🦵")

    # --- Session State Flags ---
    if "legpress_stop" not in st.session_state:
        st.session_state.legpress_stop = False
    if "legpress_reset" not in st.session_state:
        st.session_state.legpress_reset = False

    # --- Control Buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🛑 Stop"):
            st.session_state.legpress_stop = True
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.legpress_reset = True
            st.experimental_rerun()

    FRAME_WINDOW = st.image([])
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    reps = 0
    sets_done = 0
    down = False
    up = False
    set_goal_reps = 10

    def get_knee_angle(keypoints):
        try:
            hip = keypoints[11]  # Left hip
            knee = keypoints[13] # Left knee
            ankle = keypoints[15]# Left ankle

            a = np.array(hip)
            b = np.array(knee)
            c = np.array(ankle)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except:
            return None

    while cap.isOpened() and not st.session_state.legpress_stop:
        ret, frame = cap.read()
        if not ret:
            st.warning('Camera not available.')
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, save=False, conf=0.5)
        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        knee_angle = get_knee_angle(keypoints)

        if knee_angle is None:
            continue

        feedback = ""
        if knee_angle < 70:
            down = True
            feedback = "Good Press ✅"

            if up:
                reps += 1
                up = False
        elif knee_angle > 160:
            if down:
                up = True
                down = False
            feedback = "Extend your legs fully! ❗"

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

    if st.session_state.legpress_stop:
        st.info("Exercise stopped by user 🚫")
