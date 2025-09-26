import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_squats(num_sets):
    st.subheader("Squats Counter 🏋️‍♂️ (Form Checker)")

    # --- Session State Flags ---
    if "squats_stop" not in st.session_state:
        st.session_state.squats_stop = False
    if "squats_reset" not in st.session_state:
        st.session_state.squats_reset = False

    # --- Control Buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🛑 Stop"):
            st.session_state.squats_stop = True
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.squats_reset = True
            st.experimental_rerun()

    FRAME_WINDOW = st.image([])
    progress_bar = st.progress(0)

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    reps = 0
    sets_done = 0
    down = False
    up = False
    initial_hip_y = None
    set_goal_reps = 10

    while sets_done < num_sets and not st.session_state.squats_stop:
        ret, frame = cap.read()
        if not ret:
            st.warning('Camera error.')
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, save=False, conf=0.5)

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy[0]) > 12:
                keypoints = result.keypoints.xy[0].cpu().numpy()

                left_hip = keypoints[11]
                right_hip = keypoints[12]
                hip_y = (left_hip[1] + right_hip[1]) / 2

                if initial_hip_y is None:
                    initial_hip_y = hip_y

                squat_depth = (initial_hip_y - hip_y) / 100  # difference inverted
                squat_depth = np.clip(squat_depth, 0, 1)  # keep between 0 and 1
                progress_bar.progress(squat_depth)

                feedback = ""

                if squat_depth > 0.3:  # lowered threshold to 0.3 for squats
                    down = True
                    feedback = "Good Squat ✅"

                    if up:
                        reps += 1
                        up = False
                else:
                    if down:
                        up = True
                        down = False
                    feedback = "Go Lower ❗"

                annotated_frame = result.plot()

                cv2.putText(annotated_frame, f'Reps: {reps}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(annotated_frame, feedback, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                annotated_frame = frame.copy()

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)

        if st.session_state.squats_stop:
            st.warning("Exercise stopped by user 🚫")
            cap.release()
            return

        if reps >= set_goal_reps:
            sets_done += 1
            st.success(f"Set {sets_done} Completed! 🔥")

            reps = 0
            time.sleep(5)

            if sets_done >= num_sets:
                st.balloons()
                st.success("Workout Completed! ✅")
                cap.release()
                break

    cap.release()
