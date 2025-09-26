import streamlit as st
import time
import cv2
from ultralytics import YOLO

def run_plank(num_sets):
    st.subheader("Plank Timer 🧘‍♂️ (Hold your position!)")

    # --- Session State Flags ---
    if "plank_stop" not in st.session_state:
        st.session_state.plank_stop = False
    if "plank_reset" not in st.session_state:
        st.session_state.plank_reset = False

    # --- Control Buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🛑 Stop"):
            st.session_state.plank_stop = True
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.plank_reset = True
            st.experimental_rerun()

    FRAME_WINDOW = st.image([])
    duration = st.slider('Plank Duration (seconds)', min_value=10, max_value=300, step=10, value=60)

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    sets_done = 0

    while sets_done < num_sets and not st.session_state.plank_stop:
        ret, frame = cap.read()
        if not ret:
            st.warning('Camera not available.')
            break
        
        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)

        st.info(f"Hold the Plank Position for {duration} seconds!")
        progress = st.progress(0)

        for i in range(duration):
            if st.session_state.plank_stop:
                st.info("Exercise stopped by user 🚫")
                cap.release()
                return
            time.sleep(1)
            progress.progress((i + 1) / duration)

        sets_done += 1
        st.success(f"Set {sets_done} Completed! 🎉")
        time.sleep(3)

    cap.release()
    FRAME_WINDOW.empty()

    if sets_done >= num_sets and not st.session_state.plank_stop:
        st.balloons()
        st.success("Plank Workout Finished! 🎯")
