import streamlit as st
import os
from exercise.push_ups import run_pushup
from exercise.bench_press import run_bench_press
from exercise.bicep_curls import run_bicep_curls
# import other exercises similarly when ready

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Virtual Gym Trainer", layout="wide")

# ========== CUSTOM BACKGROUND ==========
page_bg_file = r"C:\Users\kedar\Desktop\projects\sound_integration\background\gym_bg.jpg"
sidebar_bg_file = r"C:\Users\kedar\Desktop\projects\sound_integration\background\sidebar_bg.jpg"

# Prepare paths for CSS
page_bg_css = page_bg_file.replace("\\", "/")
sidebar_bg_css = sidebar_bg_file.replace("\\", "/")

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("file:///{page_bg_css}");
        background-size: cover;
    }}
    [data-testid="stSidebar"] {{
        background-image: url("file:///{sidebar_bg_css}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ========== SESSION STATE ==========
if "end_exercise" not in st.session_state:
    st.session_state["end_exercise"] = False

# ========== SIDEBAR ==========
st.sidebar.title("🏋️ Virtual Trainer")
st.sidebar.markdown("Select an exercise:")

exercises = [
    "Push Ups",
    "Squats",
    "Bicep Curls",
    "Bench Press",
    "Lats Pull Down",
    "Leg Press",
    "Plank",
    "Crunches"
]

choice = st.sidebar.radio("Exercises", exercises)

# Number of sets input
num_sets = st.sidebar.number_input("Number of Sets", min_value=1, max_value=20, value=3)

# Start / End buttons
start_btn = st.sidebar.button("▶️ Start Exercise")
stop_btn = st.sidebar.button("⏹ End Exercise")

# Handle End Exercise
if stop_btn:
    st.session_state["end_exercise"] = True

# ========== MUSIC PLAYER ==========
st.sidebar.subheader("🎵 Music Player")
music_folder = r"C:\Users\kedar\Desktop\projects\sound_integration\music"
music_files = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]

if music_files:
    selected_song = st.sidebar.selectbox("Choose a track", music_files)
    music_path = os.path.abspath(os.path.join(music_folder, selected_song))
    st.sidebar.audio(music_path, format="audio/mp3")

# ========== MAIN AREA ==========
st.title("💪 Virtual Gym Trainer")
st.write(f"### Selected Exercise: **{choice}**")

# Run exercise when Start is clicked
if start_btn:
    st.session_state["end_exercise"] = False  # Reset stop flag

    if choice == "Push Ups":
        run_pushup(num_sets)
    elif choice == "Bench Press":
        run_bench_press(num_sets)
    elif choice == "Bicep Curls":
        run_bicep_curls(num_sets)
    elif choice == "Squats":
        st.info("Squats code coming soon...")
    elif choice == "Lats Pull Down":
        st.info("Lats Pull Down code coming soon...")
    elif choice == "Leg Press":
        st.info("Leg Press code coming soon...")
    elif choice == "Plank":
        st.info("Plank code coming soon...")
    elif choice == "Crunches":
        st.info("Crunches code coming soon...")
