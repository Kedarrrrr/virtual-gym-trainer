import streamlit as st
from exercise import pushups, squats, bicep_curls, bench_press

# --- Title ---
st.title("🏋️‍♂️ AI Fitness Trainer - Repetition Counter & Form Checker")

# --- Sidebar Navigation ---
exercise_list = ["Push-Ups", "Squats", "Bicep Curls", "Bench Press"]
choice = st.sidebar.selectbox("Select Exercise", exercise_list)

# --- Ask for Sets ---
num_sets = st.sidebar.number_input('Number of Sets:', min_value=1, max_value=10, step=1)

start = st.button("Start Workout 🚀")

# --- Routing Based on Choice ---
if start:
    if choice == "Push-Ups":
        pushups.run_pushup(num_sets)
    elif choice == "Squats":
        squats.run_squats(num_sets)
    elif choice == "Bicep Curls":
        bicep_curls.run_bicep_curls(num_sets)
    elif choice == "Bench Press":
        bench_press.run_bench_press(num_sets)
