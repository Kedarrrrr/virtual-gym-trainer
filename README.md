Virtual Gym Trainer
Overview

Virtual Gym Trainer is a computer vision–based fitness application designed to help users perform exercises with correct posture and form. The primary goal of this project is injury prevention by detecting incorrect movement patterns in real time.

The system eliminates the need for manual repetition counting and constant self-monitoring. Instead, it automatically counts repetitions only when correct form is detected and provides real-time visual and audio feedback, allowing users to focus entirely on execution quality.

This project is focused on form validation, not just pose detection.

Problem Statement

A large number of workout-related injuries occur due to:

Incorrect posture

Poor joint alignment

Fatigue-induced form breakdown

Distraction caused by manual repetition counting

Virtual Gym Trainer addresses these issues by acting as a real-time virtual coach that continuously evaluates body mechanics during exercise execution.

Key Features

Real-time human pose estimation

Detection and tracking of body keypoints (head, shoulders, elbows, hips, knees, ankles)

Automatic repetition counting based on validated movement cycles

Mathematical posture validation using joint angles and spatial relationships

Real-time visual feedback on posture

Audio-based feedback to alert the user about correct or incorrect form

Live data collection through webcam input

Technologies Used

Programming Language: Python

Pose Estimation Model: YOLOv8 (Pose Estimation)

Computer Vision: OpenCV

Audio Feedback: Text-to-speech / audio output module

Logic Layer: Mathematical computation of angles, distances, and movement direction

Dataset: Real-time data captured from webcam (no pre-trained exercise dataset)

System Workflow

Webcam captures live video frames

YOLOv8 Pose model detects human body keypoints

OpenCV processes frames and renders pose landmarks

Mathematical logic calculates:

Joint angles

Relative distances between keypoints

Motion direction across frames

Exercise-specific rules validate correct posture

Repetitions are counted only when valid movement patterns are detected

Audio feedback provides instant cues for posture correction or confirmation

Audio Feedback System

The application includes an audio feedback mechanism that:

Alerts the user when posture deviates from defined thresholds

Confirms correct repetitions

Reduces dependency on screen monitoring during workouts

This improves usability and makes the system practical for real-world exercise scenarios.

Why YOLOv8 Pose

High inference speed suitable for real-time applications

Accurate keypoint detection

Efficient performance on consumer-grade hardware

Simple integration with OpenCV pipelines

Limitations

Performance depends on lighting conditions and camera placement

Designed for single-person exercise tracking

Exercise logic is rule-based, not learned from large datasets

Not intended for medical diagnosis or rehabilitation use

Future Scope

Support for multiple exercises

Voice-guided workout sessions

ML-based form scoring instead of fixed rule thresholds

Mobile and edge-device deployment

Exercise history tracking and analytics

Installation
pip install ultralytics opencv-python numpy

Usage
python main.py


Ensure that a functional webcam and audio output device are available.

Use Cases

Home workouts without professional supervision

Beginners learning correct exercise form

Injury prevention during unsupervised training

Form-focused strength and fitness routines

Author

Kedar Gaikwad