# ⚡ Real-Time Weapon & Emotion Detection System
"AI doesn’t sleep, and now, neither does your security system."

A smart, reactive, and emotion-aware surveillance system powered by YOLOv4 + DeepFace. This project mixes computer vision, emotion intelligence, and real-time alerts—all wrapped in a Flask web interface. Think of it as a mini-Jarvis with sirens.

## 🚀 What’s This Project All About?
This is not just another camera feed.
This Flask-powered app does 3 things simultaneously:
- 🎯 Detects weapons (like knives) using YOLOv4
- 🧠 Analyzes facial expressions with DeepFace
- 🧍‍♂️ Counts the number of people via Haar Cascades
...and when things look sus—💥 it plays a real siren (because why not?).

## 🛠️ Core Features
Feature	Description
- 🔪 Weapon Detection  	  Identifies dangerous objects (knife/gun) in real-time
- 🙂 Emotion Scanner  	  Labels emotions like happy, angry, neutral from face snapshots
- 🧍‍♀️🧍 Head Counter	    Uses face detection to monitor how crowded the scene is
- 🚨 Auto Siren Trigger  	Plays a siren when threats or unusual crowding is detected
- 🌙 Dark Mode Toggle	    Built-in toggle to switch between light and dark themes
- 🔄 Live Feed API	      Stream real-time video with overlays + REST endpoints for frontend integration

## 🧠 Tech Stack Behind the Scenes
- YOLOv4 – Object detection for spotting weapons
- DeepFace – Emotion recognition (7-class prediction)
- OpenCV – All the camera handling & preprocessing
- Flask – Web server with live video streaming
- Pygame – For sound alerts (20-sec siren with cooldown)
- Threading – Keeps everything smooth and responsive

## 🎬 How it Works (in 1 Minute)
- Start the app: it spins up your webcam.
- Faces = detected. Emotions = scanned. Suspicious object? Detected.
- Crowd gets too big? Siren.
- Knife appears? Siren.
- You chill. It watches.

## Files You’ll See Inside
- app.py – The main brain 🧠
- templates/index.html – Live stream UI
- static/ – Your siren sound goes here
- yolov4.cfg + yolov4.weights – The YOLO setup
- haarcascade_frontalface_default.xml – Face detector

## 🚀 Future Enhancements
✅ Add mask detection  
✅ Push alerts to phone (Twilio integration?)  
✅ Integrate face recognition for known threats  
✅ Record suspicious clips


## 🤝 About This Project
- Author: Boss
- College Project? Yes. Boring? Nope.
- Built With: Sleep deprivation, caffeine, and pure AI curiosity.

