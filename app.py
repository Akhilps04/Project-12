from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
from deepface import DeepFace
import threading
import time
import numpy as np
import pygame  # For playing sound

app = Flask(__name__)

# Global variables
output_frame = None
lock = threading.Lock()
emotion_detected = "No emotion detected"
head_count = 0
dark_mode = False
alert_triggered = False
alert_active = False  # Flag to avoid repeated alerts
alert_reset_time = time.time()

# Initialize pygame for sound playback
pygame.mixer.init()
alert_sound_path = "C:/Users/HP/Desktop/lama/static/siren-alert-96052.mp3"  # Replace with the name of your alert sound file in the static folder

def detect_emotion_and_heads():
    global output_frame, lock, emotion_detected, head_count, alert_triggered, alert_active, alert_reset_time
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load pre-trained object detection model (YOLO or similar)
    net = cv2.dnn.readNetFromDarknet('C:/Users/HP/Desktop/lama/yolov4.cfg', 'C:/Users/HP/Desktop/lama/yolov4.weights')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    classes = ['knife', 'gun']  # Classes to detect
    
    # Start capturing video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        head_count = len(faces)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotion_detected = emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except:
                pass

        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)
        
        detected_objects = []
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)  # Index of the highest score
                confidence = scores[class_id]

                if confidence > 0.5 and class_id < len(classes):
                    label = classes[class_id]
                    detected_objects.append(label)

        # Check if alert conditions are met
        if ('knife' in detected_objects or 'gun' in detected_objects) and (time.time() - alert_reset_time > 60):
            alert_triggered = True
            alert_reset_time = time.time()
            play_alert_sound()
        elif head_count > 2 and (time.time() - alert_reset_time > 60):
            alert_triggered = True
            alert_reset_time = time.time()
            play_alert_sound()

        with lock:
            output_frame = frame.copy()

def play_alert_sound():
    """
    Play the alert sound for 20 seconds in a separate thread.
    """
    def sound_thread():
        pygame.mixer.music.load(alert_sound_path)
        pygame.mixer.music.play(-1)  # Loop indefinitely
        time.sleep(20)  # Play for 20 seconds
        pygame.mixer.music.stop()
    
    threading.Thread(target=sound_thread).start()

@app.route('/alert')
def alert():
    return jsonify({"alert": alert_active})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                   mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/get_emotion')
def get_emotion():
    return jsonify({"emotion": emotion_detected})

@app.route('/get_head_count')
def get_head_count():
    return jsonify({"head_count": head_count})

@app.route('/toggle_dark_mode', methods=['POST'])
def toggle_dark_mode():
    global dark_mode
    dark_mode = not dark_mode
    return jsonify({"dark_mode": dark_mode})

@app.route('/get_dark_mode')
def get_dark_mode():
    return jsonify({"dark_mode": dark_mode})

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

if __name__ == '__main__':
    print("Starting emotion detection...")
    t = threading.Thread(target=detect_emotion_and_heads)
    t.daemon = True
    t.start()
    
    print("Starting server...")
    app.run(debug=False, threaded=True, port=5000)
