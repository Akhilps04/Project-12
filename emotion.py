import cv2
from deepface import DeepFace
import time
import numpy as np

class EmotionDetector:
    def __init__(self):
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.last_emotion = None
        self.emotion_buffer = []
        self.buffer_size = 5  # Number of frames to average emotion over

    def initialize_camera(self):
        # Try multiple camera indices
        for index in range(4):
            print(f"Trying camera index {index}")
            self.cap = cv2.VideoCapture(index)
            
            # Test if camera works by reading a frame
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully connected to camera {index}")
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return True
                self.cap.release()
        
        print("No working camera found")
        return False

    def get_smooth_emotion(self, new_emotion):
        # Add new emotion to buffer
        self.emotion_buffer.append(new_emotion)
        if len(self.emotion_buffer) > self.buffer_size:
            self.emotion_buffer.pop(0)
        
        # Return most common emotion in buffer
        if self.emotion_buffer:
            return max(set(self.emotion_buffer), key=self.emotion_buffer.count)
        return new_emotion

    def process_frame(self, frame):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            try:
                # Extract and preprocess face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Analyze emotion
                result = DeepFace.analyze(
                    face_roi, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True
                )
                
                # Get and smooth emotion
                emotion = result[0]['dominant_emotion']
                smooth_emotion = self.get_smooth_emotion(emotion)
                
                # Draw rectangle and emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, smooth_emotion, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue
        
        return frame

    def run(self):
        if not self.initialize_camera():
            return

        print("Starting emotion detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Failed to grab frame")
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the result
                cv2.imshow('Emotion Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Add small delay to reduce CPU usage
                time.sleep(0.01)

        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()