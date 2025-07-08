import cv2
import numpy as np
from keras.models import load_model
import pygame
import time
import os

# Load emotion detection model
model = load_model("emotion_model.h5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Map emotion to music file
music_map = {
    "Happy": "music/happy.mp3",
    "Sad": "music/sad.mp3",
    "Angry": "music/angry.mp3"
}

# Initialize pygame mixer
pygame.mixer.init()
current_music = None

def play_music(emotion):
    global current_music
    if emotion in music_map:
        music_file = music_map[emotion]
        if current_music != music_file:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play(-1)  # loop
            current_music = music_file
            print(f"🎶 Now playing: {emotion} music")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("🟢 Timify is running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        # Show emotion on screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 50), 2)
        cv2.putText(frame, f"Mood: {emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Play music based on current emotion
        play_music(emotion)

    cv2.imshow("Timify - Mood Music AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.mixer.quit()
