
# ASD Therapy Tracking System - Flexible Input

# Install required libraries
!pip install torch torchvision opencv-python mediapipe librosa scikit-learn fer youtube-dl

import os
import cv2
import torch
import numpy as np
import librosa
import mediapipe as mp
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from fer import FER
from sklearn.ensemble import RandomForestClassifier
import youtube_dl
from google.colab import drive
from IPython.display import HTML, display
from base64 import b64encode

# Mount Google Drive
drive.mount('/content/drive')

# Person Detector
class PersonDetector:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def detect(self, frame):
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            predictions = self.model(input_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        person_detections = [(box, score) for box, score, label in zip(boxes, scores, labels) if label == 1 and score > 0.7]

        return person_detections

# Pose Tracker
class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def track(self, frame, detections):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        poses = []
        if results.pose_landmarks:
            for detection in detections:
                box = detection[0]
                landmarks = results.pose_landmarks.landmark
                pose = [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
                poses.append((box, pose))

        return poses

# Audio Analyzer
class AudioAnalyzer:
    def __init__(self, audio_file, sr=22050):
        self.audio, self.sr = librosa.load(audio_file, sr=sr)
        self.frame_length = librosa.time_to_frames(1, sr=self.sr)

    def analyze(self, time):
        frame_start = int(time * self.sr)
        frame_end = frame_start + self.frame_length

        if frame_end > len(self.audio):
            return None

        audio_frame = self.audio[frame_start:frame_end]

        mfcc = librosa.feature.mfcc(y=audio_frame, sr=self.sr, n_mfcc=13)
        vad = librosa.effects.preemphasis(audio_frame)

        return {'mfcc': mfcc.mean(axis=1), 'vad': np.mean(vad)}

# Behavior Recognizer
class BehaviorRecognizer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def recognize(self, poses, audio_features):
        behaviors = []
        for pose in poses:
            pose_features = np.array([landmark[2] for landmark in pose[1]])
            features = np.concatenate([pose_features, audio_features['mfcc']])

            behavior = self.model.predict([features])[0]
            behaviors.append(behavior)

        return behaviors

# Emotion Recognizer
class EmotionRecognizer:
    def __init__(self):
        self.detector = FER(mtcnn=True)

    def recognize(self, frame, detections):
        emotions = []
        for detection in detections:
            box = detection[0]
            x, y, w, h = map(int, box)
            face = frame[y:y+h, x:x+w]

            emotion, score = self.detector.top_emotion(face)
            emotions.append((emotion, score))

        return emotions

# Visualizer
class Visualizer:
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),
            'pose': (255, 0, 0),
            'emotion': (0, 0, 255),
            'behavior': (255, 255, 0)
        }

    def draw(self, frame, detections, poses, behaviors, emotions):
        output = frame.copy()

        for detection, pose, behavior, emotion in zip(detections, poses, behaviors, emotions):
            box = detection[0]
            cv2.rectangle(output, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), self.colors['person'], 2)

            for point in pose[1]:
                cv2.circle(output, (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])), 3, self.colors['pose'], -1)

            label = f"Behavior: {behavior}, Emotion: {emotion[0]}"
            cv2.putText(output, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['behavior'], 2)

        return output

# Function to download YouTube video
def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'input_audio.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'input_video.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Main processing function
def process_video(video_path, audio_path):
    person_detector = PersonDetector()
    pose_tracker = PoseTracker()
    audio_analyzer = AudioAnalyzer(audio_path)
    behavior_recognizer = BehaviorRecognizer()
    emotion_recognizer = EmotionRecognizer()
    visualizer = Visualizer()

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = person_detector.detect(frame)
        poses = pose_tracker.track(frame, detections)
        audio_features = audio_analyzer.analyze(frame_count / fps)
        behaviors = behavior_recognizer.recognize(poses, audio_features)
        emotions = emotion_recognizer.recognize(frame, detections)

        output_frame = visualizer.draw(frame, detections, poses, behaviors, emotions)

        out.write(output_frame)
        frame_count += 1

    cap.release()
    out.release()

# Function to display video in notebook
def display_video(video_path):
    mp4 = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f"""
    <video width=400 controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """))

def main():
    print("ASD Therapy Tracking System")
    print("1. Process YouTube videos")
    choice = input("Enter your choice (1): ")

    if choice == '1':
        urls_input = input('https://www.youtube.com/watch?v=V9YDDpo9LWg',
'https://www.youtube.com/watch?v=JBoc3w5EKfI',
'https://www.youtube.com/watch?v=aWV7UUMddCU',
'https://www.youtube.com/watch?v=f6wqlpG9rd0',
'https://www.youtube.com/watch?v=GNVTuLHdeSo',
'https://www.youtube.com/watch?v=SWtmkjd45so',
'https://www.youtube.com/watch?v=RzI6Ar5mu2Q',
'https://www.youtube.com/watch?v=aulLej6Z6W8',
'https://www.youtube.com/watch?v=7pN6ydLE4EQ',
'https://www.youtube.com/watch?v=fEEelCgBkWA',
'https://www.youtube.com/watch?v=ckZQbQwM3oU',
'https://www.youtube.com/watch?v=E8Wgwg3F4X0',
'https://www.youtube.com/watch?v=rvIPH4ccfpI',
'https://www.youtube.com/watch?v=F6iqlW6ovZc',
'https://www.youtube.com/watch?v=9qjk-Sq415s&list=PL5B0D2D5B4BFE92C1&index=6',
'https://www.youtube.com/watch?v=DI25kGJis0w',
'https://www.youtube.com/watch?v=rrLhFZG6iQY',
'https://www.youtube.com/watch?v=RKOZbT0ftL4&t=1s',
'https://www.youtube.com/watch?v=N7TBbWHB01E',
'https://www.youtube.com/watch?v=1YqVEVbXQ1c')
        urls = [url.strip() for url in urls_input.split(',')]

        for i, url in enumerate(urls):
            print(f"Processing video {i+1}/{len(urls)}")
            download_youtube_video(url)
            process_video('input_video.mp4', 'input_audio.wav')
            os.rename('output.mp4', f'output_{i+1}.mp4')
            display_video(f'output_{i+1}.mp4')

        print("All videos processed.")
    else:
        print("Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    main()


