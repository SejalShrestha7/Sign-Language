import os
import cv2
import csv
import copy
import itertools
import string
import numpy as np
import nltk

from collections import deque
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import mediapipe as mp
from tensorflow.keras.models import load_model
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load keras model
model = load_model(os.path.join(os.getcwd(), "model", "GASLmodel.keras"))

# MediaPipe hands instance
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

gesture_text = ""

# Views

def index(request):
    return render(request, 'index.html')

def find_video(word):
    path = os.path.join("static", "assets", "ASL", f"{word}.mp4")
    return os.path.isfile(path)

def analyze_text(sentence):
    words = word_tokenize(sentence.lower())
    tagged = nltk.pos_tag(words)
    stop_words = ['@', '#', "http", ":", "is", "the", "are", "am", "a", "it", "was", "were", "an", ",", ".", "?", "!", ";", "/"]
    lr = WordNetLemmatizer()
    filtered_text = []
    for w, p in tagged:
        if w not in stop_words and w not in string.punctuation:
            if p in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                filtered_text.append(lr.lemmatize(w, pos='v'))
            elif p in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                filtered_text.append(lr.lemmatize(w, pos='a'))
            else:
                filtered_text.append(w)
    return ' '.join(filtered_text)

def animation_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        analyzed_text = analyze_text(text)
        analyzed_text_list = [analyzed_text]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(analyzed_text_list)
        vocabulary = vectorizer.get_feature_names_out()
        words = set(analyzed_text.split())
        reconstructed_words = []
        for word in analyzed_text.split():
            if word in words:
                if find_video(word):
                    reconstructed_words.append(word)
                else:
                    reconstructed_words.extend(word)
        return render(request, 'animation.html', {'words': reconstructed_words, 'text': text})
    else:
        return render(request, 'animation.html')

# Gesture functions

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    temp_landmark_list = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return [x / max_value for x in temp_landmark_list]

def logging_csv(number, mode, landmark_list):
    if mode in [1, 2] and (0 <= number <= 35):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image

def draw_info_text(image, brect, handedness, text):
    label = handedness.classification[0].label
    cv2.putText(image, f"{label}: {text}", (brect[0], brect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

def draw_info(image, fps, mode, number):
    cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image

class CvFpsCalc:
    def __init__(self, buffer_len=10):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        diff = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(diff)
        return round(1000.0 / (sum(self._difftimes) / len(self._difftimes)), 2)

def generate_frames():
    global gesture_text
    cap = cv2.VideoCapture(0)
    fps_calc = CvFpsCalc()
    classifier = KeyPointClassifier()
    with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fps = fps_calc.get()
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                logging_csv(1, 1, pre_processed)
                gesture_id = classifier(pre_processed)
                gesture_text = labels[gesture_id]

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness, gesture_text)

        debug_image = draw_info(debug_image, fps, 1, 1)

        _, jpeg = cv2.imencode('.jpg', debug_image)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

def camera_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_view(request):
    return render(request, 'camera-feed.html')

def perform_prediction(request):
    global gesture_text
    return JsonResponse({ 'character': gesture_text })
