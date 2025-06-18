import os
import string
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
import cv2
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import string
import numpy as np
import nltk
import mediapipe as mp




nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


model = load_model(os.path.join(os.getcwd(), "model", "GASLmodel.keras")) 



# Create your views here.
def index(request):
    return render(request, 'index.html')


def find_video(word):
    path = os.path.join(os.getcwd(), "static", "assets", "ASL", f"{word}.mp4")
    print(path)
    return os.path.isfile(path)


def analyze_text(sentence):
    # Tokenizing the sentence
    words = word_tokenize(sentence.lower())

    # Using NLTK's Part-of-Speech tagging
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

        words = set(analyzed_text.split()) # Convert vocabulary to a set for faster lookup

        # Reconstructing the words
        reconstructed_words = []
        for word in analyzed_text.split():  
            if word in words:  
                if find_video(word):  # Check if a video exists for the word
                    reconstructed_words.append(word)
                else:
                    # If video for word is not present, break the word into letters
                    reconstructed_words.extend(word)
            else: 
                # If word not found, append individual letters
                reconstructed_words.extend(word)

        return render(request, 'animation.html', {'words': reconstructed_words, 'text': text})
    else:
        return render(request, 'animation.html')
    
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
gesture_text = ""

def generate_frames():
    global gesture_text, confidence
    cap = cv2.VideoCapture(0)  # Open the webcam
    prediction = None

    confidence = None
    gesture_text = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Process only the first hand

            # Extract hand landmarks
            h, w, c = frame.shape
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x + 50
                if x < x_min:
                    x_min = x - 50
                if y > y_max:
                    y_max = y + 50
                if y < y_min:
                    y_min = y - 50

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cropped_hand = frame[y_min:y_max, x_min:x_max]

            if cropped_hand is not None and cropped_hand.size != 0:
                cropped_hand_rgb = cv2.resize(cropped_hand, (100, 100))  # Keep 3 channels
                cropped_hand_rgb = cropped_hand_rgb / 255.0
                cropped_hand_rgb = np.expand_dims(cropped_hand_rgb, axis=0)
            else:
                print("Error: The cropped_hand image is empty or None.")
                continue

            prediction = model.predict(cropped_hand_rgb)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
            gesture_text = labels[predicted_class]

        if gesture_text is not None and confidence is not None:
            cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def camera_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_view(request):
    print(request)
    return render(request, 'camera-feed.html')

def perform_prediction(request):
    global gesture_text
  
    predicted_character = gesture_text

    print(predicted_character)


    # Return the predicted data as a JSON response
    return JsonResponse({
        'character': predicted_character,
    },content_type='application/json')