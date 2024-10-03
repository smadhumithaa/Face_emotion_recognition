import streamlit as st
import tensorflow as tf
import json
import numpy as np
import cv2
from keras_facenet import FaceNet
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


# Initialize FaceNet and MTCNN
@st.cache_resource
def model():
    embedder = FaceNet()
    detector = MTCNN()
    return embedder, detector

embedder, detector  = model()
# Database setup
Base = declarative_base()

class EmotionRecord(Base):
    __tablename__ = 'emotion_records'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    unique_id = Column(String)
    emotion = Column(String)
    timestamp = Column(DateTime)

# Create an SQLite database
engine = create_engine('sqlite:///emotion_data.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def extract_face(image):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    return face

def get_embedding(face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    embedding = embedder.embeddings(np.array([face_pixels]))[0]
    return embedding

def load_embeddings_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        embeddings = np.array(data['embeddings'])
        labels = data['labels']
    return embeddings, labels

def analyze_emotion_from_frame(frame, face_coordinates):
    x, y, w, h = face_coordinates
    face = frame[y:y+h, x:x+w]
    
    try:
        analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        dominant_emotion = analysis['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        st.write(f"Error analyzing emotion: {e}")
        return None

def recognize_face(image, embeddings, labels):
    faces = detector.detect_faces(image)
    for face in faces:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face_crop = image[y1:y2, x1:x2]
        face_embedding = get_embedding(face_crop)
        face_embedding = np.array([face_embedding])
        x, y, w, h = face['box']

        emotion = analyze_emotion_from_frame(image, (x, y, w, h))
        similarities = cosine_similarity(face_embedding, embeddings)
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[0][best_match_index]
        
        if best_match_score < 0.5:
            label = "Unknown"
        else:
            label = labels[best_match_index]

        # Draw box and label on image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}, {emotion}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save to SQL database
        record = EmotionRecord(
            name=label,
            unique_id=str(best_match_index),
            emotion=emotion,
            timestamp=datetime.now()
        )
        session.add(record)
        session.commit()

    return image

# Load embeddings and labels from JSON file
json_file = 'face_embeddings.json'
embeddings, labels = load_embeddings_from_json(json_file)

cap = None

def start_video_capture():
    global cap
    cap = cv2.VideoCapture(0)

def stop_video_capture():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        cv2.destroyAllWindows()

# Streamlit UI
st.title("Live Face Recognition and Emotion Detection")

# Button to start capturing
start_capture = st.button("Start Live Capture", key="start_button")

if start_capture:
    # Initialize video capture
    start_video_capture()
    # cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object to save video
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    stframe = st.empty()

    # Button to stop capture (unique key)
    stop_capture = st.button("Stop Live Capture", key="stop_button")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break
        
        # Process frame for face recognition and emotion detection
        frame = recognize_face(frame, embeddings, labels)
        
        # Write frame to output video file
        # out.write(frame)

        # Display the frame in the Streamlit app
        stframe.image(frame, channels="BGR")

        # Stop capturing when 'Stop Live Capture' button is pressed
        if stop_capture:
            stop_video_capture()
            cap.release()
            break
    
    if cap is not None:
        stop_video_capture()

    # Release resources
    cap.release()
    # # out.release()
    cv2.destroyAllWindows()

# Show database records
st.write("Saved Emotion Records:")
records = session.query(EmotionRecord).all()
for record in records:
    st.write(f"ID: {record.id}, Name: {record.name}, Emotion: {record.emotion}, Time: {record.timestamp}")

