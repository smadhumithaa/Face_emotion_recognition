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
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

@st.cache_resource
def model():
    embedder = FaceNet()
    detector = MTCNN()
    return embedder, detector

embedder, detector  = model()

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

json_file = 'face_embeddings.json'
embeddings, labels = load_embeddings_from_json(json_file)
import logging

# Configure logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()

    def transform(self, frame):
        logging.info("Transform called")
        
        image = frame.to_ndarray(format="yuv420p")  # YUV420p format
        if len(image.shape) == 2:  # If it's a single-channel image
            image = cv2.merge([image, image, image])  # Convert to 3 channels
        
        try:
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)  # Convert to BGR
        except cv2.error as e:
            logging.error(f"Error in cvtColor: {e}")
            print(e)
            return frame
        
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        logging.info(f"Current FPS: {fps:.2f}")
        
        # Continue with face detection and other processing...
        faces = detector.detect_faces(image)
        if not faces:
            logging.info("No faces detected.")
            return image

        for face in faces:
            x1, y1, width, height = face['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop is not None:
                logging.info("Face detected, cropping and embedding")
                face_embedding = get_embedding(face_crop)
                face_embedding = np.array([face_embedding])
                emotion = analyze_emotion_from_frame(image, (x1, y1, width, height))
                
                similarities = cosine_similarity(face_embedding, embeddings)
                best_match_index = np.argmax(similarities)
                best_match_score = similarities[0][best_match_index]

                if best_match_score < 0.4:
                    label = "Unknown"
                else:
                    label = labels[best_match_index]
                
                logging.info(f"Detected: {label} with emotion: {emotion}")
                
                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}, {emotion}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return image



# Streamlit UI
st.title("Live Face Recognition and Emotion Detection")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, media_stream_constraints={
            "video": True,
            "audio": False
        })


# Show database records
st.write("Saved Emotion Records:")
