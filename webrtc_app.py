import streamlit as st
import cv2
import json
import numpy as np
from keras_facenet import FaceNet
from facenet_pytorch import MTCNN
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

# Initialize FaceNet and MTCNN
@st.cache_resource
def load_model():
    embedder = FaceNet()
    detector = MTCNN(keep_all=True, post_process=False)
    return embedder, detector

embedder, detector  = load_model()
# Load embeddings and labels from JSON file
def load_embeddings_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        embeddings = np.array(data['embeddings'])
        labels = data['labels']
    return embeddings, labels

# Function to get embedding of a face
def get_embedding(face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    embedding = embedder.embeddings(np.array([face_pixels]))[0]
    return embedding


# Analyze emotion asynchronously
def analyze_emotion_from_frame(face):
    try:
        analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        print("Dominant Emotion:", analysis['dominant_emotion'])
        return analysis['dominant_emotion']
    except Exception as e:
        print("Error analyzing emotion:", str(e))

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.embedder, self.detector = load_model()
        self.embeddings, self.labels = load_embeddings_from_json('face_embeddings.json')

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Detect faces in the image
        boxes, _ = self.detector.detect(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = img[y1:y2, x1:x2]

                if face_crop.size > 0:
                    # Get the embedding for the detected face
                    face_embedding = get_embedding(face_crop)
                    if face_embedding is None:
                        continue  # Skip this face if embedding is not available

                    # Analyze emotion
                    emotion = analyze_emotion_from_frame(face_crop)  # Adjusted to pass only the face crop

                    # Compare embeddings
                    face_embedding = np.array([face_embedding])
                    similarities = cosine_similarity(face_embedding, self.embeddings)
                    best_match_index = np.argmax(similarities)
                    best_match_score = similarities[0][best_match_index]

                    # Labeling logic
                    label = self.labels[best_match_index] if best_match_score >= 0.5 else "Unknown"

                    # Draw bounding box and label (optional)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} ({emotion})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
json_file = 'face_embeddings.json'
embeddings, labels = load_embeddings_from_json(json_file)
# Streamlit interface
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, video_frame_callback=None)
