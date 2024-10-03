import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)

        
import json
import numpy as np
import cv2
from keras_facenet import FaceNet
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN()

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
    # Crop the face from the frame
    face = frame[y:y+h, x:x+w]
    
    try:
        # Analyze the cropped face for emotions
        analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

        # Handle the case where multiple faces are detected (DeepFace might return a list)
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Get the dominant emotion
        dominant_emotion = analysis['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
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

            # Analyze the current face for emotions
        emotion = analyze_emotion_from_frame(frame, (x, y, w, h))

            # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # if emotion:
        #         # Display the emotion label under the rectangle
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(frame, emotion, (x, y+h+30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
       
        # Compare embeddings
        similarities = cosine_similarity(face_embedding, embeddings)
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[0][best_match_index]
        
        if best_match_score < 0.5:  # Adjust threshold as needed
            label = "Unknown"
        else:
            label = labels[best_match_index]

        # Draw a box and label around the face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, emotion+'\n'+label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Load embeddings and labels from JSON file
json_file = 'face_embeddings.json'
embeddings, labels = load_embeddings_from_json(json_file)

# Capture video from the webcam and recognize faces in real-time
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_face(frame, embeddings, labels)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
