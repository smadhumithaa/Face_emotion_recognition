import json
import numpy as np
import cv2
from keras_facenet import FaceNet
from deepface import DeepFace
from facenet_pytorch import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import time

# Initialize FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN(device="cuda:0")

def extract_face(image):
    results = detector.detect(image)
    if results is None:
        return None
    return results

def get_embedding(face_pixels):
    if face_pixels is None or face_pixels.size == 0:
        return None
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

        # Handle the case where multiple faces are detected
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Get the dominant emotion
        dominant_emotion = analysis['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return None

def recognize_face(image, embeddings, labels):
    boxes, _ = detector.detect(image)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            face_crop = image[y1:y2, x1:x2]
            face_embedding = get_embedding(face_crop)

            # Analyze emotion
            emotion = analyze_emotion_from_frame(image, (x1, y1, x2-x1, y2-y1))

            # Compare embeddings
            face_embedding = np.array([face_embedding])
            similarities = cosine_similarity(face_embedding, embeddings)
            best_match_index = np.argmax(similarities)
            best_match_score = similarities[0][best_match_index]

            label = labels[best_match_index] if best_match_score >= 0.5 else "Unknown"

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{emotion},{label}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Load embeddings and labels from JSON file
json_file = 'face_embeddings.json'
embeddings, labels = load_embeddings_from_json(json_file)

# Capture video from the webcam and recognize faces in real-time
cap = cv2.VideoCapture(0)
frames_processed = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for face recognition
    frame = recognize_face(frame, embeddings, labels)
    
    # Count frames processed and calculate FPS
    frames_processed += 1
    elapsed_time = time.time() - start
    if elapsed_time >= 1.0:  # Update FPS every second
        fps = frames_processed / elapsed_time
        print(f'Frames per second: {fps:.2f}, Total frames processed: {frames_processed}')
        frames_processed = 0
        start = time.time()  # Reset start time for the next interval

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
