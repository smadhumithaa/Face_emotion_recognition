import json
import numpy as np
import os
import cv2
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

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

def save_embeddings_to_json(directory, json_file):
    embeddings_list = []
    labels_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(directory, filename)
            face = extract_face(cv2.imread(path))
            if face is not None:
                face_embedding = get_embedding(face)
                label = filename.split('.')[0]
                embeddings_list.append(face_embedding.tolist())
                labels_list.append(label)
    
    with open(json_file, 'w') as f:
        json.dump({'embeddings': embeddings_list, 'labels': labels_list}, f)
    print("Embeddings and labels saved to JSON.")

# Save embeddings and labels to JSON file
database_dir = r"C:\Users\User 1\Documents\Python Scripts\python projects\face_emotion_recognition\data"
json_file = 'face_embeddings.json'
save_embeddings_to_json(database_dir, json_file)
