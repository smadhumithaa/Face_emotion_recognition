import json
import numpy as np
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN for face detection and InceptionResnetV1 for generating embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=False, device=device)  # Single face detection
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face(image):
    # Convert image from BGR (OpenCV) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect faces
    face = detector(image_rgb)
    if face is None:
        return None
    # Return detected face as a PIL Image
    return face

def get_embedding(face_pixels):
    # Resizing is done internally in InceptionResnetV1; we only need to transform it to tensor
    face_pixels = face_pixels.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        embedding = embedder(face_pixels).cpu().numpy()[0]  # Extract embedding and move to CPU
    return embedding

def save_embeddings_to_json(directory, json_file):
    embeddings_list = []
    labels_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            face = extract_face(image)
            if face is not None:
                face_embedding = get_embedding(face)
                label = filename.split('.')[0]  # Use filename as the label
                embeddings_list.append(face_embedding.tolist())
                labels_list.append(label)

    # Save the embeddings and labels to a JSON file
    with open(json_file, 'w') as f:
        json.dump({'embeddings': embeddings_list, 'labels': labels_list}, f)
    print(f"Embeddings and labels saved to {json_file}")

# Save embeddings and labels to JSON file
database_dir = r"C:\Users\User 1\Documents\Python Scripts\python projects\face_emotion_recognition\data"
json_file = 'face_embeddings_pytorch.json'
save_embeddings_to_json(database_dir, json_file)
