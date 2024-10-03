import cv2
import numpy as np
import torch
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained models
mtcnn = MTCNN(keep_all=True)  # Keep all detected faces
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load embeddings from a JSON file (or any other format)
def load_embeddings(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        embeddings = np.array(data['embeddings'])
        labels = data['labels']
    return embeddings, labels

embeddings, labels = load_embeddings('face_embeddings_pytorch.json')

def classify_face(face_embedding):
    similarities = cosine_similarity([face_embedding], embeddings)
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[0][best_match_index]
    
    if best_match_score < 0.4:  # Threshold for unknown classification
        return "Unknown"
    else:
        return labels[best_match_index]

def main():
    print("Starting video stream. Press 'q' to quit.")
    
    video_capture = cv2.VideoCapture(0)  # Use the first camera
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = frame[y1:y2, x1:x2]
                
                # Get embedding for the detected face
                face_embedding = model(torch.tensor(face_crop).permute(2, 0, 1).float().unsqueeze(0))
                label = classify_face(face_embedding.detach().numpy())
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the resulting frame in a window
        cv2.imshow('Face Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()