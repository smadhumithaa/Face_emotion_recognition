import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)

from keras_facenet import FaceNet
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
import pickle
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
    # print(embedding)  # Debugging the embeddings
    return embedding

# Load dataset
def load_dataset(directory):
    X, y = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(directory, filename)
            face = extract_face(cv2.imread(path))
            if face is not None:
                face_embedding = get_embedding(face)
                label = filename.split('.')[0]
                X.append(face_embedding)
                y.append(label)
    return np.array(X), np.array(y)

# Directory of your database images
database_dir = r"C:\Users\User 1\Documents\Python Scripts\python projects\face_emotion_recognition\data"
X, y = load_dataset(database_dir)

# Encode labels (names)
# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train an SVM classifier on embeddings
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)

# Save the model and label encoder for later use
with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump((classifier, le), f)

def recognize_face(image):
    faces = detector.detect_faces(image)
    for face in faces:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face_crop = image[y1:y2, x1:x2]
        face_embedding = get_embedding(face_crop)
        # Predict the face
        pred = classifier.predict([face_embedding])
        prob = classifier.predict_proba([face_embedding])
        label = le.inverse_transform(pred)[0]
        if max(prob[0]) <= 0.05:  # Lower the threshold from 0.7 to 0.4 or adjust as needed
            label = "Unknown"
        print(f'Prediction: {label}, Confidence: {max(prob[0])}')  # Debugging

        # Draw a box and label around the face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Capture video from the webcam and recognize faces in real-time
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_face(frame)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
