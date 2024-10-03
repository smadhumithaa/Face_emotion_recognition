# import streamlit as st
# import json
# import numpy as np
# import cv2
# from keras_facenet import FaceNet
# from deepface import DeepFace
# from facenet_pytorch import MTCNN
# from sklearn.metrics.pairwise import cosine_similarity
# import time
# from PIL import Image

# # Initialize FaceNet and MTCNN
# embedder = FaceNet()
# detector = MTCNN()

# # Store data of detected faces
# detected_data = []

# def extract_face(image):
#     results = detector.detect(image)
#     if results is None:
#         return None
#     return results

# def get_embedding(face_pixels):
#     face_pixels = cv2.resize(face_pixels, (160, 160))
#     embedding = embedder.embeddings(np.array([face_pixels]))[0]
#     return embedding

# def load_embeddings_from_json(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#         embeddings = np.array(data['embeddings'])
#         labels = data['labels']
#     return embeddings, labels

# def analyze_emotion_from_frame(frame, face_coordinates):
#     x, y, w, h = face_coordinates
#     face = frame[y:y+h, x:x+w]
    
#     try:
#         analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
#         if isinstance(analysis, list):
#             analysis = analysis[0]
#         dominant_emotion = analysis['dominant_emotion']
#         return dominant_emotion
#     except Exception as e:
#         st.error(f"Error analyzing emotion: {e}")
#         return None

# def recognize_face(image, embeddings, labels):
#     boxes, _ = detector.detect(image)
#     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = box.astype(int)
#             face_crop = image[y1:y2, x1:x2]
#             face_embedding = get_embedding(face_crop)

#             # Analyze emotion
#             emotion = analyze_emotion_from_frame(image, (x1, y1, x2-x1, y2-y1))

#             # Compare embeddings
#             face_embedding = np.array([face_embedding])
#             similarities = cosine_similarity(face_embedding, embeddings)
#             best_match_index = np.argmax(similarities)
#             best_match_score = similarities[0][best_match_index]

#             label = labels[best_match_index] if best_match_score >= 0.5 else "Unknown"

#             # Draw box and label
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{emotion}, {label}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#             # Store detected data
#             detected_data.append({
#                 'time': current_time,
#                 'emotion': emotion,
#                 'name': label
#             })
#     return image

# # Load embeddings and labels from JSON file
# json_file = 'face_embeddings.json'
# embeddings, labels = load_embeddings_from_json(json_file)

# # Streamlit app layout
# st.title('Face Recognition and Emotion Detection')
# start_capture = st.button('Start Video Capture')
# stop_capture = st.button('Stop Video Capture')

# # Video capture control
# if start_capture:
#     cap = cv2.VideoCapture(0)
#     frames_processed = 0
#     start = time.time()

#     # Placeholder for displaying the video frame
#     video_placeholder = st.empty()

#     # Real-time video processing loop
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.write("Failed to capture video.")
#             break

#         # Process the frame for face recognition
#         frame = recognize_face(frame, embeddings, labels)

#         # Count frames processed and calculate FPS
#         frames_processed += 1
#         elapsed_time = time.time() - start
#         if elapsed_time >= 1.0:
#             fps = frames_processed / elapsed_time
#             st.write(f'FPS: {fps:.2f}')
#             frames_processed = 0
#             start = time.time()

#         # Convert frame to RGB (OpenCV uses BGR by default)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Convert the frame to PIL Image format
#         frame_pil = Image.fromarray(frame_rgb)

#         # Display the frame in Streamlit
#         video_placeholder.image(frame_pil)

#         # Stop the loop on Streamlit button press
#         if stop_capture:
#             break

#     cap.release()

# # Display detected face data
# if len(detected_data) > 0:
#     st.write("Detected Faces Data:")
#     st.json(detected_data)


import streamlit as st
import json
import numpy as np
import cv2
from keras_facenet import FaceNet
from deepface import DeepFace
from facenet_pytorch import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import time
from PIL import Image
import mysql.connector
from mysql.connector import Error

# Initialize FaceNet and MTCNN
@st.cache_resource
def model():
    embedder = FaceNet()
    detector = MTCNN(keep_all=True, post_process=False, device="cuda:0")
    return embedder, detector


embedder, detector  = model()

# Store data of detected faces
detected_data = []

# Function to connect to the MySQL database
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',  # Your MySQL host
            user='root',  # Your MySQL username
            password='root',  # Your MySQL password
            port = '3306',
            database='face_emotion_recog_data'  # Your MySQL database name
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Function to insert data into MySQL
def insert_face_data(name, emotion, timestamp):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        insert_query = """INSERT INTO detected_faces_emotion (name, emotion, timestamp) VALUES (%s, %s, %s)"""
        cursor.execute(insert_query, (name, emotion, timestamp))
        print("Updated")
        connection.commit()
        cursor.close()
        connection.close()

def extract_face(image):
    results = detector.detect(image)
    if results is None:
        return None
    return results

# def get_embedding(face_pixels):
#     face_pixels = cv2.resize(face_pixels, (160, 160))
#     embedding = embedder.embeddings(np.array([face_pixels]))[0]
#     return embedding

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
    face = frame[y:y+h, x:x+w]
    
    try:
        analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        dominant_emotion = analysis['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        st.error(f"Error analyzing emotion: {e}")
        return None

# def recognize_face(image, embeddings, labels):
#     boxes, _ = detector.detect(image)
#     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = box.astype(int)
#             face_crop = image[y1:y2, x1:x2]
#             face_embedding = get_embedding(face_crop)

#             # Analyze emotion
#             emotion = analyze_emotion_from_frame(image, (x1, y1, x2-x1, y2-y1))

#             # Compare embeddings
#             face_embedding = np.array([face_embedding])
#             similarities = cosine_similarity(face_embedding, embeddings)
#             best_match_index = np.argmax(similarities)
#             best_match_score = similarities[0][best_match_index]

#             label = labels[best_match_index] if best_match_score >= 0.5 else "Unknown"

#             # Draw box and label
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{emotion}, {label}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#             # Store detected data
#             detected_data.append({
#                 'time': current_time,
#                 'emotion': emotion,
#                 'name': label
#             })

#             # Insert data into MySQL database
#             insert_face_data(label, emotion, current_time)

#     return image


def recognize_face(image, embeddings, labels):
    boxes, _ = detector.detect(image)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            face_crop = image[y1:y2, x1:x2]

            # Check if the face crop is valid before proceeding
            face_embedding = get_embedding(face_crop)
            if face_embedding is None:
                continue  # Skip this face if embedding is not available

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
            cv2.putText(image, f"{emotion}, {label}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Store detected data
            detected_data.append({
                'time': current_time,
                'emotion': emotion,
                'name': label
            })

            # # Insert data into MySQL database
            insert_face_data(label, emotion, current_time)

    return image


# Load embeddings and labels from JSON file
json_file = 'face_embeddings.json'
embeddings, labels = load_embeddings_from_json(json_file)

# Streamlit app layout
st.title('Face Recognition and Emotion Detection')
start_capture = st.button('Start Video Capture')
stop_capture = st.button('Stop Video Capture')

# Video capture control
if start_capture:
    cap = cv2.VideoCapture(0)
    frames_processed = 0
    start = time.time()

    # Placeholder for displaying the video frame
    video_placeholder = st.empty()

    # Real-time video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        # Process the frame for face recognition
        frame = recognize_face(frame, embeddings, labels)

        # Count frames processed and calculate FPS
        frames_processed += 1
        elapsed_time = time.time() - start
        if elapsed_time >= 1.0:
            fps = frames_processed / elapsed_time
            st.write(f'FPS: {fps:.2f}')
            frames_processed = 0
            start = time.time()

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image format
        frame_pil = Image.fromarray(frame_rgb)

        # Display the frame in Streamlit
        video_placeholder.image(frame_pil)

        # Stop the loop on Streamlit button press
        if stop_capture:
            break

    cap.release()

# Display detected face data
if len(detected_data) > 0:
    st.write("Detected Faces Data:")
    st.json(detected_data)
