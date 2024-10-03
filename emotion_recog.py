# import cv2
# from deepface import DeepFace

# # Load the pre-trained Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to analyze emotions for each detected face
# def analyze_emotion_from_frame(frame, face_coordinates):
#     x, y, w, h = face_coordinates
#     # Crop the face from the frame
#     face = frame[y:y+h, x:x+w]
    
#     try:
#         # Analyze the cropped face for emotions
#         analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
#         print(analysis)
#         # Handle the case where multiple faces are detected (DeepFace might return a list)
#         if isinstance(analysis, list):
#             analysis = analysis[0]
        
#         # Get the dominant emotion
#         dominant_emotion = analysis['dominant_emotion']
#         return dominant_emotion
#     except Exception as e:
#         print(f"Error analyzing emotion: {e}")
#         return None

# # Function to start live video emotion recognition with face detection and bounding boxes
# def live_emotion_recognition():
#     # Open a connection to the webcam
#     video_capture = cv2.VideoCapture(0)
    
#     if not video_capture.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         # Capture frame-by-frame from webcam
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Failed to capture image")
#             break

#         # Convert the frame to grayscale for face detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#         # Loop through each detected face
#         for (x, y, w, h) in faces:
#             # Analyze the current face for emotions
#             emotion = analyze_emotion_from_frame(frame, (x, y, w, h))

#             # Draw a rectangle around the face
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             if emotion:
#                 # Display the emotion label under the rectangle
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(frame, emotion, (x, y+h+30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         # Display the video frame with bounding boxes and emotion labels
#         cv2.imshow('Live Emotion Recognition', frame)

#         # Press 'q' to quit the video capture
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and close the display window
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     live_emotion_recognition()

import cv2
import time
from deepface import DeepFace
from mtcnn import MTCNN  # Import MTCNN for face detection

# Initialize the MTCNN face detector
detector = MTCNN(min_face_size=30, scale_factor=0.8)  # Adjust these values

# Function to analyze emotions for each detected face
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

# Function to start live video emotion recognition with MTCNN face detection and bounding boxes
def live_emotion_recognition():
    # Open a connection to the webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame from webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        # Start timer
        start_time = time.time()

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        # Loop through each detected face
        for face in faces:
            # Get the bounding box of the face
            x, y, w, h = face['box']

            # Analyze the current face for emotions
            emotion = analyze_emotion_from_frame(frame, (x, y, w, h))

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if emotion:
                # Display the emotion label under the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, emotion, (x, y+h+30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculate the elapsed time and print it
        elapsed_time = time.time() - start_time
        print(f"Time per frame: {elapsed_time:.3f} seconds")

        # Display the video frame with bounding boxes and emotion labels
        cv2.imshow('Live Emotion Recognition', frame)

        # Press 'q' to quit the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the display window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_emotion_recognition()
