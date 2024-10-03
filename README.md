**Face Emotion Recognition System**

**Overview**

This project implements a real-time face recognition and emotion detection system using MTCNN for face detection, FaceNet (via facenet-pytorch) for face recognition, and a custom model for emotion detection. The system captures live video, detects faces, recognizes individuals, and determines their emotions in real-time.

**Features**

Face Detection: Utilizes MTCNN (Multi-task Cascaded Convolutional Networks) to detect faces in the video stream.
Face Recognition: Employs the FaceNet model to extract face embeddings and recognize individuals based on saved embeddings.
Emotion Detection: Uses a custom emotion detection model to classify facial expressions into categories such as happy, sad, angry, neutral, etc.
Real-Time Processing: Captures video from a live webcam and performs face detection, recognition, and emotion classification in real-time.
Live Video Interface: Displays video with bounding boxes and labels for both recognized faces and detected emotions.
Prerequisites

**Hardware Requirements**

A computer with a webcam.
NVIDIA GPU is recommended for better performance, but the system will also work on CPU (with slower processing).

CUDA installed (for GPU usage).

Software Requirements

Python 3.9 or higher.

**Libraries:**

torch (for PyTorch, the deep learning framework)

facenet-pytorch (for MTCNN and InceptionResnetV1 models)

scipy (for calculating cosine distances)

opencv-python (for capturing video and image processing)

numpy (for numerical operations)

cv2 (for video processing)

You can install the required packages by running:

**pip install torch facenet-pytorch scipy opencv-python numpy**

Project Structure
graphql
Copy code
.
├── face_recog_mtcnn.py       # Main script for face detection, recognition, and emotion detection
├── face_embeddings_pytorch.json  # Pre-saved face embeddings and labels in JSON format
├── README.md                 # Project README (this file)
└── requirements.txt          # List of required Python packages

**Dataset**


Face Embeddings: Pre-computed embeddings of known faces are stored in face_embeddings_pytorch.json. Each face is associated with a unique label (the name of the person).
Emotion Labels: The system classifies emotions based on facial expressions and maps them to the corresponding emotion category (e.g., happy, sad, angry, neutral).

How It Works

Face Detection: The system uses MTCNN to detect all faces in a video frame.

Face Recognition: For each detected face, FaceNet is used to extract a 512-dimensional embedding. The system compares this embedding with pre-saved embeddings to recognize the individual based on cosine similarity.

Emotion Detection: After recognizing the face, the system classifies the emotion of the person using the trained emotion detection model.

Live Display: Bounding boxes are drawn around detected faces, and labels (person's name and detected emotion) are displayed in real-time on the video feed.

