# import cv2
# from facenet_pytorch import MTCNN
# from PIL import Image
# import torch
# import time
# from tqdm.notebook import tqdm

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# class FastMTCNN(object):

#     def __init__(self, stride, resize=1, *args, **kwargs):

#         self.stride = stride
#         self.resize = resize
#         self.mtcnn = MTCNN(*args, **kwargs)
        
#     def __call__(self, frames):

#         if self.resize != 1:
#             frames = [
#                 cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
#                     for f in frames
#             ]
                      
#         boxes, probs = self.mtcnn.detect(frames[::self.stride])

#         faces = []
#         for i, frame in enumerate(frames):
#             box_ind = int(i / self.stride)
#             if boxes[box_ind] is None:
#                 continue
#             for box in boxes[box_ind]:
#                 box = [int(b) for b in box]
#                 faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
#         return faces

# fast_mtcnn = FastMTCNN(
#     stride=4,
#     resize=0.5,
#     margin=14,
#     factor=0.6,
#     min_face_size=50,
#     keep_all=True,
#     device=device
# )

# def run_detection(fast_mtcnn):
#     cap = cv2.VideoCapture(0)  # Capture video from default camera
#     frames_processed = 0
#     faces_detected = 0
#     batch_size = 60
#     start = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames = [frame]

#         faces = fast_mtcnn(frames)

#         frames_processed += len(frames)
#         faces_detected += len(faces)

#         print(
#             f'Frames per second: {frames_processed / (time.time() - start):.3f},',
#             f'faces detected: {faces_detected}\r',
#             end=''
#         )

#         # Display the resulting frame
#         for face in faces:
#             face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
#             cv2.imshow('Face', face)
#             cv2.waitKey(1)

#     cap.release()
#     cv2.destroyAllWindows()

# run_detection(fast_mtcnn)


import cv2
from facenet_pytorch import MTCNN
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class FastMTCNN(object):
    def __init__(self, stride, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]
        
        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        faces = []
        
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            # Check if boxes for this frame are None
            if boxes is None or boxes[box_ind] is None:
                continue
            
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
        return faces, frames

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.6,
    min_face_size=20,
    keep_all=True,
    device=device
)

def run_detection(fast_mtcnn):
    cap = cv2.VideoCapture(0)  # Capture video from default camera
    frames_processed = 0
    faces_detected = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames = [frame_rgb]

        faces, annotated_frames = fast_mtcnn(frames)

        frames_processed += len(frames)
        faces_detected += len(faces)

        # Display the full frame with bounding boxes
        cv2.imshow('Video', frame)

        print(
            f'Frames per second: {frames_processed / (time.time() - start):.3f},',
            f'faces detected: {faces_detected}\r',
            end=''
        )

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_detection(fast_mtcnn)
