# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # if gpus:
# #     try:
# #         for gpu in gpus:
# #             tf.config.experimental.set_memory_growth(gpu, True)
# #     except RuntimeError as e:
# #         print(e)

# import tensorflow as tf
# import tensorflow_hub as hub

# # Load a FaceNet model from TensorFlow Hub
# model_url = 'https://tfhub.dev/google/facenet/1'
# model = hub.load(model_url)

# # Define a function to get embeddings from the model
# def get_embedding(face_pixels):
#     face_pixels = face_pixels.astype('float32')
#     mean, std = face_pixels.mean(), face_pixels.std()
#     face_pixels = (face_pixels - mean) / std
#     embeddings = model(tf.convert_to_tensor([face_pixels]))
#     return embeddings.numpy()[0]


# import torch
# import torchvision
# print(torch.__version__)  # e.g., 2.0.0
# print(torchvision.__version__)  # e.g., 0.15.1



import mysql.connector
from mysql.connector import Error

# Initialize FaceNet and MTCNN
# embedder = FaceNet()
# detector = MTCNN()

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
        print(f"Error connecting to MySQL: {e}")
        return None
    
    # Function to insert data into MySQL
def insert_face_data(name, emotion, timestamp):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        insert_query = """INSERT INTO detected_faces (name, emotion, timestamp) VALUES (%s, %s, %s)"""
        cursor.execute(insert_query, (name, emotion, timestamp))
        connection.commit()
        cursor.close()
        connection.close()


create_connection()