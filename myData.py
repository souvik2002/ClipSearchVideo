import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import os

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Function to extract features from a frame
def extract_features(frame):
    # Resize frame to 224x224
    frame = cv2.resize(frame, (224, 224))
    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Preprocess input for VGG16 model
    frame = preprocess_input(frame)
    # Extract features using VGG16 model
    features = model.predict(np.array([frame]))
    return features

# Function to search for videos
def search_videos(search_item):
    # Extract features from search item
    search_features = extract_features(search_item)
    results=[]
    # Loop through all frames in the database
    for root, dirs, files in os.walk("video_frames"):
        for file in files:
            # Load frame from file
            frame = cv2.imread(os.path.join(root, file))
            # Extract features from frame
            frame_features = extract_features(frame)
            # Compute similarity between search features and frame features
            similarity = np.dot(search_features, frame_features.T)
            # If similarity is above a threshold, add video to results
            if similarity > 0.9:
                video_path = os.path.join(root, "..", "video.mp4")
                results.append(video_path)
    return results