import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class FaceMatcher:
    def __init__(self, model_path=None):
        """
        Initialize the face matcher
        
        Args:
            model_path (str, optional): Path to saved model file
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def train(self, names, encodings, n_neighbors=5):
        """
        Train the face recognition model
        
        Args:
            names (list): List of person names
            encodings (numpy.ndarray): Array of face encodings
            n_neighbors (int): Number of neighbors for KNN algorithm
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.model.fit(encodings, names)
        
    def predict(self, face_encoding, threshold=0.5):
        """
        Predict the identity of a face
        
        Args:
            face_encoding (numpy.ndarray): The face encoding to match
            threshold (float): Confidence threshold for recognition
            
        Returns:
            str: Name of the recognized person, or "Unknown"
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Reshape for single sample prediction
        face_encoding = face_encoding.reshape(1, -1)
        
        # Get nearest neighbors
        distances, indices = self.model.kneighbors(face_encoding)
        
        # Calculate confidence (1 - average distance)
        avg_distance = np.mean(distances[0])
        confidence = 1 - min(avg_distance, 1.0)  # Cap at 0
        
        if confidence >= threshold:
            return self.model.predict(face_encoding)[0], confidence
        else:
            return "Unknown", confidence
        
    def save_model(self, model_path):
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the model file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)