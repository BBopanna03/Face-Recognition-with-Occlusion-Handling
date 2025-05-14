import os
import cv2
import numpy as np
import face_recognition
from tqdm import tqdm

class FaceEncoder:
    def __init__(self, model_type='hog'):
        """
        Initialize the face encoder
        
        Args:
            model_type (str): 'hog' for CPU or 'cnn' for GPU (if available)
        """
        self.model_type = model_type
        
    def encode_face(self, image):
        """
        Encode a single face from an image
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            numpy.ndarray: The face encoding
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations (even with partial occlusion)
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model=self.model_type,
            number_of_times_to_upsample=2  # Increase for better detection of occluded faces
        )
        
        if not face_locations:
            return None, None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_image, 
            face_locations,
            num_jitters=5  # Increase for better accuracy with occlusions
        )
        
        if not face_encodings:
            return None, None
        
        return face_encodings[0], face_locations[0]
    
    def encode_faces_from_directory(self, directory):
        """
        Encode all faces from a directory structure
        
        Args:
            directory (str): Path to directory containing person subdirectories
            
        Returns:
            dict: Dictionary of face encodings by person name
        """
        encodings = {}
        names = []
        
        # Get all subdirectories (one per person)
        person_dirs = [d for d in os.listdir(directory) 
                     if os.path.isdir(os.path.join(directory, d))]
        
        for person in person_dirs:
            person_dir = os.path.join(directory, person)
            person_encodings = []
            
            # Get all images for this person
            image_files = [f for f in os.listdir(person_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {len(image_files)} images for {person}")
            
            for image_file in tqdm(image_files):
                image_path = os.path.join(person_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                encoding, _ = self.encode_face(image)
                
                if encoding is not None:
                    person_encodings.append(encoding)
                    names.append(person)
            
            print(f"Successfully encoded {len(person_encodings)} faces for {person}")
            
        return names, np.array(person_encodings) if person_encodings else None