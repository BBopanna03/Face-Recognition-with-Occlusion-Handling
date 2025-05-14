import os
import argparse
from utils.face_encoder import FaceEncoder
from utils.face_matcher import FaceMatcher

def train_face_recognizer():
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--data-dir', default='data/faces', help='Directory containing face data')
    parser.add_argument('--model-output', default='models/face_recognition_model.pkl', 
                        help='Output path for trained model')
    parser.add_argument('--model-type', default='hog', choices=['hog', 'cnn'],
                        help='Face detection model type (hog for CPU, cnn for GPU)')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    
    print("Training face recognition model...")
    print(f"Using face data from: {args.data_dir}")
    
    # Initialize face encoder
    encoder = FaceEncoder(model_type=args.model_type)
    
    # Encode all faces from the data directory
    print("Encoding faces...")
    names, encodings = encoder.encode_faces_from_directory(args.data_dir)
    
    if encodings is None or len(encodings) == 0:
        print("Error: No face encodings were generated. Check your training data.")
        return
    
    print(f"Encoded {len(encodings)} faces for {len(set(names))} people")
    
    # Train the face matcher
    print("Training recognition model...")
    matcher = FaceMatcher()
    matcher.train(names, encodings,n_neighbors=1)
    
    # Save the model
    matcher.save_model(args.model_output)
    print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    train_face_recognizer()