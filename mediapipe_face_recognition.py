import cv2
import os
import pickle
import numpy as np
import time
import argparse
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import face_recognition  # Still needed for face encodings

class MediapipeFaceRecognizer:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range detection
            min_detection_confidence=0.5
        )
        
        # Path for the face recognition model
        self.model_path = 'models/mediapipe_face_model.pkl'
        self.data_dir = 'data/faces'
        
    def train_model(self):
        """Train face recognition model using MediaPipe for detection and face_recognition for encodings"""
        print("Training model with MediaPipe face detection...")
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory {self.data_dir} not found")
            return None
            
        person_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        known_encodings = []
        known_names = []
        
        for person in person_dirs:
            person_path = os.path.join(self.data_dir, person)
            images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {len(images)} images for {person}")
            
            for img_file in images:
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Could not read {img_path}")
                    continue
                
                # Convert to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.face_detection.process(rgb_image)
                
                if not results.detections:
                    print(f"No face detected in {img_file}")
                    continue
                
                # Get face location from MediaPipe detection
                # (We still need face_recognition for the encodings)
                height, width, _ = image.shape
                detection = results.detections[0]  # Use the first detected face
                
                # Convert MediaPipe detection to face_recognition format (top, right, bottom, left)
                bbox = detection.location_data.relative_bounding_box
                x_min = max(0, int(bbox.xmin * width))
                y_min = max(0, int(bbox.ymin * height))
                x_max = min(width, int((bbox.xmin + bbox.width) * width))
                y_max = min(height, int((bbox.ymin + bbox.height) * height))
                
                # Create face location in format expected by face_recognition
                face_location = (y_min, x_max, y_max, x_min)
                
                # Get face encoding using face_recognition
                encodings = face_recognition.face_encodings(rgb_image, [face_location])
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person)
        
        if not known_encodings:
            print("Error: No face encodings generated. Check your training data.")
            return None
            
        print(f"Created {len(known_encodings)} encodings for {len(set(known_names))} people")
        
        # Create and train the KNN classifier
        n_neighbors = min(3, len(known_encodings))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        knn.fit(known_encodings, known_names)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(knn, f)
            
        print(f"Model trained and saved to {self.model_path}")
        return knn
    
    def recognize_faces(self, threshold=0.4, skip_frames=2):
        """Recognize faces using MediaPipe for detection and trained model for recognition"""
        # Try to load the model
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print("Model not found. Training now...")
            model = self.train_model()
            
        if model is None:
            print("Could not load or train model. Exiting.")
            return
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Set properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting face recognition with MediaPipe...")
        print(f"Using confidence threshold: {threshold}")
        print("Press 'q' to quit")
        
        frame_count = 0
        processing_times = []
        
        with self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    print("Error reading from webcam")
                    break
                
                frame_count += 1
                
                # Skip frames to improve performance
                if frame_count % (skip_frames + 1) != 0:
                    # Just show the frame without processing
                    cv2.imshow("MediaPipe Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe (much faster than HOG)
                results = face_detection.process(rgb_frame)
                
                # Display FPS
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Process results
                if results.detections:
                    height, width, _ = frame.shape
                    
                    for detection in results.detections:
                        # Draw face detection box
                        self.mp_drawing.draw_detection(frame, detection)
                        
                        # Get face location
                        bbox = detection.location_data.relative_bounding_box
                        x_min = max(0, int(bbox.xmin * width))
                        y_min = max(0, int(bbox.ymin * height))
                        x_max = min(width, int((bbox.xmin + bbox.width) * width))
                        y_max = min(height, int((bbox.ymin + bbox.height) * height))
                        
                        # Convert to format needed for face_recognition
                        face_location = (y_min, x_max, y_max, x_min)
                        
                        try:
                            # Get face encoding
                            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])
                            
                            if face_encoding:
                                # Get nearest neighbors
                                distances, indices = model.kneighbors([face_encoding[0]])
                                nearest_distance = distances[0][0]
                                confidence = 1 - min(nearest_distance, 1.0)
                                
                                name = "Unknown"
                                if confidence >= threshold:
                                    name = model.predict([face_encoding[0]])[0]
                                
                                # Display name and confidence
                                label = f"{name} ({confidence:.2f})"
                                cv2.putText(frame, label, (x_min, y_min - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                        except Exception as e:
                            print(f"Error in recognition: {e}")
                
                # Show frame
                cv2.imshow("MediaPipe Face Recognition", frame)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Print average processing time
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Average processing time per frame: {avg_time*1000:.2f} ms")
            print(f"Average FPS: {1.0/avg_time:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def collect_face_data(self, name, num_images=20):
        """Collect face data using MediaPipe for detection"""
        # Create directory for this person
        person_dir = os.path.join(self.data_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera")
            return
        
        print(f"Starting face collection for {name}...")
        print(f"Please position your face in different poses and conditions")
        print(f"Collecting {num_images} images...")
        print("Press SPACE to capture, 'q' to quit")
        
        count = 0
        
        with self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            
            while count < num_images:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = face_detection.process(rgb_frame)
                
                # Copy frame for display
                display_frame = frame.copy()
                
                # Display face detection
                if results.detections:
                    for detection in results.detections:
                        self.mp_drawing.draw_detection(display_frame, detection)
                
                # Show instructions
                cv2.putText(
                    display_frame, 
                    f"Collecting: {count}/{num_images} | Press SPACE to capture", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow("Face Collection", display_frame)
                
                key = cv2.waitKey(1)
                
                # Press 'q' to quit
                if key == ord('q'):
                    break
                
                # Press SPACE to capture
                if key == 32:  # SPACE key
                    # Check if face is detected
                    if results.detections:
                        # Save the original frame
                        filename = os.path.join(person_dir, f"{name}_{count:03d}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"Saved {filename}")
                        count += 1
                        time.sleep(0.5)  # Wait between captures
                    else:
                        print("No face detected! Please position your face in the frame.")
            
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {count} images for {name}")

def main():
    parser = argparse.ArgumentParser(description='MediaPipe Face Recognition System')
    parser.add_argument('--mode', choices=['train', 'recognize', 'collect'], default='recognize',
                        help='Mode to run (train, recognize, or collect)')
    parser.add_argument('--name', help='Name of the person for face collection')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to collect')
    parser.add_argument('--threshold', type=float, default=0.4, help='Recognition threshold')
    parser.add_argument('--skip-frames', type=int, default=2, help='Number of frames to skip during recognition')
    
    args = parser.parse_args()
    
    recognizer = MediapipeFaceRecognizer()
    
    if args.mode == 'train':
        recognizer.train_model()
    elif args.mode == 'collect':
        if not args.name:
            args.name = input("Enter name of the person: ")
        recognizer.collect_face_data(args.name, args.num_images)
    else:  # recognize mode
        recognizer.recognize_faces(args.threshold, args.skip_frames)

if __name__ == "__main__":
    main()