import os
import time
import cv2
import face_recognition
import pickle
import argparse
import numpy as np

def recognize_faces():
    parser = argparse.ArgumentParser(description='Recognize faces using webcam')
    parser.add_argument('--recognition-model', default='models/face_recognition_model.pkl',
                        help='Path to trained face recognition model')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Recognition confidence threshold')
    parser.add_argument('--device', default='2', help='Camera device number')
    
    args = parser.parse_args()
    
    # Load the face recognition model
    with open(args.recognition_model, 'rb') as f:
        model = pickle.load(f)
    
    # Open webcam
    cap = cv2.VideoCapture(int(args.device))
    # Add these lines right here:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.device}")
        return
    
    print("Starting face recognition...")
    print(f"Using confidence threshold: {args.threshold}")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read timeout")
         # You could add a short delay or recovery attempt here
            time.sleep(0.1)
            continue
        
        # Find faces with more generous parameters
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=2)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Try to match the face
            name = "Unknown"
            confidence = 0
            
            try:
                # Get nearest neighbor distance
                distances, indices = model.kneighbors([face_encoding])
                nearest_distance = distances[0][0]
                confidence = 1 - min(nearest_distance, 1.0)
                
                # Use lower threshold for recognition
                if confidence >= args.threshold:
                    name = model.predict([face_encoding])[0]
            except Exception as e:
                print(f"Error in recognition: {e}")
            
            # Display results
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()