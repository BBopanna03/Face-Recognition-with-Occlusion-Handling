import cv2
import os
import face_recognition
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train_model():
    print("Training occlusion-resistant model...")
    data_dir = 'data/faces'
    person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    known_encodings = []
    known_names = []
    
    for person in person_dirs:
        person_path = os.path.join(data_dir, person)
        images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(images)} images for {person}")
        
        for img_file in images:
            img_path = os.path.join(person_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # More aggressive face detection parameters
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model="hog",
                number_of_times_to_upsample=2  # More upsampling helps with partial faces
            )
            
            if not face_locations:
                continue
                
            # More jitters for better encoding quality
            encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations,
                num_jitters=10  # More jitters for better quality
            )
            
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)
    
    if not known_encodings:
        print("Error: No face encodings generated. Check your training data.")
        return None
        
    print(f"Created {len(known_encodings)} encodings for {len(set(known_names))} people")
    
    # Use fewer neighbors for more lenient matching
    n_neighbors = min(3, len(known_encodings))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(known_encodings, known_names)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/occlusion_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
        
    print("Model trained and saved to models/occlusion_model.pkl")
    return knn

def recognize_faces(threshold=0.3):
    # Try to load the model
    if os.path.exists('models/occlusion_model.pkl'):
        with open('models/occlusion_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        print("Model not found. Training now...")
        model = train_model()
        
    if model is None:
        print("Could not load or train model. Exiting.")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting occlusion-resistant face recognition...")
    print(f"Using threshold: {threshold}")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Aggressive face detection
        face_locations = face_recognition.face_locations(
            rgb_small_frame, 
            model="hog",
            number_of_times_to_upsample=2  # More upsampling for occlusions
        )
        
        # Scale back face locations
        scaled_locations = [(top*2, right*2, bottom*2, left*2) 
                          for (top, right, bottom, left) in face_locations]
        
        # Process each face
        if face_locations:
            # More jitters for better accuracy with occlusions
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, 
                face_locations,
                num_jitters=5  # More jitters helps with occlusions
            )
            
            for (top, right, bottom, left), face_encoding in zip(scaled_locations, face_encodings):
                try:
                    # Get nearest neighbors
                    distances, indices = model.kneighbors([face_encoding])
                    nearest_distance = distances[0][0]
                    confidence = 1 - min(nearest_distance, 1.0)
                    
                    name = "Unknown"
                    if confidence >= threshold:
                        name = model.predict([face_encoding])[0]
                        print(f"Recognized: {name} with confidence {confidence:.2f}")
                except Exception as e:
                    print(f"Error in recognition: {e}")
                    confidence = 0
                    name = "Error"
                
                # Display results
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display the frame
        cv2.imshow("Occlusion-Resistant Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Occlusion-Resistant Face Recognition")
    print("1. Train model")
    print("2. Run recognition")
    choice = input("Enter choice (1/2): ")
    
    if choice == '1':
        train_model()
    else:
        threshold = float(input("Enter threshold (0.2-0.5, lower = more lenient): ") or "0.3")
        recognize_faces(threshold)

if __name__ == "__main__":
    main()