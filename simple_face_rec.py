import cv2
import os
import numpy as np
import face_recognition
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Train model from images (if needed)
def train_model():
    print("Training model...")
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
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                continue
                
            encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)
    
    if not known_encodings:
        print("Error: No face encodings generated. Check your training data.")
        return None
        
    print(f"Created {len(known_encodings)} encodings for {len(set(known_names))} people")
    
    # Create and train the KNN classifier
    n_neighbors = min(5, len(known_encodings))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(known_encodings, known_names)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/simple_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
        
    print("Model trained and saved to models/simple_model.pkl")
    return knn

# Step 2: Recognize faces
def recognize_faces(model=None, threshold=0.4):
    if model is None:
        # Load model
        if os.path.exists('models/simple_model.pkl'):
            with open('models/simple_model.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            print("No model found. Training a new one...")
            model = train_model()
            
        if model is None:
            print("Could not create or load a model. Exiting.")
            return
    
    # Open webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces with more generous parameters
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=2)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=2)
            
            # Scale locations back to original size
            scaled_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]
            
            for (top, right, bottom, left), face_encoding in zip(scaled_locations, face_encodings):
                # Try to match the face
                name = "Unknown"
                confidence = 0
                
                try:
                    # Get nearest neighbor distance
                    distances, indices = model.kneighbors([face_encoding])
                    nearest_distance = distances[0][0]
                    confidence = 1 - min(nearest_distance, 1.0)
                    
                    # Use threshold for recognition
                    if confidence >= threshold:
                        name = model.predict([face_encoding])[0]
                        print(f"Recognized: {name} with confidence {confidence:.2f}")
                except Exception as e:
                    print(f"Error in recognition: {e}")
                
                # Display results
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display the frame
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function - train and recognize 
def main():
    print("Face Recognition System")
    print("1. Train a new model")
    print("2. Run face recognition")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        train_model()
    else:
        threshold = float(input("Enter recognition threshold (0.1-0.6, lower = more lenient): ") or "0.4")
        recognize_faces(threshold=threshold)

if __name__ == "__main__":
    main()