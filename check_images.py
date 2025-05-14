import os
import cv2
import face_recognition

data_dir = 'data/faces'
person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for person in person_dirs:
    person_path = os.path.join(data_dir, person)
    images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nChecking {len(images)} images for {person}...")
    
    for img_file in images:
        img_path = os.path.join(person_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  ❌ Could not read {img_file}")
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            print(f"  ❌ No face detected in {img_file}")
        else:
            print(f"  ✓ Face detected in {img_file}")