import os
import cv2
import face_recognition

# Create a data checking script
def check_training_data():
    data_dir = 'data/faces/Vaishnavi'
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return
        
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Error: No images found in {data_dir}")
        return
        
    print(f"Found {len(image_files)} images. Checking for faces...")
    
    valid_faces = 0
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error: Could not read {img_path}")
            continue
            
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        
        if face_locations:
            valid_faces += 1
            print(f"✓ {img_file}: Found {len(face_locations)} face(s)")
        else:
            print(f"✗ {img_file}: No faces detected")
    
    print(f"Summary: Found faces in {valid_faces} out of {len(image_files)} images")
    print(f"Success rate: {valid_faces/len(image_files)*100:.1f}%")

check_training_data()