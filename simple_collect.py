import cv2
import os
import time
import argparse

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def collect_face_data():
    parser = argparse.ArgumentParser(description='Collect face data for recognition')
    parser.add_argument('--name', required=True, help='Name of the person')
    parser.add_argument('--num-images', type=int, default=1, help='Number of images to collect')
    parser.add_argument('--data-dir', default='data/faces', help='Directory to store face data')
    parser.add_argument('--device', default='0', help='Camera device number')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between captures (seconds)')
    
    args = parser.parse_args()
    
    # Create directory for this person
    person_dir = os.path.join(args.data_dir, args.name)
    create_directory(person_dir)
    
    # Open webcam
    cap = cv2.VideoCapture(int(args.device))
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.device}")
        return
    
    print(f"Starting face collection for {args.name}...")
    print(f"Collecting {args.num_images} images automatically...")
    
    count = 0
    
    while count < args.num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Save image automatically
        filename = os.path.join(person_dir, f"{args.name}_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
        time.sleep(args.delay)  # Wait between captures
            
    cap.release()
    print(f"Collected {count} images for {args.name}")

if __name__ == "__main__":  # Note the double underscores
    collect_face_data()