import os
import cv2
import argparse
import time
from ultralytics import YOLO

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def collect_face_data():
    parser = argparse.ArgumentParser(description='Collect face data for recognition')
    parser.add_argument('--name', required=True, help='Name of the person')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to collect')
    parser.add_argument('--data-dir', default='data/faces', help='Directory to store face data')
    parser.add_argument('--model', default='models/yolov8n-face.pt', help='Path to YOLO model')
    parser.add_argument('--device', default='0', help='Camera device number')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between captures (seconds)')
    
    args = parser.parse_args()
    
    # Create directory for this person
    person_dir = os.path.join(args.data_dir, args.name)
    create_directory(person_dir)
    
    # Load YOLO model for face detection
    model = YOLO("C:/Users/vaish/OneDrive/Desktop/Face_Recognision/models/yolov8n-face.pt")
    model = YOLO("yolov8n.pt")
    # Open webcam
    cap = cv2.VideoCapture(int(args.device))
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.device}")
        return
    
    print(f"Starting face collection for {args.name}...")
    print(f"Please position your face in different poses and conditions")
    print(f"Collecting {args.num_images} images...")
    
    count = 0
    
    while count < args.num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Detect faces using YOLO
        results = model(frame, conf=0.25)
        
        # Display frame with detections
        annotated_frame = results[0].plot()
        
        # Show instructions
        cv2.putText(
            annotated_frame, 
            f"Collecting: {count}/{args.num_images} | Press SPACE to capture", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow("Face Collection", annotated_frame)
        
        key = cv2.waitKey(1)
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press SPACE to capture
        if key == 32:  # SPACE key
            # Get the original frame (without annotations)
            filename = os.path.join(person_dir, f"{args.name}_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1
            time.sleep(args.delay)  # Wait between captures
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images for {args.name}")

if __name__ == "__main__":
    collect_face_data()