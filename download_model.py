from roboflow import Roboflow

# First install the Roboflow package if not already installed
# !pip install roboflow

# Replace "YOUR_API_KEY" with the actual API key you get from Roboflow
rf = Roboflow(api_key="F8rb4E1M530yOr1Eofh0")
project = rf.workspace("roboflow-universe-projects").project("face-detection-mik1i")
version = project.version(4)
model = version.model

# Download the model to your models directory
model.save("models/yolov8n-face")