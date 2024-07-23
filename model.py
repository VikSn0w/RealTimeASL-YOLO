# Train YOLOv8 model
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.pt')  # Load a new model from the yaml configuration file
model.train(data='dataset/data.yaml', epochs=50, batch=4)
