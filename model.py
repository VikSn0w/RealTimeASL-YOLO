# Train YOLOv8 model
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8l.pt')  # Load a new model from the yaml configuration file
    model.train(data='dataset/data.yaml', epochs=100, batch=32)