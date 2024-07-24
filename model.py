# Train YOLOv8 model
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8l.pt')  # Load a new model from the yaml configuration file
    model.train(data='dataset/data.yaml', epochs=100, batch=4)
    # Load a model
    model = YOLO("runs/detect/train/")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(["im1.jpg", "im2.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk