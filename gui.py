import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
        cap.release()


class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("Real-Time Hand Gesture Recognition with YOLOv8")
        self.disply_width = 1280
        self.display_height = 720
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.model = model

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.recognize_gesture(cv_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def recognize_gesture(self, frame):
        # Resize frame to the model's input size
        input_size = (640, 640)  # Example size, adjust according to the model

        height, width = frame.shape[:2]

        # Calculate the coordinates for the crop
        center_x, center_y = width // 2, height // 2
        half_width, half_height = 320, 320
        start_x = center_x - half_width
        start_y = center_y - half_height
        end_x = center_x + half_width
        end_y = center_y + half_height

        # Crop the image
        cropped_image = frame[start_y:end_y, start_x:end_x]

        results = self.model(cropped_image)

        # Scale the detection results back to the original frame size
        scale_x = frame.shape[1] / input_size[0]
        scale_y = frame.shape[0] / input_size[1]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                label = model.names[int(box.cls[0])]
                confidence = box.conf[0]
                if confidence > 0.3:  # Adjust confidence threshold as needed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update the displayed image
        qt_img = self.convert_cv_qt(frame)
        self.image_label.setPixmap(qt_img)


if __name__ == "__main__":
    model = YOLO('runs/detect/train/weights/best.pt')  # Load the trained model
    app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec_())
