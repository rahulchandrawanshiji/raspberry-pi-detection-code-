from ultralytics import YOLO

# YOLOv8 model load karein
model = YOLO("yolov8n.pt")  # Model file

# Model ki class names print karein
print(model.names)
