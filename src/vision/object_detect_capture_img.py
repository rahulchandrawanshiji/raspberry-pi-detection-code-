# from ultralytics import YOLO
# import cv2

# # YOLOv8 model load karein
# model = YOLO("yolov8n.pt")  # Trained model ka path

# # Camera open karein
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("❌ Camera nahi khul paaya")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Frame nahi mila")
#         break

#     # YOLO se detection karein
#     results = model(frame)

#     # Detections ko frame par draw karein
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
#             conf = box.conf[0].item()  # Confidence score
#             cls = int(box.cls[0].item())  # Class index
#             label = model.names[cls]  # Class label

#             if conf > 0.6:  # Confidence threshold
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Show karein
#     cv2.imshow("YOLOv8 Live Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import time
import cv2
from ultralytics import YOLO

# YOLOv8 model load karein
model = YOLO("yolov8n.pt")  # Trained model ka path

# Camera open karein
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera nahi khul paaya")
    exit()

start_time = time.time()  # Start time store karein

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame nahi mila")
        break

    # YOLO se detection karein
    results = model(frame)

    # Detections ko frame par draw karein
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = model.names[cls]  # Class label

            if conf > 0.6:  # Confidence threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show karein
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Agar 10 seconds ho gaye hain to image save karein
    if time.time() - start_time >= 30:
        image_path = "detected_image.jpg"
        cv2.imwrite(image_path, frame)
        print(f"✅ Image saved as {image_path}")
        break  # Loop exit karein

    # Agar user 'q' dabaye to exit karein
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
