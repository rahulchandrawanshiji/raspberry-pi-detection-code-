# import cv2

# # Only "person" class to reduce load
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor"]

# net = cv2.dnn.readNetFromCaffe(
#     "/home/pi/ui_raspberry_button/simple_ui_project/src/vision/models/MobileNetSSD_deploy.prototxt",
#     "/home/pi/ui_raspberry_button/simple_ui_project/src/vision/models/MobileNetSSD_deploy.caffemodel"
# )



# # Use only working device index 0
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("❌ Camera open nahi ho paya")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Frame grab nahi ho paya")
#         break

#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.6:  # Increase threshold to reduce noise
#             idx = int(detections[0, 0, i, 1])
#             label = CLASSES[idx]
#             if label != "person":
#                 continue  # Ignore everything except person

#             box = detections[0, 0, i, 3:7] * [w, h, w, h]
#             (startX, startY, endX, endY) = box.astype("int")
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow("Live Detection (Press 'q' to Quit)", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




#working code of live vedio
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




from ultralytics import YOLO
import cv2

# YOLOv8 model load karein
print("🟢 YOLO model load ho raha hai...")
model = YOLO("yolov8n.pt")  # Trained model ka path
print("✅ YOLO model successfully loaded!")

# Camera open karein
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera nahi khul paaya")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame nahi mila")
        break

    # YOLO se detection karein
    results = model(frame)
    print(results)  # 👉 Yeh print karein output dekhne ke liye

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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
































# '''from ultralytics import YOLO

# # YOLOv8 model load karein
# model = YOLO("yolov8n.pt")  # Pre-trained COCO model

# # Available classes ko print karein
# print("📌 YOLOv8 Model se Detect Hone Wale Objects:")
# for class_id, class_name in model.names.items():
#     print(f"{class_id}: {class_name}")
