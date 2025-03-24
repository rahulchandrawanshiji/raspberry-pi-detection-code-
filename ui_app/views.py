# from django.shortcuts import render
# from django.http import JsonResponse
# import subprocess

# def home(request):
#     return render(request, 'index.html')

# def start_detection(request):
#     try:
#         subprocess.Popen(["python", "src/vision/object_detector.py"])  # Script path
#         return JsonResponse({"message": "Object detection started!"})
#     except Exception as e:
#         return JsonResponse({"message": f"Error: {str(e)}"}, status=500)


#working code 
# from django.shortcuts import render
# from django.http import JsonResponse, StreamingHttpResponse
# import subprocess
# import cv2

# def home(request):
#     return render(request, 'index.html')

# def start_detection(request):
#     try:
#         subprocess.Popen(["python", "src/vision/object_detector.py"])
#         return JsonResponse({"message": "Object detection started!"})
#     except Exception as e:
#         return JsonResponse({"message": f"Error: {str(e)}"}, status=500)

# def generate_frames():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# def video_feed(request):
#     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')





from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import cv2
from ultralytics import YOLO

# Load YOLO Model
model = YOLO("yolov8n.pt")

def home(request):
    return render(request, 'index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open Camera
    if not cap.isOpened():
        print("❌ Error: Camera not opened")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # **🔥 YOLO Detection Integration**
        results = model(frame)  # Run Object Detection
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                if conf > 0.5:  # Confidence Threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # **🔥 Convert frame to JPEG format**
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
