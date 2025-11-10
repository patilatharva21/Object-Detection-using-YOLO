#print("hello")
import cv2
from ultralytics import YOLO

# Load your YOLOv8 model (replace 'your_model.pt' with your actual model file)
#model = YOLO('best7.pt')
model = YOLO('cocov8.pt')

# Initialize webcam (0 is the default camera, change if you use an external cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop to capture video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the current frame
    results = model(frame)

    # Draw bounding boxes and labels on the detected objects
    annotated_frame = results[0].plot()

    # Show the frame with annotations
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
