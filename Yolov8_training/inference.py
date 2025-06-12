import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("train25/weights/best.pt")

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)[0]

    # Filter predictions by confidence threshold > 0.7
    results.boxes = results.boxes[results.boxes.conf > 0.7]

    # Visualize the filtered results
    annotated_frame = results.plot()

    # Display the frame with bounding boxes
    cv2.imshow("YOLOv8 Camera", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
