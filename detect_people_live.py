import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = sv.ByteTrack()

# Hardcoded for now, but could be 0, 1, or 2 depending on Apple Continuity Camera setup
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed detected. Try changing the index in VideoCapture.")
        break

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # TODO: Enum for person class = 0, ideally something that already exists in supervision
    # but there seems to be no built in enum, you just really put the class_id manually based on the docs.
    person_detections = detections[detections.class_id == 0]
    tracked = tracker.update_with_detections(person_detections)
    annotated = box_annotator.annotate(scene=frame.copy(), detections=tracked)
    annotated = label_annotator.annotate(scene=annotated, detections=tracked)

    cv2.imshow("iPhone Camera Feed (Camo)", annotated)
    # Wait for q, use ascii + & 0xFF for compatibility
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
