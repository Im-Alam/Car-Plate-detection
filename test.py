import cv2
import sys
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Set up video source. Multiple camera options are available then we selected first channel as video source.
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

vid_source = cv2.VideoCapture(s)

# Initialize GOTURN tracker.It scans current working directory for goturn.caffemodel an goturn.prototxt. 
# and use them to initialize the tracker.
tracker = cv2.TrackerGOTURN.create()

# Load YOLO model from YOLO
model = YOLO('license_plate_detector_model.pt')

# Set up window
window_name = 'Video Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Read first frame
has_frame, frame = vid_source.read()
if not has_frame:
    print("Error: Could not read video")
    sys.exit(1)

# Run YOLO on the first frame to detect license plates
results = model(frame)
results.result.box
for result in results:
for box in result.boxes:
    # Get the coordinates of the bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # Get the confidence score of the prediction
    confidence = box.conf[0]

    # Draw the bounding box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw the confidence score near the bounding box
    cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

bbox = results.bbox()

# Extract the first detected bounding box
# for result in results:
#     for box in result.boxes:
#         bbox = box.xyxy[0].cpu().numpy()  # Convert tensor to NumPy
#         bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
#         break  # Use only the first detected license plate

if bbox is None:
    print("No license plate detected, using manual selection.")
    bbox = cv2.selectROI("Select License Plate", frame, False)
    cv2.destroyWindow("Select License Plate")

# Initialize tracker with first frame and detected bbox
ok = tracker.init(frame, bbox)

# Loop for tracking
while cv2.waitKey(1) != 27:  # Exit on ESC key
    has_frame, frame = vid_source.read()
    if not has_frame:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display output
    cv2.imshow(window_name, frame)

# Cleanup
vid_source.release()
cv2.destroyAllWindows()
