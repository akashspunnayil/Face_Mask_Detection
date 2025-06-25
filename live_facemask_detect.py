
#-- Live Face-Mask Detection - Using YOLO pretrained model --#

import cv2
from ultralytics import YOLO
import pygame
import time

# Load custom face mask detection YOLO model
model = YOLO("./best.pt")
names = model.names  # Should include 'mask' and 'no_mask'



#-- Detection --#

# Access default live-cam (0)
cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("mask_video2.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_filename = f"recordings/recorded_{time.strftime('%Y%m%d_%H%M%S')}.avi"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_filename, fourcc, 20.0, (frame_width, frame_height))


img_counter = 0
print("Live cam started. Press 's' to save image, 'q' to quit.")

# Optional sound setup
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("chime_alert.mp3")

alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Live cam frame not received")
        break

    # -----------------------------
    # Draw Line, Rectangle, Circle, Text
    # -----------------------------
    height, width = frame.shape[:2]

    results = model(frame)

    # 🔹 Extract results
    boxes = results[0].boxes
    names = model.names  # COCO class names

    # Uses OpenCV's haarcascade
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    
    detected_classes = [int(box.cls[0].item()) for box in boxes]
    labels = [model.names[cls_id] for cls_id in detected_classes]

    # Draw detection boxes and play alert for "without_mask"
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        label_text = f"{label}: {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)
        print(f"🟢 Detected: {label_text} at ({x1}, {y1}, {x2}, {y2})")

        # Play sound for each "without_mask" detected
        if "without_mask" in label.lower():
            alert_sound.play()

        
    # -----------------------------
    # Save frame to video file
    # -----------------------------
    out.write(frame)

    # -----------------------------
    # Show live-cam feed
    # -----------------------------
    cv2.imshow('Live-cam Feed with Drawing', frame)

    # -----------------------------
    # Keypress handling
    # -----------------------------
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Exiting live-cam...")
        break
    elif key & 0xFF == ord('s'):
        # Save snapshot image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(img_filename, frame)
        print(f"Snapshot saved: {img_filename}")
        img_counter += 1

# -----------------------------
# Release resources
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Recorded video saved as: {output_video_filename}")
