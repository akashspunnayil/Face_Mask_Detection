# Face Mask Detection (YOLO-based Streamlit App)

This app performs face mask detection using a YOLOv8 model trained to detect masks. It supports image, video, and webcam inputs via a Streamlit interface.

## Objective

To identify whether individuals are wearing masks using a pre-trained YOLO model.

## Features

- Upload and detect from:
  - Static image
  - Video file
  - Webcam snapshot
- Annotated result preview
- Downloadable outputs

## Workflow

- Load image/video from user
- Run YOLOv8 model inference
- Display and allow download of annotated output

## Dependencies

- `streamlit`
- `opencv-python`
- `ultralytics`
- `PIL`, `numpy`, `tempfile`

## Output

- Mask detection with bounding boxes
- Downloadable result image or video

