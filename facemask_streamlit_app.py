import streamlit as st
import numpy as np
import tempfile
import os
import time
from PIL import Image

st.set_page_config(layout="wide")
st.title("😷 Face Mask Detection App (Pre-trained YOLO-based)")

# --- Lazy model loader (cached resource) ---
@st.cache_resource
def _get_model(model_path: str = "./best.pt"):
    # lazy import inside cached function
    from ultralytics import YOLO
    return YOLO(model_path)

# Global holders (start unloaded)
model = None
CLASS_NAMES = None

def ensure_model():
    """
    Ensure the global `model` and `CLASS_NAMES` are loaded.
    Raises the underlying exception if loading fails after showing an error in Streamlit.
    """
    global model, CLASS_NAMES
    if model is None:
        try:
            with st.spinner("Loading YOLO model..."):
                model = _get_model()
                CLASS_NAMES = model.names
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            raise

# Detection Function (uses local cv2 import)
def detect_and_annotate(image_np):
    import cv2  # local import for headless/server safety
    # ensure model is available
    ensure_model()

    results = model(image_np)
    boxes = results[0].boxes
    detected_classes = [int(box.cls[0].item()) for box in boxes]
    labels = [model.names[cls_id] for cls_id in detected_classes]

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_np, f"{label}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image_np

# --- Sidebar Options ---
option = st.sidebar.radio("Choose Input Mode", ["Upload Image", "Upload Video", "Webcam Snapshot"])

# === Upload Image ===
if option == "Upload Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        # Save to temp file for cv2 reading
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tfile.write(uploaded_img.read())
        tfile.flush()

        # Read image using OpenCV (local import)
        import cv2
        img = cv2.imread(tfile.name)

        # Ensure model loaded and Run YOLO model
        try:
            ensure_model()
            results = model(img)
        except Exception as e:
            st.error(f"Model error: {e}")
            raise

        # Get annotated image from YOLO
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Display using matplotlib-like behavior
        st.markdown("### 🔍 Detection Result")
        st.image(result_img_rgb, caption="YOLO Detection", width=700)

        # Download option
        result_pil = Image.fromarray(result_img_rgb)
        img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        result_pil.save(img_bytes.name)
        st.download_button("📥 Download Result Image", open(img_bytes.name, "rb").read(), file_name="detected.jpg")


elif option == "Upload Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        # Slider to let user define how many seconds to process
        st.markdown("⏱️ Choose how long to process")
        max_duration = st.slider(
            "Process only first N seconds",
            min_value=0.1,
            max_value=180.0,
            value=10.0,
            step=0.05,
            format="%.1f"
        )

        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())

        import cv2
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(fps * max_duration)

        width, height = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = f"detected_{int(time.time())}.avi"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        frame_count = 0

        # Ensure model loaded before processing loop
        try:
            ensure_model()
        except Exception:
            st.error("Could not load model; aborting video processing.")
            cap.release()
            out.release()
            raise

        with st.spinner(f"Processing first {max_duration} seconds..."):
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = detect_and_annotate(frame)
                out.write(annotated)
                stframe.image(annotated, channels="BGR", width=700)
                frame_count += 1

        cap.release()
        out.release()

        st.success(f"✅ Processed first {frame_count} frames ({frame_count / fps:.1f} sec)!")
        with open(output_path, 'rb') as f:
            st.download_button("📥 Download Partial Video", f, file_name="partial_detected.avi")

elif option == "Webcam Snapshot":
    st.info("📷 Use below button to capture a webcam image (experimental)")
    picture = st.camera_input("Capture image")

    if picture is not None:
        try:
            image = Image.open(picture).convert("RGB")
            img_np = np.array(image)

            st.image(img_np, caption="Captured Image", width=700)

            # Ensure model loaded and run detection
            try:
                ensure_model()
            except Exception:
                st.error("Model not available on this host.")
                raise

            result_img = detect_and_annotate(img_np.copy())
            st.image(result_img, caption="Detected", width=700)

            result_pil = Image.fromarray(result_img)
            img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            result_pil.save(img_bytes.name)
            st.download_button("📥 Download Result Image", open(img_bytes.name, "rb").read(), file_name="detected.jpg")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

