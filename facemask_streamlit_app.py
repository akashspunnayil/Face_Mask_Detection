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
    If loading fails, show a friendly error and return False.
    """
    global model, CLASS_NAMES
    if model is None:
        try:
            with st.spinner("Loading YOLO model..."):
                model = _get_model()
                CLASS_NAMES = model.names
        except Exception as e:
            st.error(
                "YOLO model is not available in this deployment environment.\n\n"
                "Reason: " + str(e) + "\n\n"
                "To run detection please deploy on a VM/Docker host that supports PyTorch, "
                "or run this app locally with the required packages installed."
            )
            return False
    return True

# Detection Function (uses local cv2 import)
def detect_and_annotate(image_np):
    import cv2  # local import for headless/server safety
    # ensure model is available
    if not ensure_model():
        # return the original image if model not available
        return image_np

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

        # Run YOLO model (detect_and_annotate handles ensure_model)
        try:
            result_img = detect_and_annotate(img.copy())
        except Exception as e:
            st.error(f"Model error: {e}")
            raise

        # Convert to RGB for display (detect_and_annotate returns BGR)
        try:
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        except Exception:
            result_img_rgb = result_img

        # Display using matplotlib-like behavior
        st.markdown("### 🔍 Detection Result")
        st.image(result_img_rgb, caption="YOLO Detection", width=700)

        # Download option
        result_pil = Image.fromarray(result_img_rgb)
        img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        result_pil.save(img_bytes.name)
        st.download_button("📥 Download Result Image", open(img_bytes.name, "rb").read(), file_name="detected.jpg")


# === Upload Video ===
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
        if not ensure_model():
            st.error("Could not load model; aborting video processing.")
            cap.release()
            out.release()
        else:
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


# === Webcam Snapshot ===
elif option == "Webcam Snapshot":
    st.info("📷 Use below button to capture a webcam image (experimental)")
    picture = st.camera_input("Capture image")

    if picture is not None:
        try:
            image = Image.open(picture).convert("RGB")
            img_np = np.array(image)

            st.image(img_np, caption="Captured Image", width=700)

            # Run detection (detect_and_annotate handles ensure_model)
            result_img = detect_and_annotate(img_np.copy())
            # Convert for display
            try:
                import cv2
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            except Exception:
                result_rgb = result_img

            st.image(result_rgb, caption="Detected", width=700)

            result_pil = Image.fromarray(result_rgb)
            img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            result_pil.save(img_bytes.name)
            st.download_button("📥 Download Result Image", open(img_bytes.name, "rb").read(), file_name="detected.jpg")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

