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
                "or run this app locally."
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

# === Upload Image only ===
st.markdown("## Upload an image for mask detection")
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_img:
    # Save to temp file for cv2 reading
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tfile.write(uploaded_img.read())
    tfile.flush()

    # Read image using OpenCV (local import)
    import cv2
    img = cv2.imread(tfile.name)

    # Ensure model loaded and Run YOLO model (ensure_model used inside detect)
    try:
        # try detection (detect_and_annotate will handle ensure_model)
        results_img = detect_and_annotate(img.copy())
    except Exception as e:
        st.error(f"Model error: {e}")
        raise

    # If YOLO ran via results, convert; otherwise detect_and_annotate returned original image
    # ensure result is RGB for display
    try:
        result_img_rgb = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
    except Exception:
        # fallback if conversion fails
        result_img_rgb = results_img

    # Display result
    st.markdown("### 🔍 Detection Result")
    st.image(result_img_rgb, caption="YOLO Detection", width=700)

    # Download option
    result_pil = Image.fromarray(result_img_rgb)
    img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    result_pil.save(img_bytes.name)
    st.download_button("📥 Download Result Image", open(img_bytes.name, "rb").read(), file_name="detected.jpg")
else:
    st.info("Upload an image to run detection. The model may be unavailable on this host; if so, deploy locally or on a VM with PyTorch support.")

