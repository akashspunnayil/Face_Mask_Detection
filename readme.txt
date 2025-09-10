
# create a new environment with Python 3.10
conda create -n yolo310 python=3.10 -y
conda activate yolo310

pip install --upgrade pip

# Reinstall your packages

pip install streamlit==1.28.2 opencv-python-headless==4.11.0.86 pillow==9.5.0 numpy==1.26.4


pip install streamlit ultralytics opencv-python-headless numpy pillow
pip install torch torchvision




MASK/
├── images/               ← All images
├── annotations/          ← All XML files
├── labels/               ← YOLO-format .txt files (generated)
├── datasets/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
├── data.yaml             ← Configuration file for training


Aravind IPCS
4:53 PM
runs/detect/train/
├── weights/
│   ├── best.pt     ← Most accurate model
│   ├── last.pt     ← Model at final epoch
├── results.png     ← Training performance graph
├── confusion_matrix.png


