import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms
import os
import gdown

st.set_page_config(page_title="Live Hand Gesture Recognition")

st.title("Hand Gesture Recognition")
st.write("Take a picture of your hand gesture.")

# -----------------------------
# Labels
# -----------------------------

gesture_to_label = {
    "01_palm": 0,
    "02_l": 1,
    "03_fist": 2,
    "04_fist_moved": 3,
    "05_thumb": 4,
    "06_index": 5,
    "07_ok": 6,
    "08_palm_moved": 7,
    "09_c": 8,
    "10_down": 9
}

label_to_gesture = {v: k for k, v in gesture_to_label.items()}

# -----------------------------
# Transform
# -----------------------------

transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))

])

# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# -----------------------------
# CNN
# -----------------------------

class GestureCNN(nn.Module):

    def __init__(self):
        super(GestureCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x
        

# -----------------------------
# Load model
# -----------------------------

MODEL_PATH = "hand_gesture_cnn.pth"

@st.cache_resource
def load_model():

    # Agar model file present nahi hai to Google Drive se download karo
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1WVV-wdmi2bsGsm1q8t2svx06S-Piqx6M"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = GestureCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model


model = load_model()

# -----------------------------
# Prediction
# -----------------------------

def predict(image):

    image = Image.fromarray(image)

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

    return (
        label_to_gesture[pred.item()],
        confidence.item()*100
    )
uploaded_image = st.camera_input("Capture Hand Gesture")

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.write("Image Size:", image.size)
    
    st.image(
        image,
        caption="Captured Image",
        use_container_width=True
    )

    width, height = image.size

    crop_size = min(width, height)
    
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    image = image.crop((left, top, right, bottom))
    
    st.image(
        image,
        caption="Center Cropped Image",
        use_container_width=True
    )
    gesture, confidence = predict(np.array(image))

    st.success(f"Prediction : {gesture}")

    st.info(f"Confidence : {confidence:.2f}%")
