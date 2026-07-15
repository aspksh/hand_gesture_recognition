import streamlit as st

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms

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
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
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
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256,10)
        )

    def forward(self,x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)

        x = self.classifier(x)

        return x

# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    model = GestureCNN()

    model.load_state_dict(
        torch.load(
            "hand_gesture_recog_cnn.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
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

    gesture, confidence = predict(np.array(image))

    st.success(f"Prediction : {gesture}")

    st.info(f"Confidence : {confidence:.2f}%")
