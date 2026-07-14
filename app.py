import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

import streamlit
import streamlit_webrtc


st.write("Streamlit:", streamlit.__version__)
st.write("streamlit-webrtc:", streamlit_webrtc.__version__)
st.title("Live Hand Gesture Recognition")

st.write("Place your hand inside the green box for prediction.")

gesture_to_label = {
    "01_palm":0,
    "02_l":1,
    "03_fist":2,
    "04_fist_moved":3,
    "05_thumb":4,
    "06_index":5,
    "07_ok":6,
    "08_palm_moved":7,
    "09_c":8,
    "10_down":9
}

label_to_gesture = {v:k for k,v in gesture_to_label.items()}

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)




class GestureCNN(nn.Module):

    def __init__(self):
        super(GestureCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.gap(x)

        x = self.classifier(x)

        return x
    
@st.cache_resource
def load_model():

    model = GestureCNN().to(device)

    model.load_state_dict(
        torch.load(
            "hand_gesture_recog_cnn.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model

model = load_model()

def predict(image):

    image = Image.fromarray(image)

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(output,dim=1)

        confidence,pred = torch.max(probs,1)

    return (
        label_to_gesture[pred.item()],
        confidence.item()*100
    )
img = np.zeros((300,300,3), dtype=np.uint8)

print(predict(img))

class VideoProcessor(VideoProcessorBase):

  def recv(self, frame):

    img = frame.to_ndarray(format="bgr24")

    img = cv2.flip(img, 1)

    x1, y1 = 150, 100
    x2, y2 = 450, 400

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    roi = img[y1:y2, x1:x2]

    try:

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        gesture = "TEST"
        confidence = 100

        cv2.putText(
            img,
            f"{gesture} ({confidence:.2f}%)",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    except Exception as e:

        cv2.putText(
            img,
            str(e),
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    
    key="gesture",
    video_processor_factory=VideoProcessor,

    rtc_configuration={
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    
    video_html_attrs={
    "style": {"width": "100%"},
    "autoPlay": True,
    "controls": False,
    }
)
