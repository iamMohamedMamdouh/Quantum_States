import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class_names = ["entangled", "mixed", "pure"]

class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (128 // 8) * (128 // 8), 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


@st.cache_resource
def load_model():
    model = QuantumClassifier()
    model.load_state_dict(torch.load("quantum_bloch_cnn.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

st.title("Quantum State Classifier")
st.markdown("Upload a Bloch Sphere image and we will classify its state (Pure / Mixed / Entangled)")

uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy().squeeze()
        pred_class = np.argmax(probs)

    st.markdown(f"###  Prediction:  **{class_names[pred_class].capitalize()}**")
    st.bar_chart(probs)
