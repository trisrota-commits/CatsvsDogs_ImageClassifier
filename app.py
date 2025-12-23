import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import load_model

# -----------------------
# App setup
# -----------------------
st.set_page_config(page_title="Cats vs Dogs Classifier")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model()

classes = ["Cat", "Dog"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# UI
# -----------------------
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    st.subheader(f"Prediction: {classes[prediction.item()]}")
    st.write(f"Confidence: {confidence.item() * 100:.2f}%")
