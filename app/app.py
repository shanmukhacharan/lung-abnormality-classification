"""
Minimal Streamlit demo for the fine-tuned CNN.

Run locally with:
    streamlit run app/app.py

Requires a trained model checkpoint at MODEL_SAVE_PATH (see src/config.py).
Train one first with `python main.py`, or point MODEL_PATH below at your
own checkpoint. This app is a demonstration of the inference pipeline,
not a clinical tool — see the disclaimer at the bottom.
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config, models  # noqa: E402
from src.data import build_transform  # noqa: E402

MODEL_PATH = config.MODEL_SAVE_PATH


@st.cache_resource
def load_model():
    model = models.build_finetuned_cnn(num_classes=len(config.CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


st.title("Lung Abnormality Classification — Demo")
st.caption("Fine-tuned ResNet50 trained to classify chest X-rays into COVID-19, Lung Opacity, Normal, or Viral Pneumonia.")

uploaded_file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # grayscale, matches training pipeline
    st.image(image, caption="Uploaded X-ray", width=300)

    if not Path(MODEL_PATH).exists():
        st.error(
            f"No trained model checkpoint found at '{MODEL_PATH}'. "
            "Train the fine-tuned CNN first with `python main.py`, which saves a checkpoint there."
        )
    else:
        transform = build_transform()
        input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

        model = load_model()
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

        pred_idx = probs.argmax()
        pred_class = config.CLASSES[pred_idx]
        confidence = probs[pred_idx] * 100

        st.success(f"Prediction: **{pred_class}** ({confidence:.1f}% confidence)")

        st.subheader("Class probabilities")
        for cls, prob in zip(config.CLASSES, probs):
            st.write(f"{cls}: {prob * 100:.1f}%")
            st.progress(float(prob))

st.markdown("---")
st.caption(
    "This project is intended for research and educational purposes only and is "
    "not a clinically validated diagnostic tool. Predictions should not be used "
    "to inform real medical decisions."
)
