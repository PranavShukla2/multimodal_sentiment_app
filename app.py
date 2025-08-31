import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer

# Import your model class and config
from model import MultimodalSentimentClassifier
import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal Sentiment Classifier",
    page_icon="🤖",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main App Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f0f2f6;
    }
    h1 {
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #1C357A;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .stFileUploader>div>div>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the tokenizer and model from disk."""
    device = config.DEVICE
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MultimodalSentimentClassifier(num_classes=config.NUM_CLASSES)
    
    # Load the trained model weights.
    # IMPORTANT: Make sure 'model_weights.pth' is in your project directory.
    try:
        model.load_state_dict(torch.load(config.MODEL_WEIGHTS, map_location=device))
    except FileNotFoundError:
        st.error("Model weights file ('model_weights.pth') not found. Please ensure it is in the root directory.")
        return None, None
        
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Image Transformations ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Main App Interface ---
st.title("Multimodal Sentiment Classifier")
st.markdown("<p style='text-align: center;'>Analyze sentiment from text, images, or both!</p>", unsafe_allow_html=True)

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
with col2:
    text_input = st.text_area("Or Enter Text", height=150)

# --- Prediction Logic ---
if st.button("Predict Sentiment"):
    if model is None:
        st.stop() # Stop execution if model loading failed

    if not uploaded_image and not text_input.strip():
        st.warning("Please upload an image or enter some text for prediction.")
    else:
        with st.spinner('Analyzing...'):
            device = config.DEVICE
            label_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

            # Process image if available
            image_tensor = None
            if uploaded_image is not None:
                image = Image.open(uploaded_image).convert("RGB")
                image_tensor = image_transform(image).unsqueeze(0).to(device)

            # Process text if available
            input_ids = None
            attention_mask = None
            if text_input.strip():
                encoding = tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

            # Perform prediction
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
                pred_idx = torch.argmax(outputs, dim=1).item()
                pred_label = label_map.get(pred_idx, "Unknown")
            
            # Display result
            st.success(f"**Predicted Sentiment:** {pred_label}")

            if uploaded_image:
                st.image(image, caption='Uploaded Image.', use_column_width=True)

