import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
import json
import io

# --- Model Definition (Must be identical to the one used for training) ---
# We define the class here so PyTorch knows how to load the model weights.

class LitCNNModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # --- Model Architecture ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Flattened size: 64 * 54 * 54 = 186624
        self.fc1 = nn.Linear(64 * 54 * 54, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # Defines the forward pass
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Caching and Helper Functions ---

MODEL_PATH = 'plant_disease_prediction_model.pth'
CLASS_INDEX_PATH = 'class_indices.json'
NUM_CLASSES = 38 # From our notebook

@st.cache_resource
def load_model(model_path, num_classes):
    """
    Loads the trained PyTorch Lightning model from a .pth file.
    We use @st.cache_resource to load this only once.
    """
    model = LitCNNModel(num_classes=num_classes)
    # Load the state_dict (weights)
    # map_location='cpu' ensures it works on machines without a GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode (important!)
    return model

@st.cache_data
def load_class_names(class_json_path):
    """
    Loads the class index mapping from the JSON file.
    We use @st.cache_data for this since it's simple data.
    """
    try:
        with open(class_json_path, 'r') as f:
            class_indices_map = json.load(f)
        # Invert the map to go from index -> class name
        # e.g., {"Apple___Apple_scab": 0} -> {0: "Apple___Apple_scab"}
        class_names = {v: k for k, v in class_indices_map.items()}
        return class_names
    except FileNotFoundError:
        st.error(f"Error: {class_json_path} not found. Please make sure it's in the same directory as app.py.")
        return None

def predict_image_class(model, image_bytes, class_names):
    """
    Runs a prediction on a single image (provided as bytes).
    """
    
    # 1. Define the transformations (must be same as training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load and preprocess the image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension [1, 3, 224, 224]

    # 3. Make prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)
    
    predicted_class_index = predicted_index.item()
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = confidence.item() * 100
    
    return predicted_class_name, confidence_score

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Plant Disease Detector")

st.title("🌱 Plant Disease Detector")
st.write("Upload an image of a plant leaf, and the model will predict its condition.")

# Load model and class names
model = load_model(MODEL_PATH, NUM_CLASSES)
class_names = load_class_names(CLASS_INDEX_PATH)

if class_names:
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image bytes
            image_bytes = uploaded_file.getvalue()
            
            # Display the image
            st.image(image_bytes, caption='Uploaded Image.', use_column_width=True)
            
            # Add a button to trigger prediction
            if st.button('Classify'):
                with st.spinner('Analyzing the leaf...'):
                    # Make prediction
                    prediction, confidence = predict_image_class(model, image_bytes, class_names)
                    
                    # Format the prediction for readability
                    clean_prediction = prediction.replace("___", " - ").replace("_", " ")
                    
                    with col2:
                        st.success(f"**Prediction:** {clean_prediction}")
                        st.info(f"**Confidence:** {confidence:.2f}%")
                        
                        if "healthy" in clean_prediction:
                            st.balloons()
                        
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Convolutional Neural Network (CNN) trained with PyTorch Lightning "
    "on the PlantVillage dataset to identify 38 different plant disease classes."
)