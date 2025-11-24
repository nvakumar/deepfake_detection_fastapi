from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import gc  # Added for memory management

# Initialize App
app = FastAPI()

# Setup Templates 
# directory="." means it looks for index.html in the SAME folder as main.py
templates = Jinja2Templates(directory=".")

# -------------------------
# 1. Configuration
# -------------------------
MODEL_PATH = "deepfake_weights.pth"
CLASS_NAMES = ["FAKE", "REAL"]
DEVICE = torch.device("cpu") 

# -------------------------
# 2. Load Model (Optimized)
# -------------------------
def load_model():
    print("Loading model architecture...")
    try:
        # 1. Define Architecture
        model = models.convnext_base(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # 2. Load Weights
        print("Loading weights...")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        # Clear the state_dict from memory immediately
        del state_dict
        gc.collect()

        # ---------------------------------------------------------
        # OPTIMIZATION: DYNAMIC QUANTIZATION
        # ---------------------------------------------------------
        print("Applying Dynamic Quantization to reduce memory usage...")
        
        # This converts Linear layers from Float32 (heavy) to Int8 (light)
        # We skip Conv2d because standard dynamic quantization supports Linear/RNN/LSTM best on CPU
        model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear},  
            dtype=torch.qint8        
        )
        print("Quantization complete!")
        # ---------------------------------------------------------

        return model
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return None

# Load model globally
model = load_model()

# -------------------------
# 3. Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# 4. Routes
# -------------------------

# Serve the HTML Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle the Prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model failed to load. Check server logs."}

    try:
        # Read and Process Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item() * 100

        # Memory cleanup after prediction
        del img_tensor
        del image_bytes
        gc.collect()

        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}
