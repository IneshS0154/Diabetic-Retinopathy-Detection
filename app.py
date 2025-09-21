import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- Define the model architecture ----
model = models.efficientnet_b3(pretrained=False)
num_classes = 5
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ---- Image preprocessing ----
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- DR Stage Mapping ----
DR_STAGES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# ---- FastAPI app ----
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Diabetic Retinopathy Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess image
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension

    # Model inference
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # Map numeric class to DR stage
    stage = DR_STAGES[int(predicted.item())]

    return JSONResponse({
        "prediction": int(predicted.item()),
        "stage": stage
    })
