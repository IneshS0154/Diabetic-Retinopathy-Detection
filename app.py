import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Define your model architecture (must match training!)
class YourModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: replace with your actual model definition
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*224*3, 5)  # example: 5 classes
        )

    def forward(self, x):
        return self.backbone(x)

# Load model
model = YourModelClass()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Transforms (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess image
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return JSONResponse({"prediction": int(predicted.item())})
