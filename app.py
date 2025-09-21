import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

model = models.efficientnet_b3(pretrained=False)
num_classes = 5  
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize(300),           
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0)  #

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return JSONResponse({"prediction": int(predicted.item())})
