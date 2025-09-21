# Diabetic Retinopathy Detection API

This is a FastAPI service that loads a trained PyTorch model (`dr_model.pth`) 
to predict diabetic retinopathy stage from an uploaded fundus image.

## Usage
- Endpoint: `/predict/`
- Method: `POST`
- Body: multipart form-data with field `file` (image)

Example with `curl`:
```bash
curl -X POST -F "file=@test.jpg" https://username-dr-docker-space.hf.space/predict/
    