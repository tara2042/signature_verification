
import cv2
import pytesseract
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io

class SignatureEmbedder(nn.Module):
    def __init__(self):
        super(SignatureEmbedder, self).__init__()

        # Load pretrained ResNet with proper weights
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()  # Remove final classification layer

    def forward(self, x):
        return self.model(x)
    
def preprocess_signature(image_bytes):
    # Read image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor


# use the following statements to run the app
# python d:\signature_kyc\signature\run\app.py
# uvicorn run.app:app # <--reload>