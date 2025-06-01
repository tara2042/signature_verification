# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from run.model import SignatureEmbedder, preprocess_signature
from torch.nn.functional import cosine_similarity
import torch

# Creating FAST API
app = FastAPI()
model = SignatureEmbedder()
model.eval()

@app.get("/")
def root():
    return {"message": "Signature verification API is running"}

@app.post("/verify-signature/")
async def verify_signature(reference: UploadFile = File(...), input: UploadFile = File(...)):
    try:
        # Read and preprocess both images
        ref_bytes = await reference.read()
        inp_bytes = await input.read()

        img1 = preprocess_signature(ref_bytes)
        img2 = preprocess_signature(inp_bytes)

        # Generate embeddings
        with torch.no_grad():
            emb1 = model(img1)
            emb2 = model(img2)

        # Cosine similarity
        similarity_score = cosine_similarity(emb1, emb2).item()

        # Threshold decision
        is_match = similarity_score > 0.85

        return JSONResponse({
            "similarity_score": round(similarity_score, 4),
            "match": is_match
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
