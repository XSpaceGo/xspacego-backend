from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import uuid
import cv2
import numpy as np
import torch

from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

# ================= APP ================= #

app = FastAPI(title="XSpaceGo AI Image Enhancement")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= PATHS ================= #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "realesr-general-x4v3.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_DIM = 2048

# ================= DEVICE ================= #

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ================= MODEL (LAZY LOAD) ================= #

sr_model = None

def get_model():
    global sr_model
    if sr_model is None:
        print("🔄 Loading Real-ESRGAN General x4 v3 (SRVGGNetCompact)...")

        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )

        sr_model = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=DEVICE,
        )

        print("✅ Model loaded successfully")

    return sr_model

# ================= HELPERS ================= #

def safe_resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(MAX_DIM / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# ================= ROUTES ================= #

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "Real-ESRGAN General x4 v3 (SRVGGNetCompact)",
        "device": DEVICE,
    }

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise ValueError("Empty file")

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format (use JPG or PNG)")

        img = safe_resize(img)

        sr = get_model()
        output, _ = sr.enhance(img, outscale=4)

        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, output)

        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        print("❌ Enhancement error:", repr(e))
        raise HTTPException(status_code=400, detail=str(e))
