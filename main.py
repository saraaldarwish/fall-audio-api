import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import librosa

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models import convnext_tiny
from torchvision import transforms
from PIL import Image

# ----------------- SETTINGS -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256

# same transforms as in your Kaggle notebook
img_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ----------------- MODEL DEFINITION -----------------
class ConvNeXtWithMeta(nn.Module):
    def __init__(self, num_classes: int, meta_dim: int = 2):
        super().__init__()
        # no pretrained weights here; we will load your trained state_dict
        backbone = convnext_tiny(weights=None)

        # adapt first conv to 1 channel (from 3)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        backbone.features[0][0] = new_conv

        # remove original classifier head
        in_feats = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
        self.backbone = backbone

        # metadata branch
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # final classifier
        self.classifier = nn.Linear(in_feats + 16, num_classes)

    def forward(self, x, meta):
        feats = self.backbone(x)
        meta_feats = self.meta_mlp(meta)
        fused = torch.cat([feats, meta_feats], dim=1)
        logits = self.classifier(fused)
        return logits


# ----------------- LOAD BUNDLE (SAME AS KAGGLE) -----------------
BUNDLE_PATH = "safe_fall_severity_bundle.pt"

bundle = torch.load(BUNDLE_PATH, map_location=DEVICE)

class_to_idx = bundle["class_to_idx"]
idx_to_class = bundle["idx_to_class"]

# class list in correct order
CLASSES = [idx_to_class[i] for i in range(len(idx_to_class))]

model = ConvNeXtWithMeta(num_classes=len(CLASSES), meta_dim=2).to(DEVICE)
model.load_state_dict(bundle["model_state_dict"])
model.eval()


# ----------------- PREPROCESSING -----------------
def audio_to_tensor(path: str) -> torch.Tensor:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    img = Image.fromarray(S_db).convert("L")
    x = img_tf(img).unsqueeze(0)  # [1, 1, 224, 224]
    return x.to(DEVICE)


def meta_dummy() -> torch.Tensor:
    """
    For now we ignore environment metadata and just use zeros.
    This matches how you computed Local_probs in Kaggle.
    """
    meta = torch.zeros((1, 2), dtype=torch.float32)
    return meta.to(DEVICE)


@torch.no_grad()
def run_inference(path: str):
    x = audio_to_tensor(path)
    meta = meta_dummy()
    logits = model(x, meta)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    label = idx_to_class[idx]
    return label, {cls: float(p) for cls, p in zip(CLASSES, probs)}


# ----------------- FASTAPI APP -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        label, probs = run_inference(tmp_path)
    finally:
        os.remove(tmp_path)

    return {"label": label, "probs": probs}
