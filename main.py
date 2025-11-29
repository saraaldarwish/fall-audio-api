import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from PIL import Image
import librosa

# ----------------- CONFIG -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256

# grayscale + resize + normalize (same as training)
img_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ----------------- MODEL DEF -----------------
class ConvNeXtWithMeta(nn.Module):
    def __init__(self, num_classes: int, meta_dim: int = 2):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = convnext_tiny(weights=weights)

        # (1) adapt first conv to 1 channel
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

        # (2) remove original classifier head
        in_feats = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
        self.backbone = backbone

        # (3) metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # (4) final classifier on [audio_feats || meta_feats]
        self.classifier = nn.Linear(in_feats + 16, num_classes)

    def forward(self, x, meta):
        feats = self.backbone(x)        # (B, in_feats)
        meta_feats = self.meta_mlp(meta)  # (B, 16)
        fused = torch.cat([feats, meta_feats], dim=1)
        logits = self.classifier(fused)
        return logits


# ----------------- LOAD BUNDLE -----------------
BUNDLE_PATH = "safe_fall_severity_bundle.pt"
bundle = torch.load(BUNDLE_PATH, map_location=DEVICE)

class_to_idx = bundle["class_to_idx"]
idx_to_class = bundle["idx_to_class"]
env_map = bundle["env_map"]
position_enc_map = bundle["position_enc_map"]
surface_categories = bundle["surface_categories"]

# build CLASSES list in the correct index order: 0,1,2,...
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


def meta_from_filename(path: str) -> torch.Tensor:
    """Extract CC from filename and map to encoded position/surface."""
    fname = os.path.basename(path).split(".")[0]  # 'AA-BBB-CC-DDD-FF'
    parts = fname.split("-")
    if len(parts) == 5:
        cc = parts[2]
    else:
        cc = "00"  # fallback

    env_info = env_map.get(cc, {"position": "Unknown", "surface": "Unknown"})
    position = env_info.get("position", "Unknown")
    surface = env_info.get("surface", "Unknown")

    pos_enc = position_enc_map.get(position, -1)

    try:
        surf_enc = surface_categories.index(surface)
    except ValueError:
        surf_enc = -1

    meta = torch.tensor([pos_enc, surf_enc], dtype=torch.float32).unsqueeze(0)
    return meta.to(DEVICE)


@torch.no_grad()
def run_inference(path: str):
    x = audio_to_tensor(path)
    meta = meta_from_filename(path)
    logits = model(x, meta)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    label = idx_to_class[idx]
    return label, {cls: float(p) for cls, p in zip(CLASSES, probs)}


# ----------------- FASTAPI APP -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save uploaded file to temp path
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        label, probs = run_inference(tmp_path)
    finally:
        os.remove(tmp_path)

    return {"label": label, "probs": probs}
