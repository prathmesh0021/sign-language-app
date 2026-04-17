import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from utils.label_map import LABELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL =================
class SignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = Swin3D_T_Weights.DEFAULT
        self.backbone = swin3d_t(weights=weights)
        self.backbone.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ================= LOAD MODEL =================
MODEL_PATH = "model/best_model.pth"
NUM_CLASSES = 226

def load_model():
    model = SignModel(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ================= PREPROCESS =================
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def preprocess_video(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, max_frames).astype(int)

    frames = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frame = (frame - MEAN) / STD

        frames.append(frame)

    cap.release()

    frames = np.array(frames)
    frames = torch.tensor(frames).permute(3,0,1,2).float()

    return frames.unsqueeze(0)

# ================= PREDICT =================
def predict(video_path):
    input_tensor = preprocess_video(video_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

    top5_prob, top5_idx = torch.topk(probs, 5)

    top5_prob = top5_prob.cpu().numpy()[0]
    top5_idx = top5_idx.cpu().numpy()[0]

    prediction = LABELS[int(top5_idx[0])]
    confidence = float(top5_prob[0]) * 100

    top5 = [LABELS[int(i)] for i in top5_idx]

    return prediction, confidence, top5