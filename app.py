import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os

# =====================================================================
# 1. MODEL ARCHITECTURE DEFINITIONS
# =====================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        for feat in features:
            self.encoders.append(DoubleConv(in_ch, feat))
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = feat
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feat*2, feat, 2, 2))
            self.decoders.append(DoubleConv(feat*2, feat))
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, skips[::-1]):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final(x)

# =====================================================================
# 2. MODEL LOADING
# =====================================================================
@st.cache_resource
def load_models():
    device = torch.device('cpu')
    unet_path = 'best_leukemia_unet.pth'
    clf_path = 'best_classifier.pth'
    
    if not os.path.exists(unet_path) or not os.path.exists(clf_path):
        st.error("Weights file check karo! Same folder me honi chahiye jahan app.py hai.")
        st.stop()
        
    unet_model = UNet(in_ch=3, out_ch=1)
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    unet_model.eval()
    
    resnet_model = models.resnet50(weights=None)
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    resnet_model.load_state_dict(torch.load(clf_path, map_location=device))
    resnet_model.eval()
    
    return unet_model, resnet_model

unet_model, resnet_model = load_models()

# =====================================================================
# 3. TRANSFORMS & UTILITIES
# =====================================================================
seg_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = {0: 'MALIGNANT', 1: 'NORMAL'}

# =====================================================================
# 4. LOCAL STREAMLIT DASHBOARD
# =====================================================================
st.set_page_config(page_title="Local Leukemia Detection AI", layout="wide")
st.title("🔬 Offline Medical AI Dashboard: Leukemia Detection")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a blood cell microscopy image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    with st.spinner("Local pipeline is processing the image..."):
        seg_input = seg_transform(image).unsqueeze(0)
        with torch.no_grad():
            seg_output = unet_model(seg_input)
            pred_mask = torch.sigmoid(seg_output).squeeze().numpy()
            binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        clf_input = clf_transform(image).unsqueeze(0)
        with torch.no_grad():
            clf_output = resnet_model(clf_input)
            probabilities = torch.softmax(clf_output, dim=1).squeeze().numpy()
            pred_class_idx = np.argmax(probabilities)
            diagnosis = class_names[pred_class_idx]
            confidence = probabilities[pred_class_idx] * 100

        mask_resized = cv2.resize(binary_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay_img = img_np.copy()
        color = [255, 0, 0] if pred_class_idx == 0 else [0, 255, 0]
        overlay_img[mask_resized == 255] = overlay_img[mask_resized == 255] * 0.4 + np.array(color) * 0.6
        overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("U-Net Mask")
        st.image(binary_mask, use_container_width=True, channels="GRAY")
    with col3:
        st.subheader("Diagnostic Overlay")
        st.image(overlay_img, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Diagnostic Analysis Report")
    if diagnosis == "MALIGNANT":
        st.error(f"## Diagnosis: {diagnosis} (Confidence: {confidence:.2f}%)")
    else:
        st.success(f"## Diagnosis: {diagnosis} (Confidence: {confidence:.2f}%)")