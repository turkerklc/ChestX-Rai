import os
import sys
import torch
import cv2
import numpy as np
import random
import pandas as pd
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- ORTAM AYARLARI ---
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

try:
    from model.model import XRayResNet50
except ImportError:
    sys.exit("❌ Model dosyası bulunamadı.")

# Loglama
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --- AYARLAR ---
PROJECT_ROOT = CURRENT_DIR.parent.parent
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "images"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data_Entry_2017.csv"
MODEL_PATH = PROJECT_ROOT / "saved_models" / "resnet50_epoch_5.pth" # Son epoch

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def get_ground_truth(img_name):
    """Gerçek tanıyı CSV'den bulur."""
    if not CSV_PATH.exists(): return "CSV Yok"
    try:
        df = pd.read_csv(CSV_PATH)
        row = df[df['Image Index'] == img_name]
        return row.iloc[0]['Finding Labels'] if not row.empty else "Bilinmiyor"
    except: return "Hata"

def main():
    # Görselleştirme için CPU daha stabildir
    device = torch.device("cpu")
    logger.info(f"🚀 Demo Başlatılıyor... (Cihaz: {device})")

    # 1. Modeli Yükle
    model = XRayResNet50(num_classes=len(LABELS), pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # 2. Rastgele Resim Seç
    all_images = list(IMG_DIR.glob("*.png"))
    if not all_images: return

    img_path = random.choice(all_images)
    img_name = img_path.name
    true_label = get_ground_truth(img_name)

    logger.info(f"\n📸 Resim: {img_name}")
    logger.info(f"🩺 GERÇEK: {true_label}")

    # 3. Hazırlık (Preprocessing)
    # Grad-CAM için (0-1 arası float numpy)
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_float = np.float32(img_rgb) / 255
    img_rgb_float = cv2.resize(img_rgb_float, (224, 224))
    
    # Model için (Tensor)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(Image.fromarray((img_rgb_float * 255).astype(np.uint8))).unsqueeze(0).to(device)

    # 4. TAHMİN (Prediction)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze()
    
    # Sonuçları Sırala
    prob_list = probs.cpu().numpy()
    results = sorted(zip(LABELS, prob_list), key=lambda x: x[1], reverse=True)
    
    top_prediction = results[0] # En yüksek tahmin (İsim, Oran)
    
    print("\n--- 🤖 YAPAY ZEKA KARARI ---")
    for i, (label, prob) in enumerate(results[:3]):
        icon = "🚨" if prob > 0.5 else "🔸"
        print(f"{icon} {label:<20} : {prob:.1%}")

    # 5. AÇIKLAMA (Grad-CAM)
    target_layers = [model.backbone.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # En yüksek tahmin edilen sınıf için harita çıkar
    targets = None # Otomatik olarak en yüksek sınıfı hedefler
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Haritayı resimle birleştir
    cam_image = show_cam_on_image(img_rgb_float, grayscale_cam, use_rgb=True)
    
    # 6. Yan Yana Kaydet (Orijinal vs Grad-CAM)
    # Orijinal resmi de 224x224 yapalım
    orig_img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Yan yana birleştir
    combined = np.hstack((orig_img_resized, cam_image))
    
    # Üzerine Yazı Yaz (OpenCV ile)
    # Metin: "Gerçek: ... | Tahmin: ... (%...)"
    final_img_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    text = f"Pred: {top_prediction[0]} ({top_prediction[1]:.1%})"
    cv2.putText(final_img_bgr, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    save_name = "demo_result.jpg"
    cv2.imwrite(save_name, final_img_bgr)
    
    logger.info(f"\n✅ SONUÇ KAYDEDİLDİ: {save_name}")
    logger.info("👉 Bu resimde solda orijinali, sağda modelin odaklandığı yeri görebilirsin.")

if __name__ == "__main__":
    main()