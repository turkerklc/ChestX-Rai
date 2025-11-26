import os
import sys
import torch
import cv2
import numpy as np
import random
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Grad-CAM kütüphanesi
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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
MODEL_PATH = PROJECT_ROOT / "saved_models" / "resnet50_epoch_5.pth"

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def load_model(device):
    model = XRayResNet50(num_classes=len(LABELS), pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def main():
    # Grad-CAM bazen MPS'de (Mac GPU) hata verebilir, görselleştirme için CPU en garantisidir.
    device = torch.device("cpu") 
    logger.info(f"🖥️  İşlem Cihazı: {device} (Görselleştirme için CPU önerilir)")

    # 1. Modeli Yükle
    model = load_model(device)
    
    # 2. Rastgele Resim Seç
    all_images = list(IMG_DIR.glob("*.png"))
    if not all_images:
        logger.error("❌ Resim bulunamadı.")
        return

    img_path = random.choice(all_images)
    logger.info(f"📸 Seçilen Resim: {img_path.name}")
    
    # 3. Resmi Hazırla
    # Grad-CAM kütüphanesi 0-1 aralığında float32 numpy array ister
    rgb_img = cv2.imread(str(img_path))[:, :, ::-1] # BGR -> RGB
    rgb_img = np.float32(rgb_img) / 255
    # Boyutlandır (Model 224x224 istiyor)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    
    # Tensor'a çevir
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(Image.fromarray((rgb_img * 255).astype(np.uint8))).unsqueeze(0).to(device)

    # 4. Hedef Katmanı Belirle
    # ResNet50'nin son konvolüsyon katmanı: model.backbone.layer4
    target_layers = [model.backbone.layer4[-1]]

    # 5. Grad-CAM Oluştur
    cam = GradCAM(model=model, target_layers=target_layers)

    # Hangi hastalık için harita çıkaralım?
    # None = Modelin en yüksek tahmin ettiği sınıfı otomatik seçer
    # ClassifierOutputTarget(8) = Zorla 'Infiltration' (8. sınıf) için bak
    targets = None 

    logger.info("🌡️  Isı haritası hesaplanıyor...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Batch boyutundan kurtul

    # 6. Görüntü ile Haritayı Birleştir
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 7. Kaydet
    save_name = "output_gradcam.jpg"
    cv2.imwrite(save_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    logger.info(f"✅ BAŞARILI! Sonuç kaydedildi: {save_name}")
    logger.info("👉 Bu dosyayı açarak modelin nereye odaklandığını görebilirsin.")
    
    # Eğer VS Code kullanıyorsan terminalden 'code output_gradcam.jpg' diyebilirsin.

if __name__ == "__main__":
    main()