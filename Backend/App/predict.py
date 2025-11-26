import os
import sys
import torch
import random
import logging
import pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

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
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data_Entry_2017.csv" # Gerçek tanı için
MODEL_PATH = PROJECT_ROOT / "saved_models" / "resnet50_epoch_5.pth"

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def get_ground_truth(img_name):
    """CSV dosyasından gerçek tanıyı bulur."""
    if not CSV_PATH.exists():
        return "CSV Bulunamadı"
    
    try:
        df = pd.read_csv(CSV_PATH)
        # Resim adına göre satırı bul
        row = df[df['Image Index'] == img_name]
        if not row.empty:
            return row.iloc[0]['Finding Labels']
        return "Bilinmiyor"
    except Exception:
        return "Okuma Hatası"

def load_trained_model(device):
    model = XRayResNet50(num_classes=len(LABELS), pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, img_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
            
    return probs.squeeze().cpu().numpy()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Modeli Yükle
    model = load_trained_model(device)
    
    # 2. Rastgele Resim Seç
    all_images = list(IMG_DIR.glob("*.png"))
    if not all_images:
        logger.error("❌ Klasörde resim yok.")
        return

    random_img_path = random.choice(all_images)
    img_name = random_img_path.name
    
    # 3. Gerçek Tanıyı Bul (Ground Truth)
    true_label = get_ground_truth(img_name)
    
    logger.info(f"\n📸 Resim: {img_name}")
    logger.info(f"🩺 GERÇEK TANI (CSV): {true_label}")
    
    # 4. Model Tahmini
    probabilities = predict_image(model, random_img_path, device)
    
    print("\n--- 🤖 YAPAY ZEKA TAHMİNİ (Top 3) ---")
    results = zip(LABELS, probabilities)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # İlk 3 tahmini ne olursa olsun göster
    for i, (label, prob) in enumerate(sorted_results[:3]):
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # Eğer olasılık %50'den büyükse renkli/uyarıcı yap
        prefix = "🚨" if prob > 0.5 else f"{i+1}."
        print(f"{prefix} {label:<20} : {bar} {prob:.1%}")

    # Eğer hasta sağlıklıysa modelin düşük vermesi normaldir
    if "No Finding" in true_label:
        print("\n✅ Yorum: Hasta sağlıklı görünüyor, modelin düşük oran vermesi NORMAL.")
    else:
        print("\n⚠️ Yorum: Hastalık var. Eğer oranlar düşükse model daha fazla eğitilmeli.")

if __name__ == "__main__":
    main()