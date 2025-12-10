import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import sys
import multiprocessing

# --- IMPORT AYARLARI ---
try:
    from train import ChestXrayDataset, build_model, load_data, split_data, CONFIG
except ImportError:
    print("⚠️ train.py modülü bulunamadı. Lütfen dosyanın train.py ile aynı klasörde olduğundan emin ol.")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_cpu = multiprocessing.cpu_count()
NUM_WORKERS = min(16, max_cpu) 

MODEL_PATH = "chest_xray_model.pth"

def evaluate_model():
    print(f"🚀 Değerlendirme Başlıyor...")
    print(f"🔥 Hesaplama Cihazı (Model): {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   Kart: {torch.cuda.get_device_name(0)}")
    print(f"⚙️  Veri Yükleyici (Loader): {NUM_WORKERS} CPU Çekirdeği kullanılıyor.")
    
    # 1. Veriyi Hazırla
    full_df = load_data()
    _, val_df = split_data(full_df)
    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(full_df['Finding Labels'])
    classes = mlb.classes_
    print(f"📋 Sınıflar ({len(classes)}): {classes}")

    # 2. Dataset ve Loader
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = ChestXrayDataset(val_df, CONFIG['IMG_DIR'], val_transform, mlb)
    
    # BURASI KRİTİK: GPU'ya hızlı aktarım için pin_memory=True şarttır
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['BATCH_SIZE'] * 2, # Değerlendirmede batch size'ı artırabiliriz (daha hızlı olur)
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True 
    )
    
    # 3. Modeli Yükle
    model = build_model(num_classes=len(classes))
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model dosyası bulunamadı!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    # 4. Tahminleri Topla
    all_targets = []
    all_preds = []
    
    print("🧪 Test ediliyor...")
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(DEVICE)
            # labels GPU'ya gitmese de olur, CPU'da biriktireceğiz
            
            outputs = model(images)
            probs = torch.sigmoid(outputs) 
            
            all_preds.append(probs.cpu().numpy()) # Sonucu CPU'ya geri çek
            all_targets.append(labels.numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 5. Metrikleri Hesapla (AUC)
    print("\n📊 --- SINIF BAZLI PERFORMANS (AUC) ---")
    print("AUC skoru 0.5 = Kötü, 1.0 = Mükemmel")
    print("-" * 40)
    
    auc_scores = []
    for i, class_name in enumerate(classes):
        try:
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                auc_scores.append(auc)
                print(f"{class_name:<20}: {auc:.4f}")
            else:
                print(f"{class_name:<20}: Yetersiz veri")
        except ValueError:
            print(f"{class_name:<20}: Hata")
            
    print("-" * 40)
    if auc_scores:
        print(f"🏆 Ortalama AUC: {np.mean(auc_scores):.4f}")
    
    # 6. Nodül Kontrolü
    if 'Nodule' in classes:
        nodule_idx = np.where(classes == 'Nodule')[0][0]
        avg_nodule_prob = np.mean(all_preds[:, nodule_idx])
        print(f"\n🕵️‍♂️ Nodül Analizi: Ort. Olasılık {avg_nodule_prob:.4f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate_model()