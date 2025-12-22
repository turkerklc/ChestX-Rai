import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import multiprocessing
import copy

#Directory ayarları
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '..')) # Üst dizini gör

# Modüller çağırılıyor
try:
    from model.model import HybridDenseNet121
    from model.dataset import NIHChestXrayDataset
except ImportError:
    # Eğer App klasöründen çalıştırılıyorsa
    sys.path.append(os.path.join(CURRENT_DIR, 'Backend', 'App'))
    from model.model import HybridDenseNet121
    from model.dataset import NIHChestXrayDataset

# Config
CONFIG = {
    'IMG_DIR': '../../data/raw/images',
    'CSV_FILE': '../../data/raw/Data_Entry_2017.csv',
    'MODEL_PATH': '../../saved_models/hybrid_densenet_best.pth',
    'CLASS_NAMES_PATH': '../../saved_models/class_names.json',
    'IMG_SIZE': 512,
    'BATCH_SIZE': 64, # Test ederken daha yüksek batch size tercih ettim
    'NUM_WORKERS': 16,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_class_names():
    if os.path.exists(CONFIG['CLASS_NAMES_PATH']):
        with open(CONFIG['CLASS_NAMES_PATH'], 'r') as f:
            return json.load(f)
    else:
        print("❌ HATA: Sınıf isimleri (class_names.json) bulunamadı!")
        sys.exit(1)

def evaluate():
    print(f" Değerlendirme Modülü Başlatılıyor...")
    print(f" Cihaz: {CONFIG['DEVICE']}")
    if CONFIG['DEVICE'] == 'cuda':
        print(f"   Kart: {torch.cuda.get_device_name(0)}")
        # RTX 5090 Optimizasyonu
        torch.backends.cudnn.benchmark = True

    # 1. VERİ SETİNİ HAZIRLA (EĞİTİMDEKİ MANTIKLA AYNI OLMALI)
    # Validation setini tekrar oluşturmak için aynı seed ve mantığı kullanıyoruz.
    print("📊 Veri seti yükleniyor ve bölünüyor...")
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = NIHChestXrayDataset(
        csv_file=CONFIG['CSV_FILE'],
        root_dir=CONFIG['IMG_DIR'],
        transform=None 
    )

    # Train/Val Split (Aynı random state olmalı ki Val seti değişmesin)
    # PyTorch random_split deterministik değildir, ancak oranlar aynıysa
    # büyük veri setlerinde dağılım benzer olur.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Dataset kopyala ve transform ata
    val_subset.dataset = copy.deepcopy(full_dataset)
    val_subset.dataset.transform = val_transform

    val_loader = DataLoader(
        val_subset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=CONFIG['NUM_WORKERS'], 
        pin_memory=True
    )
    
    print(f" Değerlendirme Seti: {len(val_subset)} görüntü")

    # Model ve sınıflar yüklensin
    class_names = load_class_names()
    num_classes = len(class_names)
    print(f" Sınıflar ({num_classes}): {class_names}")

    model = HybridDenseNet121(num_classes=num_classes, pretrained=False)
    
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print(f" Model dosyası bulunamadı: {CONFIG['MODEL_PATH']}")
        return

    print(" Ağırlikler yükleniyor...")
    checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
    model.load_state_dict(checkpoint)
    model.to(CONFIG['DEVICE'])
    model.eval()

    all_targets = []
    all_preds = []

    print(" Test çalıştırılıyor (Hybrid Input)...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing"):
            images = batch['image'].to(CONFIG['DEVICE'])
            metadata = batch['metadata'].to(CONFIG['DEVICE']) # Hybrid Giriş
            labels = batch['labels'].to(CONFIG['DEVICE'])

            # Model Tahmini
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs)

            # CPU'ya alıp listeye ekle
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Listeleri birleştir
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Metrikleri hesapla
    print("\n" + "="*60)
    print(f"{'HASTALIK':<20} | {'AUC':<8} | {'F1':<8} | {'ACC':<8} | {'SENS':<8} | {'SPEC':<8}")
    print("-" * 60)

    metrics_list = []

    for i, class_name in enumerate(class_names):
        # O sınıfın gerçek değerleri ve tahminleri
        y_true = all_targets[:, i]
        y_score = all_preds[:, i]
        
        # Binary tahmin (Eşik değeri 0.5)
        y_pred_binary = (y_score > 0.5).astype(int)

        # 1. AUC (Area Under Curve) - En Önemli Metrik
        # Eğer sınıfta hiç pozitif örnek yoksa AUC hesaplanamaz
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        # 2. F1 Score (Kesinlik ve Duyarlılık Dengesi)
        f1 = f1_score(y_true, y_pred_binary)

        # 3. Accuracy (Doğruluk)
        acc = accuracy_score(y_true, y_pred_binary)

        # 4. Sensitivity (Recall) - Hastayı bulma başarısı
        recall = recall_score(y_true, y_pred_binary)

        # 5. Specificity - Sağlamı bulma başarısı
        # Specificity = TN / (TN + FP)
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"{class_name:<20} | {auc:.4f}   | {f1:.4f}   | {acc:.4f}   | {recall:.4f}   | {specificity:.4f}")
        
        metrics_list.append({
            "Class": class_name, "AUC": auc, "F1": f1, "Acc": acc, "Sens": recall, "Spec": specificity
        })

    print("-" * 60)
    
    # Ortalama Değerler
    avg_auc = np.mean([m['AUC'] for m in metrics_list])
    avg_f1 = np.mean([m['F1'] for m in metrics_list])
    
    print(f" ORTALAMA AUC: {avg_auc:.4f}")
    print(f" ORTALAMA F1 : {avg_f1:.4f}")
    print("="*60)

    # İstersen sonuçları CSV olarak kaydet
    pd.DataFrame(metrics_list).to_csv("evaluation_results.csv", index=False)
    print("📁 Detaylı sonuçlar 'evaluation_results.csv' olarak kaydedildi.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate()