import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
from pathlib import Path

# --- ORTAM AYARLARI ---
# Bu dosyanın bulunduğu yer (App klasörü)
CURRENT_DIR = Path(__file__).resolve().parent

# Modülleri (model klasörünü) görebilmek için yolu ekle
sys.path.append(str(CURRENT_DIR))

try:
    from model.dataset import NIHChestXrayDataset
    from model.model import XRayResNet50
except ImportError as e:
    print(f"❌ Import Hatası: {e}")
    print("Lütfen dosya yapısının 'App/model/dataset.py' ve 'App/model/model.py' şeklinde olduğundan emin olun.")
    sys.exit(1)

# Loglama Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- KONFİGÜRASYON (Ayarlar) ---
class Config:
    PROJECT_NAME = "NIH_XRay_xAI"
    
    # Dosya Yolları (Dinamik)
    # App -> Backend -> xAI-chest (Proje Ana Dizini)
    # train.py, 'App' içinde olduğu için 2 basamak yukarı çıkıyoruz.
    PROJECT_ROOT = CURRENT_DIR.parent.parent
    
    CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data_Entry_2017.csv"
    IMG_DIR = PROJECT_ROOT / "data" / "raw" / "images"
    SAVE_DIR = PROJECT_ROOT / "saved_models"
    
    # Hiperparametreler (Modelin Ayarları)
    BATCH_SIZE = 32         # M2 Mac için ideal (RAM şişerse 16 yap)
    LEARNING_RATE = 1e-4    # 0.0001 (Hassas öğrenme)
    NUM_EPOCHS = 5          # Lite veri olduğu için 5 tur hızlı biter
    IMAGE_SIZE = 224        # ResNet standardı
    NUM_WORKERS = 2         # Veri yükleme işçisi

def get_device():
    """Donanımı otomatik seçer."""
    if torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda") # NVIDIA
    else:
        return torch.device("cpu") # İşlemci (Yavaş)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Tek bir eğitim turunu (Epoch) çalıştırır."""
    model.train() # Modeli eğitim moduna al
    running_loss = 0.0
    
    # İlerleme çubuğu (Progress Bar)
    loop = tqdm(loader, leave=True, desc="Eğitim")
    
    for batch in loop:
        # 1. Veriyi Cihaza Yükle
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        # 2. İleri Yayılım (Forward Pass) - Tahmin Et
        outputs = model(images)
        
        # 3. Hatayı Hesapla (Loss)
        loss = criterion(outputs, labels)
        
        # 4. Geri Yayılım (Backward Pass) - Öğren
        optimizer.zero_grad() # Eski türevleri temizle
        loss.backward()       # Hatanın kaynağını bul
        optimizer.step()      # Ağırlıkları güncelle
        
        # İstatistikleri Güncelle
        running_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")
        
    return running_loss / len(loader)

def main():
    device = get_device()
    logger.info(f"🚀 Proje: {Config.PROJECT_NAME}")
    logger.info(f"🖥️  Cihaz: {device}")
    
    # 1. Klasör ve Dosya Kontrolü
    if not Config.CSV_PATH.exists() or not Config.IMG_DIR.exists():
        logger.error(f"❌ Kritik dosyalar bulunamadı!")
        logger.error(f"   CSV: {Config.CSV_PATH}")
        logger.error(f"   IMG: {Config.IMG_DIR}")
        return

    # Kayıt klasörünü oluştur
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    # 2. Veri Seti Hazırlığı
    logger.info("📊 Veri seti hazırlanıyor...")
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        # ImageNet istatistiklerine göre normalize et (Önemli!)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = NIHChestXrayDataset(
        csv_file=str(Config.CSV_PATH), 
        root_dir=str(Config.IMG_DIR), 
        transform=transform
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    
    logger.info(f"✅ Eğitim verisi yüklendi: {len(dataset)} görüntü")

    # 3. Model Kurulumu
    model = XRayResNet50(num_classes=dataset.num_classes, pretrained=True)
    model = model.to(device)
    
    # 4. Loss ve Optimizer
    # Multi-label (Çoklu Etiket) olduğu için BCEWithLogitsLoss şarttır.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 5. Büyük Döngü Başlıyor
    logger.info("🔥 Eğitim Başlıyor...")
    
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        
        logger.info(f"📉 Epoch {epoch+1} Bitti. Ortalama Hata (Loss): {avg_loss:.4f}")
        
        # Modeli Kaydet
        save_path = Config.SAVE_DIR / f"resnet50_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"💾 Checkpoint kaydedildi: {save_path.name}")

    logger.info("\n🎉 TEBRİKLER! Tüm eğitimler başarıyla tamamlandı.")
    logger.info(f"📂 Modeller şurada: {Config.SAVE_DIR}")

if __name__ == "__main__":
    main()