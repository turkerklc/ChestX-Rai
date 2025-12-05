import os
import time
import json
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from tqdm import tqdm

# --- AYARLAR (RTX 5090 İÇİN OPTİMİZE EDİLDİ) ---
CONFIG = {
    'IMG_SIZE': 512,        # ResNet standart girişi (Daha yüksek isterseniz 512 yapın ama 224 daha stabildir)
    'BATCH_SIZE': 64,       # 5090'ın belleği yeter, artırılabilir (128 denenebilir)
    'EPOCHS': 25,           # Eğitim süresi
    'LEARNING_RATE': 1e-4,  # Hassas öğrenme
    'DATA_CSV': 'Data_Entry_2017.csv',
    'IMG_DIR': 'images',    # Resimlerin olduğu klasör
    'MODEL_SAVE_PATH': 'chest_xray_model.pth',
    'CLASS_NAMES_SAVE_PATH': 'class_names.json'
}

# --- CİHAZ SEÇİMİ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Cihaz: {device}")
if torch.cuda.is_available():
    print(f"🔥 Ekran Kartı: {torch.cuda.get_device_name(0)}")

# --- 1. VERİ HAZIRLIĞI (DATA PROCESSING) ---
def load_data():
    print("📊 Veri seti okunuyor ve işleniyor...")
    df = pd.read_csv(CONFIG['DATA_CSV'])
    
    # Gereksiz sütunları atalım, sadece Resim Adı, Hastalıklar ve Hasta ID kalsın
    df = df[['Image Index', 'Finding Labels', 'Patient ID']]
    
    # Hastalıkları listeye çevir (Örn: "Infiltration|Pneumonia" -> ["Infiltration", "Pneumonia"])
    df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
    
    return df

# --- 2. HASTA BAZLI BÖLME (PATIENT-LEVEL SPLIT) ---
def split_data(df):
    print("✂️ Veri, hasta bazlı bölünüyor (Data Leakage Önlemi)...")
    
    patient_ids = df['Patient ID'].unique()
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    
    train_df = df[df['Patient ID'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(val_ids)].reset_index(drop=True)
    
    print(f"✅ Eğitim Seti: {len(train_df)} görüntü")
    print(f"✅ Doğrulama Seti: {len(val_df)} görüntü")
    
    return train_df, val_df

# --- 3. DATASET SINIFI ---
class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mlb=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mlb = mlb
        
        # Etiketleri One-Hot Encode yap (0 ve 1'lere çevir)
        self.labels = self.mlb.transform(self.df['Finding Labels'])
        self.image_names = self.df['Image Index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Eğer resim bulunamazsa siyah bir resim döndür (kodu patlatma)
            print(f"⚠️ Uyarı: {img_path} bulunamadı, atlanıyor.")
            image = Image.new('RGB', (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# --- 4. MODEL MİMARİSİ ---
def build_model(num_classes):
    print("🏗️ ResNet-50 modeli indiriliyor ve hazırlanıyor...")
    # Weights parametresi yeni PyTorch sürümleri için güncellendi
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Son katmanı bizim hastalık sayımıza göre değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)

# --- 5. EĞİTİM DÖNGÜSÜ ---
def train_model():
    # 1. Veriyi Yükle
    full_df = load_data()
    
    # 2. Etiketleyiciyi Hazırla (Binarizer)
    mlb = MultiLabelBinarizer()
    mlb.fit(full_df['Finding Labels'])
    classes = mlb.classes_
    print(f" Tespit Edilecek Sınıflar ({len(classes)}): {classes}")
    
    # Sınıf isimlerini kaydet (Frontend için kritik!)
    with open(CONFIG['CLASS_NAMES_SAVE_PATH'], 'w') as f:
        json.dump(list(classes), f)
    print(f" Sınıf listesi kaydedildi: {CONFIG['CLASS_NAMES_SAVE_PATH']}")

    # 3. Veriyi Böl
    train_df, val_df = split_data(full_df)

    # 4. Dönüşümler (Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.RandomHorizontalFlip(), # Ayna efekti
        transforms.RandomRotation(10),     # Hafif döndürme
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet standartları
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. DataLoaderları Oluştur
    train_dataset = ChestXrayDataset(train_df, CONFIG['IMG_DIR'], train_transform, mlb)
    val_dataset = ChestXrayDataset(val_df, CONFIG['IMG_DIR'], val_transform, mlb)

    # num_workers=8 veya 16 yapabilir Eray (CPU çekirdeğine göre)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)

    # 6. Modeli Kur
    model = build_model(len(classes))
    
    # Multi-Label için Loss Fonksiyonu: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    
    # Öğrenme hızını zamanla azalt (Scheduler)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_loss = float('inf')

    print("\n🔥 EĞİTİM BAŞLIYOR... (Kahveni al, bu biraz sürebilir)\n")
    
    for epoch in range(CONFIG['EPOCHS']):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())
            
        train_loss = train_loss / len(train_loader.dataset)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Scheduler Adımı
        scheduler.step(val_loss)

        # Süre ve Log
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Bitti | Süre: {epoch_time:.0f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Checkpoint: Eğer model geliştiyse kaydet
        if val_loss < best_val_loss:
            print(f"⭐ Validation Loss düştü ({best_val_loss:.4f} -> {val_loss:.4f}). Model kaydediliyor...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])

    print("\n✅ EĞİTİM TAMAMLANDI!")
    print(f"🏆 En iyi model şuraya kaydedildi: {CONFIG['MODEL_SAVE_PATH']}")

if __name__ == '__main__':
    train_model()