import os
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path

# Loglama ayarları (Terminalde temiz bilgi görmek için)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-Ray veri seti için PyTorch Dataset sınıfı.
    
    Çoklu etiket (multi-label) sınıflandırma ve hasta meta verilerini işler.
    Klasörde fiziksel olarak bulunmayan resimleri otomatik filtreler (Lite Modu).
    """

    def __init__(self, 
                 csv_file: str, 
                 root_dir: str, 
                 transform: Optional[Callable] = None):
        """
        Dataset'i başlatır ve veriyi hazırlar.

        Args:
            csv_file (str): 'Data_Entry_2017.csv' dosyasının tam yolu.
            root_dir (str): Resimlerin bulunduğu kök klasör yolu.
            transform (callable, optional): Resimlere uygulanacak transformasyonlar (örn. Resize, Tensor).
        """
        self.csv_file = Path(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

        self._validate_paths()
        self.df = self._load_and_filter_data()
        self.all_labels, self.label_map = self._process_labels()
        self.num_classes = len(self.all_labels)

        logger.info(f"✅ Dataset Hazır: {len(self.df)} görüntü, {self.num_classes} sınıf.")

    def _validate_paths(self):
        """Dosya ve klasörlerin varlığını kontrol eder."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"❌ CSV bulunamadı: {self.csv_file}")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"❌ Resim klasörü bulunamadı: {self.root_dir}")

    def _load_and_filter_data(self) -> pd.DataFrame:
        """
        CSV'yi okur ve sadece diskte mevcut olan resimlerle eşleşenleri filtreler.
        """
        df = pd.read_csv(self.csv_file)
        initial_len = len(df)

        # Klasördeki fiziksel dosyaları listele
        available_images = set(os.listdir(self.root_dir))

        # Filtreleme (Sadece mevcut resimleri tut)
        df_filtered = df[df['Image Index'].isin(available_images)].reset_index(drop=True)
        
        filtered_count = len(df_filtered)
        if filtered_count < initial_len:
            logger.warning(f"⚠️ Lite Mod: {initial_len} satırdan {filtered_count} tanesi yüklendi (Diğerleri klasörde yok).")
        
        return df_filtered

    def _process_labels(self):
        """Benzersiz hastalık etiketlerini çıkarır ve haritalar."""
        # 'No Finding' hariç tüm etiketleri ayrıştır
        all_labels = sorted(list(set(
            [l for labels in self.df['Finding Labels'] for l in labels.split('|') if l != "No Finding"]
        )))
        
        label_map = {label: i for i, label in enumerate(all_labels)}
        # logger.info(f"🏷️ Sınıflar: {all_labels}")
        return all_labels, label_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Belirtilen indisteki veri örneğini getirir.
        
        Returns:
            Dict: 'image', 'labels', 'metadata', 'image_name' içeren sözlük.
        """
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = self.root_dir / img_name
        
        # 1. Resmi Yükle (Hata Yönetimi ile)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Resim okuma hatası ({img_name}): {e}")
            # Hata durumunda siyah kare döndür (Eğitimi kırmamak için)
            image = Image.new('RGB', (224, 224))

        # 2. Transform Uygula
        if self.transform:
            image = self.transform(image)

        # 3. Multi-Hot Encoding
        label_str = row['Finding Labels']
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for label in label_str.split('|'):
            if label in self.label_map:
                label_vec[self.label_map[label]] = 1.0

        # 4. Metadata (Yaş ve Cinsiyet)
        try:
            gender = 1.0 if row['Patient Gender'] == 'M' else 0.0
            age = float(row['Patient Age'])
            # Yaşı normalize etmek (0-100 arası varsayımıyla) model performansını artırabilir
            # age = age / 100.0 
        except ValueError:
            gender, age = 0.0, 0.0 # Hatalı veri varsa varsayılan değer

        metadata = torch.tensor([age, gender], dtype=torch.float32)

        return {
            'image': image,
            'labels': label_vec,
            'metadata': metadata,
            'image_name': img_name
        }

# --- TEST BLOĞU ---
if __name__ == "__main__":
    from torchvision import transforms
    
    # Bu dosyanın bulunduğu konumu referans alarak yolları belirle
    # Bu sayede kodu nereden çalıştırırsan çalıştır yollar bozulmaz.
    CURRENT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = CURRENT_DIR.parent.parent.parent # App/model -> App -> Root
    
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    CSV_PATH = DATA_DIR / "Data_Entry_2017.csv"
    IMG_DIR = DATA_DIR / "images"

    print(f"📍 Proje Kök Dizini: {PROJECT_ROOT}")
    print(f"🔍 Aranan Veri Yolu: {DATA_DIR}")

    if CSV_PATH.exists() and IMG_DIR.exists():
        tx = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        try:
            ds = NIHChestXrayDataset(str(CSV_PATH), str(IMG_DIR), transform=tx)
            
            if len(ds) > 0:
                sample = ds[0]
                print("\n✅ Örnek Veri Çıktısı:")
                print(f"   🖼️  Resim Shape: {sample['image'].shape}")
                print(f"   📊 Etiketler: {sample['labels']}")
                print(f"   👤 Metadata: {sample['metadata']}")
            else:
                logger.warning("Dataset boş. Klasörde resim yok mu?")
                
        except Exception as e:
            logger.error(f"Test sırasında hata: {e}")
    else:
        logger.error("❌ Dosyalar bulunamadı! Lütfen 'data' klasörünün proje ana dizininde olduğundan emin ol.")