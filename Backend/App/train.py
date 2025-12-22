import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing
import copy

try:
      from model.model import HybridDenseNet121
      from model.dataset import NIHChestXrayDataset
except ImportError:
      import sys
      sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
      from model.model import HybridDenseNet121
      from model.dataset import NIHChestXrayDataset

CONFIG = {
      'IMG_DIR': '../../data/raw/images',
      'CSV_FILE': '../../data/raw/Data_Entry_2017.csv',
      'MODEL_SAVE_PATH': '../../saved_models',
      'CLASS_NAMES_SAVE_PATH': '../../Backend/App/class_names.json',
      'IMG_SIZE': 512,
      'BATCH_SIZE': 32,
      'EPOCHS': 20,
      'LEARNING_RATE': 1e-4,
      'NUM_WORKERS': 16,
      'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(message)s')
logger = logging.getLogger()

def calculate_pos_weights(dataset, device):
      """
      Sınıf dengesizliğini (kimi hastalıkların miktarları arasında çok fark var)
      çözmek için 'Weighted Cross Entropy' ağırlıklarını hesaplar.
      
      """
      
      logger.info("Sınıf ağırlıkları hesaplanıyor")
      df = dataset.df

      label_map = dataset.label_map 
      # Sınıf sırasına göre sayım yap
      sorted_labels = sorted(label_map.keys(), key=lambda k: label_map[k])
    
      pos_weights = []
      total_samples = len(df)

      for label in sorted_labels:
            
            # Bu hastalık kaç kişide var? Hastalığın geçtiği satır sayısını veriyor yani
            pos_count = df['Finding Labels'].apply(lambda x: label in x).sum() 

            if pos_count == 0:   #sıfıra bölünme hatasını önlemek için
                  pos_count = 1

            weight = (total_samples - pos_count) / pos_count # ağırlık formülü (weigted cross entropy)
            pos_weights.append(weight) 
            
      weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
      logger.info(f"Hesaplanan Ağırlıklar (İlk 5): {weights_tensor[:5]}")
      return weights_tensor

def train_model():
      
      os.makedirs(CONFIG['MODEL_SAVE_PATH'], exist_ok=True) #eğer ilgili dosya yoksa oluştur ama bizde zaten var.

      #data augmentation kısmı
      # modeli eğitirken görüntü ile oynayarak modeli zorluyoruz. Çeviriyoruz döndürüyoruz falan. 
      train_transform = transforms.Compose([
            transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
            transforms.RandomHorizontalFlip(), #ayna efekti
            transforms.RandomRotation(10),     #hafif döndürme
            transforms.ToTensor(),             
            transforms.Normalize([0.485,0.456, 0.406], [0.229, 0.224, 0.225]) #renkleri standartlaştırma için
      ])

      # doğrulama yaparken, eğitimde olduğu gibi görüntü ile oynamıyoruz.
      val_transform = transforms.Compose([
            transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

      logger.info("Dataset hazırlanıyor...")

      full_dataset = NIHChestXrayDataset(
            csv_file = CONFIG['CSV_FILE'],
            root_dir = CONFIG['IMG_DIR'],
            transform=None # Transform'u split sonrası vereceğiz
      )

      class_names = full_dataset.all_labels
      #sınıf isimlerini kaydet
      with open(CONFIG['CLASS_NAMES_SAVE_PATH'], 'w') as f:
            json.dump(class_names, f)
      
      # Train/Val Split (%80 - %20)
      train_size = int(0.8 * len(full_dataset))
      val_size = len(full_dataset) - train_size
      train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

      train_subset.dataset = copy.deepcopy(full_dataset)
      val_subset.dataset = copy.deepcopy(full_dataset)

      train_subset.dataset.transform = train_transform
      val_subset.dataset.transform = val_transform

      #Data loader
      train_loader = DataLoader(train_subset, batch_size=CONFIG['BATCH_SIZE'], 
                            shuffle = True, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
      #shuffle = true, her epoch başında verileri karıştırır. Model sırayı ezberlemesin.
      #pin_memory=True, veriyi RAM den VRAM'e atarken daha hızlı olur.
      
      val_loader = DataLoader(val_subset, batch_size=CONFIG['BATCH_SIZE']
                              ,shuffle = False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
      
      logger.info(f"Eğitim seti: {len(train_subset)}, Doğrulama seti: {len(val_subset)}")

      #model.py deki model sınıfı
      model = HybridDenseNet121(num_classes=full_dataset.num_classes).to(CONFIG['DEVICE'])
      
      pos_weight = calculate_pos_weights(full_dataset, CONFIG['DEVICE'])
      
      #BCEWithLogitsLoss, çoklu etiket için en uygun hata ölçme yöntemi
      criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

      optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
      
      #ReduceLROnPlateau, Eğer loss düşmezse öğrenöe hızını (lr) yavaşlatır.
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

      best_val_loss = float('inf')
      logger.info("Training başlıyor...")

      for epoch in range(CONFIG['EPOCHS']):
            start_time = time.time()

            model.train() # Eğitim kısmı
            train_loss = 0.0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]")

            for batch in loop:
                  # Görsel ve metadata girişi
                  # Verileri gpu'ya at
                  images = batch['image'].to(CONFIG['DEVICE'])
                  metadata = batch['metadata'].to(CONFIG['DEVICE'])

                  labels = batch['labels'].to(CONFIG['DEVICE'])

                  optimizer.zero_grad()

                  #Model iki girdi ile çağrılacak

                  outputs = model(images, metadata)
                  
                  loss = criterion(outputs, labels) #Hatayı ölç
                  loss.backward() #Türev al
                  optimizer.step() #Ağırlıkları güncelle
                  
                  train_loss += loss.item()
                  loop.set_postfix(loss=loss.item())
           
            avg_train_loss = train_loss / len(train_loader)

            #Validation
            model.eval()  # Test modu
            val_loss = 0.0

            with torch.no_grad():
                  for batch in val_loader:
                        images = batch['image'].to(CONFIG['DEVICE'])
                        metadata = batch['metadata'].to(CONFIG['DEVICE'])
                        labels = batch['labels'].to(CONFIG['DEVICE']) 

                        outputs = model(images, metadata)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)

            #log

            scheduler.step(avg_val_loss)
            duration = time.time() - start_time

            logger.info(f"🏁 Epoch {epoch+1} | Süre: {duration:.0f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                logger.info(f"En iyi model kaydediliyor... ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
                best_val_loss = avg_val_loss
                save_path = f"{CONFIG['MODEL_SAVE_PATH']}/hybrid_densenet_best.pth"
                torch.save(model.state_dict(), save_path)
      
      logger.info("Training tamamlandı!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if CONFIG['DEVICE'] == 'cuda':
      torch.backends.cudnn.benchmark = True
      print(f"Cihaz: {torch.cuda.get_device_name(0)}")
    else: 
      print(f"Cihaz: {CONFIG['DEVICE']}")
    
    train_model()