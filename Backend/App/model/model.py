import torch
import torch.nn as nn
from torchvision import models
import logging

# Loglama ayarı (Terminalde temiz bilgi görmek için)
logger = logging.getLogger(__name__)

class XRayResNet50(nn.Module):
    """
    NIH Chest X-Ray için özelleştirilmiş ResNet50 Modeli.
    
    Özellikler:
    - Pretrained ImageNet ağırlıkları ile başlar (Transfer Learning).
    - Son katman (FC) 14 hastalık sınıfına göre yeniden yapılandırılır.
    - xAI (Grad-CAM) entegrasyonuna uygun yapıdadır.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(XRayResNet50, self).__init__()
        
        logger.info(f"🧠 Model Mimarisisi Başlatılıyor: ResNet50 (Pretrained={pretrained})")
        
        # 1. Backbone (Omurga) Yükle
        # ImageNet ağırlıklarını kullanmak, eğitimin çok daha hızlı ve başarılı olmasını sağlar.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # 2. Classifier (Sınıflandırıcı) Katmanını Değiştir
        # ResNet50'nin orijinal FC (Fully Connected) katmanı 2048 giriş -> 1000 çıkış verir.
        # Biz bunu 2048 giriş -> num_classes (14) çıkış yapacağız.
        
        in_features = self.backbone.fc.in_features # Genelde 2048'dir
        
        # Yeni katmanı oluşturuyoruz
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        logger.info(f"🔧 Modelin son katmanı {in_features} giriş -> {num_classes} çıkış (Hastalık) olarak güncellendi.")

    def forward(self, x):
        """
        Veri modelin içinden akar.
        x: Görüntü Batch'i [Batch_Size, 3, 224, 224]
        return: Tahminler (Logits) [Batch_Size, num_classes]
        """
        return self.backbone(x)

# --- TEST BLOĞU (Terminalden çalıştırılınca devreye girer) ---
if __name__ == "__main__":
    # Logları ekrana basması için basit konfigürasyon
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. Modeli 14 hastalık sınıfı için oluştur
        model = XRayResNet50(num_classes=14)
        
        # 2. Sahte bir veri ile test et (M2 MPS veya CPU)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        
        # [Batch Size=2, Kanal=3 (RGB), Boyut=224x224]
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        # Forward pass (Tahmin yap)
        output = model(dummy_input)
        
        print("\n✅ TEST BAŞARILI!")
        print(f"   Giriş Boyutu: {dummy_input.shape}")
        print(f"   Çıkış Boyutu: {output.shape}") 
        print(f"   Cihaz: {device}")
        
    except Exception as e:
        print(f"❌ HATA: {e}")