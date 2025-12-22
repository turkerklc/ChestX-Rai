NIH Chest X-Ray Multi-Label Classification & XAI
Bu proje, NIH Chest X-Ray veri setini kullanarak 14 farklı göğüs hastalığını tespit eden Hibrit bir Derin Öğrenme Modeli geliştirmeyi amaçlar. Standart CNN yaklaşımlarından farklı olarak, görsel verileri hasta demografik bilgileriyle (Yaş ve Cinsiyet) birleştiren bir mimari kullanır ve Grad-CAM ile açıklanabilir yapay zeka (XAI) çıktıları sunar.


- DenseNet121 omurgasından gelen görsel özellikler (Image Features), hasta meta verileri (Yaş ve Cinsiyet) ile birleştirilerek sınıflandırma yapılır.

- ImageNet ağırlıklarıyla eğitilmiş DenseNet121 kullanılmıştır.

-  Veri setindeki dengesizliği yönetmek için dinamik olarak hesaplanan Weighted Cross-Entropy Loss kullanılır.

- Grad-CAM entegrasyonu sayesinde modelin röntgen üzerinde odaklandığı bölgeler ısı haritası (heatmap) olarak görselleştirilir.

- FastAPI tabanlı backend servisi ile model, web veya mobil uygulamalara entegre edilebilir.

Cross-Platform: NVIDIA GPU (CUDA), Apple Silicon (MPS) ve CPU üzerinde çalışabilir.

 Mimari Yapı
Model, HybridDenseNet121 sınıfı altında kurgulanmıştır:

Görsel Giriş: 512x512 piksel göğüs röntgeni -> DenseNet121 -> 1024 özellik vektörü.

Meta Giriş: Normalize edilmiş Yaş (Age/100) ve Cinsiyet (M=1, F=0) -> 2 özellik vektörü.

Füzyon (Fusion): 1024 + 2 = 1026 boyutlu birleşik vektör.

Sınıflandırıcı: Fully Connected Layers + ReLU + Dropout -> 14 Hastalık Sınıfı.

 Proje Yapısı
Plaintext

├── Backend
│   └── App
│       ├── api.py           # FastAPI sunucusu ve tahmin endpointleri
│       ├── train.py         # Model eğitim döngüsü ve validasyon
│       ├── evaluate.py      # AUC, F1, Sensitivity metriklerinin hesaplanması
│       └── model
│           ├── model.py     # HybridDenseNet121 mimarisi
│           └── dataset.py   # Veri yükleme, işleme ve augmentation
├── data
│   └── raw                  # Ham veri (images ve csv)
├── saved_models             # Eğitilen model (.pth) ve sınıf isimleri (.json)
└── requirements.txt         # Gerekli Python kütüphaneleri
 Kurulum
Gerekli bağımlılıkları yükleyin:

Bash

pip install -r requirements.txt
Not: PyTorch kurulumu sisteminize (CUDA/Mac) göre değişiklik gösterebilir.

 Kullanım
1. Model Eğitimi (Training)
Modeli eğitmek için aşağıdaki komutu çalıştırın. Eğitim sonucunda en iyi ağırlıklar saved_models/hybrid_densenet_best.pth olarak kaydedilecektir.

Bash

python Backend/App/train.py
Konfigürasyon (Batch size, Epoch, Learning Rate) train.py içerisindeki CONFIG sözlüğünden değiştirilebilir.

2. Model Değerlendirme (Evaluation)
Eğitilen modelin AUC, F1-Score, Sensitivity ve Specificity metriklerini görmek için:

Bash

python Backend/App/evaluate.py
3. API Başlatma
Tahmin sistemini ve arayüzü ayağa kaldırmak için:

Bash

python Backend/App/api.py
API varsayılan olarak http://localhost:8000 adresinde çalışır.

 API Endpointleri
API, Swagger UI dokümantasyonu ile gelir (/docs).

POST /predict: Görüntü, yaş ve cinsiyet alır; hastalık olasılıklarını JSON olarak döner.

POST /explain: Görüntü, yaş ve cinsiyet alır; Grad-CAM ısı haritası uygulanmış görseli döner.

 Gereksinimler
Python 3.8+

PyTorch, Torchvision

Pandas, Numpy, OpenCV

FastAPI, Uvicorn

grad-cam (pytorch-grad-cam)