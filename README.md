# ChestX-Rai:  NIH Chest X-Ray Hybrid Classification & XAI
   
Bu proje, NIH Chest X-Ray veri setini kullanarak 14 farklı göğüs hastalığını tespit eden **Hibrit bir Derin Öğrenme Modeli** geliştirmeyi amaçlar. Standart CNN yaklaşımlarından farklı olarak, görsel verileri hasta demografik bilgileriyle (Yaş ve Cinsiyet) birleştiren bir mimari kullanır ve Grad-CAM ile açıklanabilir yapay zeka (XAI) çıktıları sunar.

## Proje Özellikleri

- DenseNet121 omurgasından gelen görsel özellikler, hasta meta verileri (Yaş ve Cinsiyet) ile birleştirilerek sınıflandırma yapılır.
- ImageNet ağırlıklarıyla eğitilmiş DenseNet121 kullanılmıştır.
- Veri setindeki dengesizliği yönetmek için dinamik olarak hesaplanan **Weighted Cross-Entropy Loss** kullanılır.
- Grad-CAM entegrasyonu sayesinde modelin röntgen üzerinde odaklandığı bölgeler ısı haritası (heatmap) olarak görselleştirilir.
- FastAPI tabanlı backend servisi ile model, web veya mobil uygulamalara entegre edilebilir.

## Mimari Yapı

Model, `HybridDenseNet121` sınıfı altında kurgulanmıştır:

1.  **Görsel Giriş:** 512x512 piksel göğüs röntgeni -> DenseNet121 -> 1024 özellik vektörü.
2.  **Meta Giriş:** Normalize edilmiş Yaş (Age/100) ve Cinsiyet (M=1, F=0) -> 2 özellik vektörü.
3.  **Füzyon (Fusion):** 1024 + 2 = 1026 boyutlu birleşik vektör.
4.  **Sınıflandırıcı:** Fully Connected Layers + ReLU + Dropout -> 14 Hastalık Sınıfı.

## Proje Yapısı

```text
├── Backend
│   └── App
│       ├── api.py           
│       ├── train.py         
│       ├── evaluate.py      
│       └── model
│           ├── model.py     
│           └── dataset.py   
├── Frontend                 
│   ├── src
│   ├── public
│   └── package.json
├── data
│   └── raw                  
├── saved_models             
└── requirements.txt         
```


## Kurulum

Proje Python tabanlıdır ve bağımlılıkların çakışmaması için **Sanal Ortam (Virtual Environment)** kullanılması şiddetle önerilir.
Projeyi klonladıktan sonra sanal ortam oluşturmak adına:

- Windows için:
```bash
python -m venv venv
.\venv\Scripts\activate
```

- Mac/Linux için:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bağımlılıkların Kurulması

- Backend Kurulumu:
```bash
pip install -r requirements.txt
```

- Frontend Kurulumu:
```bash
cd Frontend
npm install
```

## Model Eğitimi

Modeli eğitmek için aşağıdaki komutu çalıştırın. Eğitim sonucunda en iyi ağırlıklar saved_models/hybrid_densenet_best.pth olarak kaydedilecektir.

```py
python Backend/App/train.py
```


## Server ve Arayüzü Başlatma

1. Sunucuyu ayağa kaldırmak için:

```py
python Backend/App/api.py
```

2. Önyüzü ayağa kaldırmak için:

```bash
cd Frontend
npm run dev
```


## Model Değerlendirme

Eğitilen modelin AUC, F1-Score, Sensitivity ve Specificity metriklerini görmek için:
```py
python Backend/App/evaluate.py
```
