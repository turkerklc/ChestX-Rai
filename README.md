# ChestAI: Explainable Chest X-Ray Analysis

![Status](https://img.shields.io/badge/Status-Development-blue) ![Python](https://img.shields.io/badge/Python-3.12-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![License](https://img.shields.io/badge/License-MIT-green)

**ChestAI**, akciğer röntgenlerinden 14 farklı toraks hastalığını tespit eden ve Grad-CAM tekniği kullanarak modelin odaklandığı bölgeleri görselleştiren (xAI) bir derin öğrenme projesidir.

## Proje Özellikleri

* **Derin Öğrenme Mimarisi:** NIH Chest X-Ray veri seti üzerinde eğitilmiş, özelleştirilmiş ResNet-50 modeli.
* **Açıklanabilir Yapay Zeka (xAI):** Modelin karar mekanizmasını şeffaflaştıran Grad-CAM ısı haritası görselleştirmesi.
* **Web Arayüzü:** React ve Tailwind CSS ile geliştirilmiş modern kullanıcı arayüzü.
* **Çapraz Platform:** macOS (Apple Silicon) ve Windows (NVIDIA CUDA) üzerinde çalışabilen hibrit backend yapısı.

## Proje Yapısı

```text
xAI-chest/
├── App/                    # Backend Kaynak Kodları (FastAPI & PyTorch)
│   ├── api.py              # API Sunucusu
│   ├── train.py            # Model Eğitim Scripti
│   ├── predict.py          # Tekli Tahmin Scripti
│   ├── explain.py          # Grad-CAM Görselleştirme Modülü
│   └── model/              # Model Mimarisi ve Veri İşleme
│       ├── model.py        # ResNet-50 Sınıfı
│       └── dataset.py      # Veri Yükleyici (DataLoader)
├── Frontend/               # React Web Arayüzü
├── data/                   # Veri Seti Dizini (Git tarafından takip edilmez)
│   └── raw/
│       ├── Data_Entry_2017.csv
│       └── images/
├── saved_models/           # Eğitilmiş model dosyaları (.pth)
└── requirements.txt        # Python bağımlılıkları
Kurulum Rehberi
Projeyi yerel ortamda çalıştırmak için aşağıdaki adımları takip edin.

1. Projenin Klonlanması
Bash

git clone [https://github.com/KULLANICI_ADI/xAI-chest.git](https://github.com/KULLANICI_ADI/xAI-chest.git)
cd xAI-chest
2. Backend Kurulumu (Python)
macOS (Apple Silicon M1/M2/M3)

Bash

# Sanal ortam oluşturma
python3.12 -m venv venv

# Ortamı aktif etme
source venv/bin/activate

# Bağımlılıkların yüklenmesi
pip install -r requirements.txt
Windows (NVIDIA GPU) Not: Windows ortamında PyTorch'un CUDA sürümü ayrıca kurulmalıdır.

Bash

# Sanal ortam oluşturma
python -m venv venv

# Ortamı aktif etme
venv\Scripts\activate

# PyTorch (CUDA Destekli) kurulumu
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Diğer bağımlılıkların yüklenmesi
pip install -r requirements.txt
3. Veri Setinin Hazırlanması
NIH Chest X-Ray veri setini indirin ve proje dizininde aşağıdaki yapıyı oluşturun:

CSV Dosyası: data/raw/Data_Entry_2017.csv konumuna yerleştirilmelidir.

Görüntüler: data/raw/images/ klasörü içerisine çıkartılmalıdır.

Uygulamayı Çalıştırma
Sistemi çalıştırmak için iki ayrı terminal penceresi kullanılmalıdır.

Terminal 1: Backend (API)
Bash

# Sanal ortamın aktif olduğundan emin olun
source venv/bin/activate  # Windows: venv\Scripts\activate

cd Backend/App
uvicorn api:app --reload
API sunucusu http://127.0.0.1:8000 adresinde çalışmaya başlayacaktır.

Terminal 2: Frontend (Web Arayüzü)
Bash

cd Frontend
npm install  # İlk kurulumda gereklidir
npm run dev
Web arayüzü http://localhost:5173 adresinde erişilebilir olacaktır.

Model Eğitimi
Modeli sıfırdan eğitmek için aşağıdaki komut kullanılabilir. Eğitim parametreleri (Epoch, Batch Size vb.) train.py dosyası içerisindeki Config sınıfından düzenlenebilir.

Bash

# App klasöründeyken
python train.py
Lisans ve Referanslar
Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. Tıbbi teşhis için tek başına kullanılamaz.

Veri Seti: NIH Chest X-Ray Dataset