# ChestAI - Web Interface 

Bu klasör, ChestAI projesinin kullanıcı arayüzünü (Frontend) içerir. **React**, **Vite** ve **Tailwind CSS** kullanılarak geliştirilmiştir.

## Tech Stack

* **Framework:** React 18
* **Build Tool:** Vite
* **Stil:** Tailwind CSS
* **HTTP İstekleri:** Axios
* **İkonlar:** React Icons

## Kurulum ve Çalıştırma

Bu arayüzü geliştirmek veya çalıştırmak için bilgisayarınızda **Node.js** yüklü olmalıdır.

### 1. Bağımlılıkları Yükleyin
Frontend klasörünün içindeyken terminalde şu komutu çalıştırın:

      npm install

Geliştirme Sunucusunu Başlatın
Arayüzü yerel sunucuda (localhost) çalıştırmak için:

      npm run dev

Site varsayılan olarak http://localhost:5173 adresinde açılacaktır.

Backend Bağlantısı
Bu arayüz, analiz yapabilmek için Backend API'sine ihtiyaç duyar. Lütfen Backend/App klasöründeki API'nin çalışır durumda olduğundan emin olun:

API Adresi: http://127.0.0.1:8000

Dosya Yapısı
src/App.jsx: Ana uygulama mantığı ve tasarımın bulunduğu dosya.

src/index.css: Tailwind tanımlamalarının ve global stillerin bulunduğu dosya.

tailwind.config.js: Tema ve renk ayarları.

Prodüksiyon (Canlı) İçin Derleme
Projeyi canlı sunucuya atmadan önce optimize edilmiş sürümü oluşturmak için:

      npm run build

Bu işlem dist klasörü içerisine statik dosyaları oluşturur.