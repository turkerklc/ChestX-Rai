from fastapi.middleware.cors import CORSMiddleware
import sys
import torch
import cv2
import numpy as np
import io
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

try: 
      from model.model import XRayResNet50
except ImportError:
      print("Model dosyası bulunamadı")
      sys.exit(1)

PROJECT_ROOT = CURRENT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "saved_models" / "chest_xray_model.pth"

LABELS = [ 
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

app = FastAPI(
      title = "Chest X-Ray xAI ",
      description="Sağlıkta Yapay Zeka: Hastalık tahmini ve Grad-CAM ile açıklanabilirlik.",
      version="1.0"
)
app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # Güvenlik için prodüksiyonda site adresi yazılır, şimdilik herkese açık (*) olsun.
      allow_credentials = True,
      allow_methods=["*"],  # GET, POST, vb. hepsi serbest
      allow_headers=["*"],
      )

model = None
device = None

@app.on_event("startup")
async def startup_event():
      global model, device

      #cihaz seçimi
      device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
      print(f" API başlatılıyor... Cihaz: {device}")

      if not MODEL_PATH.exists():
            print(f"Model dosyası bulunamadı -> {MODEL_PATH}")
            #hata durumunda boş modelle devam etmemek için raise et ileride
            return
      
      #Modeli Yükle
      print("Model hafızaya yükleniyor...")
      model = XRayResNet50(num_classes=len(LABELS), pretrained=False)

      #Map_location
      checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
      model.load_state_dict(checkpoint)
      model = model.to(device)
      model.eval()

      print("Model hazır ve istek bekliyor")

def process_image(image_bytes):
      #Gelen byte verisini PIL görüntüsüne çevirir.
      try:
         image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
         return image
      except Exception:
         raise HTTPException(status_code = 400, detail="Gönderilen dosya geçerli bir resim değil.")
      
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Röntgen görüntüsünü alır, hastalık olasılıklarını JSON olarak döner.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    image_bytes = await file.read()
    image = process_image(image_bytes)
    
    # Resmi Hazırla (Eğitimdeki aynı transformlar)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tahmin
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Sonuçları Sözlüğe Çevir
    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    
    # Olasılığa göre büyükten küçüğe sırala
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return JSONResponse(content=sorted_results)

# --- ENDPOINT 2: xAI / ISI HARİTASI (RESİM) ---
@app.post("/explain")
async def explain_endpoint(file: UploadFile = File(...)):
    """
    Röntgeni alır, Grad-CAM ısı haritası uygulanmış halini RESİM (PNG) olarak döner.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    # Grad-CAM için CPU kullanmak görselleştirmede daha stabildir
    target_device = torch.device("cpu")
    # Modeli geçici olarak CPU'ya kopyala veya taşı (Hafif bir işlem)
    # Not: Performans için ana model device'ında da yapılabilir ama MPS bazen Grad-CAM kancalarında sorun çıkarabilir.
    viz_model = XRayResNet50(num_classes=len(LABELS), pretrained=False).to(target_device)
    viz_model.load_state_dict(model.state_dict()) # Ana modelin ağırlıklarını al
    viz_model.eval()

    image_bytes = await file.read()
    
    # 1. Görüntüyü OpenCV formatına (Numpy) çevir
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 0-1 Arası Float ve Resize
    img_float = np.float32(img_rgb) / 255
    img_float = cv2.resize(img_float, (224, 224))
    
    # 2. Tensor Hazırla
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(Image.fromarray((img_float * 255).astype(np.uint8))).unsqueeze(0).to(target_device)
    
    # 3. Grad-CAM Çalıştır
    # ResNet50'nin son konvolüsyon katmanı: backbone.layer4
    target_layers = [viz_model.layer4[-1]]
    cam = GradCAM(model=viz_model, target_layers=target_layers)
    
    # Otomatik en yüksek sınıfı hedefle (targets=None)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # 4. Görüntüleri Birleştir
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    # 5. Geriye Resim Olarak Gönder
    res, im_png = cv2.imencode(".png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    # Dosya direkt çalıştırılırsa (python api.py)
    uvicorn.run(app, host="0.0.0.0", port=8000)