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

# --- DÜZELTİLMİŞ ALFABETİK LİSTE ---
LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',       # <-- 10. Sıra (Doğru Yer)
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',        # <-- 13. Sıra (Doğru Yer)
    'Pneumothorax'
]

# Başlangıçta Listeyi Kontrol Et (Terminalde Yazar)
print("--- LİSTE KONTROLÜ ---")
print(f"10. Sıra (Beklenen: No Finding): {LABELS[10]}")
print(f"13. Sıra (Beklenen: Pneumonia):  {LABELS[13]}")
print("----------------------")

app = FastAPI(
      title = "Chest X-Ray xAI ",
      description="Sağlıkta Yapay Zeka: Hastalık tahmini ve Grad-CAM ile açıklanabilirlik.",
      version="1.0"
)
app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials = True,
      allow_methods=["*"],
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
    
    # --- İŞTE BU SATIR SİLİNMİŞTİ, GERİ EKLENDİ ---
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tahmin
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # --- SUÇÜSTÜ LOGLARI ---
    max_index = probs.argmax()
    max_score = probs[max_index]
    current_label_at_index = LABELS[max_index]
    
  

    # Sonuçları Sözlüğe Çevir
    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    
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

    target_device = torch.device("cpu")
    viz_model = XRayResNet50(num_classes=len(LABELS), pretrained=False).to(target_device)
    viz_model.load_state_dict(model.state_dict())
    viz_model.eval()

    image_bytes = await file.read()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img_float = np.float32(img_rgb) / 255
    img_float = cv2.resize(img_float, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(Image.fromarray((img_float * 255).astype(np.uint8))).unsqueeze(0).to(target_device)
    
    target_layers = [viz_model.layer4[-1]]
    cam = GradCAM(model=viz_model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    res, im_png = cv2.imencode(".png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)