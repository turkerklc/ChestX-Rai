from fastapi.middleware.cors import CORSMiddleware
import sys
import torch
import cv2
import numpy as np
import io
import json
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision import transforms

#Grad-CAM kütüphanesi
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

#Directory ayarları
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

#HybridDenseNet'i import ediyoruz
try: 
      from model.model import HybridDenseNet121
except ImportError:
      print("Model dosyası (model.py) bulunamadı")
      sys.exit(1)

PROJECT_ROOT = CURRENT_DIR.parent.parent

#train.py'nin kaydettiği dosya ismi
MODEL_PATH = PROJECT_ROOT / "saved_models" / "hybrid_densenet_best.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "saved_models" / "class_names.json"

app = FastAPI(
      title = "Chest X-Ray xAI ",
      description="DenseNet-121 ve metadata ile hastalık tespiti",
      version="2.0"
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
LABELS = []

#Grad-Cam kütüphanesi normalde yalnızca görüntü girişi alan modeller ile çalışır. Fakat bizim modelimiz çok girişli. 
#Metadata'yı içeri hapsedeceğiz.

class GradCAMModelWrapper(torch.nn.Module):
     def __init__(self, model, metadata_tensor):
          super().__init__()
          self.model = model
          self.metadata = metadata_tensor

     def forward(self, x):
          return self.model(x, self.metadata)
     
@app.on_event("startup")
async def startup_event():
      
      global model, device, LABELS

      #cihaz seçimi
      if torch.cuda.is_available():
           device = torch.device("cuda")
           print(f"Cihaz: NVDİA GPU ({torch.cuda.get_device_name(0)})")
        
      elif torch.backends.mps.is_available():
           device = torch.device("mps")
           print(f"Cihaz: Apple Silicon")
      
      else:
           device = torch.device("cpu")
           print("Cihaz: CPU")

      if not MODEL_PATH.exists():
            print(f"Model dosyası bulunamadı -> {MODEL_PATH}")
            return
      
      
      #Labelları yükle
      if CLASS_NAMES_PATH.exists():
           with open(CLASS_NAMES_PATH, "r") as f:
                LABELS = json.load(f)
           print(f"Etiketler yüklendi ({len(LABELS)} sınıf): {LABELS}")
      else:
           print("Hata: class_names.json bulunamadı!")

      #Modeli Yükle
      if not MODEL_PATH.exists():
           print(f"Hata: Model dosyası yok -> {MODEL_PATH}")
           return
  
      print("Hybrid DenseNet hafızaya yükleniyor...")
      # num_classes, JSON'dan gelen liste uzunluğuna eşit olmalı
      model = HybridDenseNet121(num_classes=len(LABELS), pretrained=False)

      #Ağırlıkları yükle
      checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
      model.load_state_dict(checkpoint)
      model = model.to(device)
      model.eval()

      print("Model hazır ve istek bekliyor")

def process_inputs(image_bytes, age: int, gender: str):
     """
        Görüntü ve metadata modele girebilecek hale geliyor.
     """

     #Görüntü kısmı

     try:
          image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
     except Exception:
          raise HTTPException(status_code=400, detail="Geçersiz resim dosyası.")
     
     #Bu kısım train.py ile aynı
     transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
     
     img_tensor = transform(image).unsqueeze(0).to(device)

     #Metadata kısmı
     #Yaş normalizasyonu
     norm_age = float(age) / 100.0

     #Cinsiyet normalizasyonu
     gender_numeric = 1.0 if gender.upper() == 'M' else 0.0

     #Tensor oluşturma
     meta_tensor = torch.tensor([[norm_age, gender_numeric]], dtype=torch.float32).to(device)
    
     return img_tensor, meta_tensor, image


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...),
                           age: int = Form(...),
                           gender: str = Form(...)
):
    """
    Hybrid Model tahmini. İnput olarak görüntü, yaş ve cinsiyet alır.
    """
      
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    if age is None:
         raise HTTPException(status_code= 400, detail= "Lütfen hastanın YAŞ bilgisini giriniz. Hibrit model bu veri olmadan çalışamaz.")
    
    if gender is None:
         raise HTTPException(status_code= 400, detail = "Lütfen CİNSİYET bilgisini giriniz. Model bu bilgi olmadan çalışmaz")
    
    try: 
        age_int = int(age)
    except ValueError:
         raise HTTPException(status_code=400, detail = "Geçersiz yaş formatı. Lütfen sayı giriniz")
    
    image_bytes = await file.read()
    
    #Ön işleme
    img_tensor, meta_tensor, _ = process_inputs(image_bytes, age, gender)

    #Tahmin 
    with torch.no_grad():
         logits = model(img_tensor, meta_tensor)
         probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Sonuçları Etiketlerle Eşleştir
    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return JSONResponse(content=sorted_results)


@app.post("/explain")
async def explain_endpoint(file: UploadFile = File(...),
                           age: int = Form(...),
                           gender: str = Form(...)
):
    """
        Grad-CAM Hybrid Model için ısı haritası üretir. 
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    image_bytes = await file.read()
    #Ön işleme
    img_tensor, meta_tensor, original_image = process_inputs(image_bytes, age, gender)

    #Grad-CAM ayarları
    # model.features bir Sequential bloktur. En sonuncusu genellikle 'norm5'tir.
    target_layers = [model.features[-1]] # DenseNet için standart hedef

    #Meta data için wrapper
    wrapper_model = GradCAMModelWrapper(model, meta_tensor)
    cam = GradCAM(model=wrapper_model, target_layers=target_layers)

    # Haritayı Oluştur
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :] # [512, 512]

    #OpenCV işlemleri
    img_resized = original_image.resize((512, 512))
    img_float = np.float32(img_resized) / 255.0

    # Isı haritasını resmin üzerine bindir
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # PNG olarak kaydet ve gönder
    res, im_png = cv2.imencode(".png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)