import { useState } from 'react';
import axios from 'axios';
import { FaCloudUploadAlt, FaHeartbeat, FaInfoCircle, FaUsers, FaArrowRight, FaMap, FaMapMarked, FaWatchmanMonitoring, FaCannabis, FaExclamation, FaExclamationTriangle } from 'react-icons/fa';
import logo from './assets/logo.jpeg'

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('M');
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);

  const handleAnalyze = async () => {
    if (!selectedFile) return alert("Lütfen bir röntgen seçin!");

    setLoading(true);
    setPredictions(null);
    setHeatmapUrl(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // TAHMİN
      const predRes = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(predRes.data);

      // ISI HARİTASI
      const explainRes = await axios.post('http://127.0.0.1:8000/explain', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob' 
      });
      
      const imageObjectUrl = URL.createObjectURL(explainRes.data);
      setHeatmapUrl(imageObjectUrl);

    } catch (error) {
      console.error("Hata:", error);
      alert("API Hatası! Backend çalışıyor mu? (uvicorn api:app)");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {

      setSelectedFile(null);
      setPreviewUrl(null);
      
      setPredictions(null);
      setHeatmapUrl(null);

      setAge("");
      setGender('');

      const fileInput = document.getElementsById('file-upload')
      if (fileInput) {
        fileInput.value = '';
      }
    }

  const handleAgeChange = (e) => {
    const value = e.target.value;

    if (value === "") {
      setAge("");
      return;
    }

    const numValue = parseInt(value, 10);

    if(!isNaN(numValue) && numValue >= 0 && numValue <= 150) {
      setAge(value);
    }
  };
 
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if(!file) return; // eğer dosya seçilmediyse dur.

    const validTypes = ['image/jpeg', 'image/png', 'image/jpg']; // sadece ekteki uzantıları kabul et.

    if (!validTypes.includes(file.type)) {
      alert("Hata: sadece JPEG, JPG ve PNG formatındaki resimler kabul edilir")
      e.target.value = null //input'u temizle
      return;
    }

    //Dosya boyutunu kontrol et
    const maxSize = 5 * 1024 * 1024; 
    if (file.size > maxSize) {
      alert("Hata: Dosya boyutu çok yüksek! Maksimum 5MB yükleyebilirsiniz");
      e.target.value = null;
      return;
    }
    
    //Her şey okeyse devam et
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPredictions(null);
    setHeatmapUrl(null);
  };

  return (
    <div className="font-sans text-gray-800 bg-gray-50 min-h-screen">
      <nav className="fixed top-0 w-full bg-white/90 backdrop-blur-md shadow-sm z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex justify-between items-center">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => window.scrollTo(0,0)}>
           <img src={logo} alt="ChestXRai Logo" className="h-12 w-auto object-contain" 
            />
            <span className="font-bold text-2xl text-gray-800">ChestX-R<span className="text-[rgb(70,65,180)]">ai</span></span>
          </div>
          <div className="hidden md:flex space-x-10 font-normal">
            <a href="#analyzer" className="hover:text-blue-600 transition">Analiz</a>
            <a href="#details" className="hover:text-blue-600 transition">Proje Hakkında</a>
            <a href="#about" className="hover:text-blue-600 transition">Biz Kimiz</a>
          </div>
        </div>
      </nav>

      <section id="analyzer" className="pt-32 pb-20 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-extrabold text-gray-900 mb-4">
              Yapay Zeka Destekli <span className="text-red-600">Röntgen Analizi</span>
            </h1>
            <p className="text-xl text-gray-500 max-w-2xl mx-auto">
              Saniyeler içinde 14 farklı akciğer hastalığını tespit edin ve xAI teknolojisi ile görsel kanıtları inceleyin.
            </p>
          </div>

          <div className="grid lg:grid-cols-12 gap-8 bg-white p-2 rounded-3xl shadow-2xl border border-gray-100 overflow-hidden">
            {/* SOL PANEL */}
            <div className="lg:col-span-5 p-8 bg-blue-50/50 flex flex-col justify-center">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <FaCloudUploadAlt className="text-blue-600" /> Görüntü Yükle
              </h2>
              <div className="border-3 border-dashed border-blue-200 rounded-2xl bg-white p-8 text-center hover:border-blue-400 transition-all cursor-pointer relative group h-80 flex flex-col justify-center items-center">
                <input type="file" id = "file-upload" accept=".jpg, .jpeg, .png" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="max-h-full max-w-full rounded-lg shadow-sm object-contain" />
                ) : (
                  <div className="group-hover:scale-105 transition-transform duration-300">
                    <div className="bg-blue-100 p-4 rounded-full inline-block mb-4">
                      <FaCloudUploadAlt className="text-4xl text-blue-600" />
                    </div>
                    <p className="text-gray-500 font-medium">Dosyayı buraya sürükleyin</p>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4 mt-6">
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase">Yaş</label>
                  <input type="number" value={age} onChange={handleAgeChange} min = "0" max ="150" className="w-full mt-1 p-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" placeholder="45" />
                </div>
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase">Cinsiyet</label>
                  <select value={gender} onChange={(e) => setGender(e.target.value)} className="w-full mt-1 p-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none">
                    <option value="M">Erkek</option>
                    <option value="F">Kadın</option>
                    <option value="Other">Diğer</option>
                  </select>
                </div>
              </div>
             <div className="flex gap-3 mt-6">
            {/* ANA BUTON (Analiz Et) */}
            <button 
              onClick={handleAnalyze} 
              disabled={loading || !selectedFile}
              className={`flex-1 py-4 rounded-xl font-bold text-white shadow-lg transition-all transform active:scale-95 ${
                loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 hover:-translate-y-0.5'
              }`}
            >
              {loading ? 'Analiz Ediliyor...' : 'Analizi Başlat'}
            </button>

            {/* SIFIRLA BUTONU (Sadece dosya seçiliyse veya sonuç varsa görünür) */}
            {(selectedFile || predictions) && (
              <button 
                onClick={handleReset}
                className="px-6 py-4 rounded-xl font-bold text-white bg-red-600 hover:bg-red-700 hover:-translate-y-0.5 transition-all active:scale-95 border border-gray-300"
              >
                Sıfırla
              </button>
            )}
          </div>
            </div>

            {/* SAĞ PANEL */}
            <div className="lg:col-span-7 p-8 flex flex-col justify-center min-h-[500px]">
              {!predictions && !loading && (
                <div className="text-center text-gray-400">
                  <FaArrowRight className="text-4xl opacity-20 mx-auto mb-6" />
                  <h3 className="text-xl font-medium">Sonuçlar burada görünecek</h3>
                </div>
              )}
              {loading && (
                <div className="text-center">
                  <FaHeartbeat className="text-6xl text-blue-600 animate-pulse mx-auto mb-4" />
                  <h3 className="text-2xl font-bold text-gray-800">Yapay Zeka Çalışıyor...</h3>
                </div>
              )}
              {predictions && (
                <div className="grid md:grid-cols-2 gap-8 animate-fade-in">
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 mb-4">Bulgular</h3>
                    <div className="space-y-4">
                      {Object.entries(predictions).slice(0, 4).map(([label, score]) => (
                        <div key={label} className="bg-white p-3 rounded-xl border border-gray-100 shadow-sm">
                          <div className="flex justify-between items-end mb-2">
                            <span className="font-medium text-gray-700">{label}</span>
                            <span className={`font-bold ${score > 0.5 ? 'text-red-600' : 'text-blue-600'}`}>
                              {(score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-3">
                            <div className={`h-full rounded-full ${score > 0.5 ? 'bg-red-500' : 'bg-blue-600'}`} style={{ width: `${score * 100}%` }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 mb-4">xAI Görsel Kanıt</h3>
                    {heatmapUrl && <img src={heatmapUrl} alt="xAI Heatmap" className="w-full rounded-xl shadow-md border border-gray-200" />}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
      
      <section id="details" className="py-24 bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-10">Proje Hakkında</h2>
          <div className="grid md:grid-cols-3 gap-10">
             <div className="p-6 bg-gray-50 rounded-xl"><FaInfoCircle className="text-4xl text-blue-600 mx-auto mb-4"/> <h3 className="font-bold">ResNet-50</h3>
             <p className = "text-gray-600 text-sm leading-relaxed">
              
              ResNet-50, derin öğrenme dünyasının en güvenilir mimarilerinden biridir. 
              Biz bu projede, bu mimariyi 112.000 adet göğüs röntgeni görüntüsüyle 
              eğiterek, zatürre ve diğer akciğer hastalıklarını %90'a varan doğrulukla 
              tespit edebilecek hale getirdik. Modelimiz, pikseller arasındaki en ince 
              detayları bile yakalayabilir. 
             
             </p>
             </div>
             <div className="p-6 bg-gray-50 rounded-xl"><FaMapMarked className="text-4xl text-blue-600 mx-auto mb-4"/> <h3 className="font-bold">Grad-CAM</h3>
            <p className = "text-gray-600 text-sm leading-relaxed">
              Grad-CAM, derin öğrenme modellerinin (özellikle de görüntü işleme modellerinin) nasıl karar verdiğini açıklamak için kullanılan tekniklerden biridir.
              Yapay zekanın verdiği kararı, girdi olarak aldığı görselin hangi bölgesine bakarak verdiğini göstermek için bir ısı haritası oluşturur. Bu x-AI'nın (Explainable AI)
              temel taşlarından biridir.
              </p>
              </div>
            <div className = "p-6 bg-gray-50 rounded-xl"><FaExclamation className="text-4xl text-blue-600 mx-auto mb-4"/> <h3 className="font-bold">Uyarı</h3>
             <p className = "text-gray-600 text-sm leading-relaxed">
              Bu proje yalnızca eğitim ve akademik araştırma amaçlı geliştirilmiştir. 
              Sunulan sonuçlar kesinlik taşımaz ve profesyonel bir tıbbi teşhis veya doktor muayenesi yerine geçmez. 
              Elde edilen veriler tedavi amaçlı kullanılmamalıdır. Geliştiriciler, olası hatalı sonuçlardan veya bu sonuçlara dayanarak alınan kararlardan sorumlu tutulamaz. 
              Herhangi bir sağlık sorununuzda lütfen uzman bir hekime başvurunuz.
             </p>
             </div>
          </div>
        </div>
      </section>

      <section id="about" className="py-20 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Biz Kimiz</h2>
          <div className="flex justify-center gap-10">
            <div className="bg-white p-6 rounded-xl shadow-lg w-64"><h3 className="font-bold text-lg">Türker Kılıç</h3><p className="text-blue-500">AI Engineer</p></div>
            <div className="bg-white p-6 rounded-xl shadow-lg w-64"><h3 className="font-bold text-lg">Ferhat Köknar</h3><p className="text-blue-500">No Code</p></div>
          </div>
        </div>
      </section>

      <footer className="bg-gray-900 text-white py-8 text-center"><p>&copy; 2025 ChestX-Rai</p></footer>
    </div>
  );
}

export default App;