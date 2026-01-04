import { useState, useRef } from 'react';
import axios from 'axios';
import { FaCloudUploadAlt, FaHeartbeat, FaInfoCircle, FaArrowRight, FaMapMarked, FaExclamation, FaChevronUp, FaChevronDown, FaDownload, FaSpinner, FaCheckCircle, FaGithub } from 'react-icons/fa';
import logo from './assets/logo.jpeg';

const DISEASE_DESCRIPTIONS = {
  'Atelectasis': 'Akciğerin bir kısmının veya tamamının sönmesi (büzüşmesi) durumudur. Genellikle hava yollarının tıkanması sonucu oluşur.',
  'Cardiomegaly': 'Kalbin normal boyutlarından daha büyük olması durumudur (Kalp Büyümesi). Yüksek tansiyon veya kalp yetmezliği belirtisi olabilir.',
  'Effusion': 'Akciğer zarları arasında anormal sıvı birikmesidir (Plevral Efüzyon). Nefes darlığına yol açabilir.',
  'Infiltration': 'Akciğer dokusuna hava yerine sıvı, kan veya iltihap dolmasıdır. Genellikle zatürre veya tüberkülozda görülür.',
  'Mass': 'Akciğerde 3 cm\'den büyük anormal doku büyümesidir. Tümör veya kist olabilir, ileri tetkik gerektirir.',
  'Nodule': 'Akciğerde 3 cm\'den küçük, yuvarlak doku büyümesidir. Genellikle iyi huyludur ancak takip edilmesi gerekir.',
  'Pneumonia': 'Akciğer dokusunun iltihaplanmasıdır (Zatürre). Bakteri, virüs veya mantar kaynaklı olabilir.',
  'Pneumothorax': 'Akciğer ile göğüs duvarı arasına hava kaçmasıdır (Akciğer Sönmesi). Ani nefes darlığı ve ağrı yapar.',
  'Consolidation': 'Akciğerdeki hava keseciklerinin (alveol) sıvı veya iltihapla dolup katılaşmasıdır.',
  'Edema': 'Akciğer dokusunda aşırı sıvı birikmesidir (Ödem). Genellikle kalp yetmezliğine bağlı gelişir.',
  'Emphysema': 'Hava keseciklerinin hasar görüp genişlemesiyle oluşan kronik bir hastalıktır. Genellikle sigara kullanımıyla ilişkilidir.',
  'Fibrosis': 'Akciğer dokusunun kalınlaşması ve sertleşmesidir (Yara dokusu). Akciğerin esnekliğini kaybetmesine neden olur.',
  'Pleural_Thickening': 'Akciğer zarlarının kalınlaşmasıdır. Geçirilmiş enfeksiyonlar veya asbest maruziyeti sonucu oluşabilir.',
  'Hernia': 'Organların (genellikle mide) diyaframdaki bir açıklıktan göğüs boşluğuna kaymasıdır (Fıtık).',
  'No Finding': 'Röntgen görüntüsünde herhangi bir patolojik bulguya rastlanmamıştır. Ciğerler temiz görünüyor.'
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('M');
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [expandedDisease, setExpandedDisease] = useState(null);

  const resultsRef = useRef(null);

  // --- ÖZEL KAYDIRMA FONKSİYONU ---
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      // Elementin sayfanın en tepesine olan mutlak mesafesini al
      const elementPosition = element.getBoundingClientRect().top + window.scrollY;
      
      // HESAPLAMA:
      // (Elementin Yeri) - (Ekranın Yarısı) + (Elementin Yarısı) = Tam Ortalar
      // Sonraki sayı ile ince ayar yapıyoruz:
      // Eğer bölümü daha AŞAĞIDA görmek istiyorsan (yukarıda boşluk kalsın), bu sayıyı NEGATİF yap veya azalt.
      // Eğer bölümü daha YUKARIDA görmek istiyorsan, bu sayıyı ARTIR.
      
      // ŞU ANKİ AYAR: -100 (Bölümü ekranda biraz daha aşağı iter, ferah durur)
      const offsetPosition = elementPosition - (window.innerHeight / 2) + (element.offsetHeight / 2) - 100;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth' // YUMUŞAK GEÇİŞ
      });
    }
  };

  const toggleDescription = (diseaseName) => {
    if (expandedDisease === diseaseName) {
      setExpandedDisease(null);
    } else {
      setExpandedDisease(diseaseName);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return alert("Lütfen bir röntgen seçin!");

    setLoading(true);
    setPredictions(null);
    setHeatmapUrl(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('age', age);
    formData.append('gender', gender);

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

      // Sonuçlara kaydır (Hafif gecikmeli ki render olsun)
      setTimeout(() => {
        // Sonuçları da ortalayarak göster
        if (resultsRef.current) {
            const element = resultsRef.current;
            const elementPosition = element.getBoundingClientRect().top + window.scrollY;
            const offsetPosition = elementPosition - (window.innerHeight / 2) + (element.offsetHeight / 2);
            window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
        }
      }, 100);

    } catch (error) {
      console.error("Hata:", error);

      if (error.response) {
        
        // 1. Durum: Backend'den gelen 400 Hatası (Bizim manuel fırlattıklarımız)
        if (error.response.status === 400) {
          const mesaj = error.response.data.detail || "Eksik bilgi girildi.";
          alert(mesaj);
        } 
        
        // 2. Durum: FastAPI'nin Otomatik Doğrulama Hatası (422)
        // Yaş boş bırakıldığında veya harf girildiğinde burası çalışır.
        else if (error.response.status === 422) {
          alert("Lütfen YAŞ kısmını boş bırakmayınız ve sadece sayı giriniz.");
        } 
        
        // 3. Durum: Diğer sunucu hataları (500 vb.)
        else {
          alert(`Sunucu Hatası! Kod: ${error.response.status}`);
        }
        // -----------------------------

      } else if (error.request) {
        alert("API Hatası! Backend çalışmıyor olabilir. (uvicorn api:app çalışıyor mu?)");
      } else {
        alert("Beklenmedik bir hata oluştu.");
      }
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
    setGender('M');

    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleAgeChange = (e) => {
    const value = e.target.value;
    if (value === "") {
      setAge("");
      return;
    }
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 150) {
      setAge(value);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];

    if (!validTypes.includes(file.type)) {
      alert("Hata: sadece JPEG, JPG ve PNG formatındaki resimler kabul edilir")
      e.target.value = null;
      return;
    }

    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      alert("Hata: Dosya boyutu çok yüksek! Maksimum 5MB yükleyebilirsiniz");
      e.target.value = null;
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPredictions(null);
    setHeatmapUrl(null);
  };

  const handleDownloadHeatmap = () => {
    if (!heatmapUrl) return;
    const link = document.createElement('a');
    link.href = heatmapUrl;
    link.download = 'ChestXRai-Analiz.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="font-sans text-gray-800 bg-gray-50 min-h-screen">
      {/* NAVBAR */}
      <nav className="fixed top-0 w-full bg-white/80 backdrop-blur-md border-b border-gray-100 z-50 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6 h-20 flex justify-between items-center">
          
          {/* LOGO */}
          <div className="flex items-center gap-3 cursor-pointer group" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <img src={logo} alt="ChestXRai Logo" className="h-10 w-auto group-hover:scale-105 transition-transform" />
            <span className="font-bold text-2xl tracking-tight text-gray-800">
              ChestX-R<span className="text-[rgb(70,65,180)]">ai</span>
            </span>
          </div>

          {/* MENÜ - DİKKAT: Burada <a> yerine <button> kullandık ki ışınlanma olmasın */}
          <div className="hidden md:flex items-center space-x-8 font-medium text-gray-600">
            <button onClick={() => scrollToSection('details')} className="hover:text-[rgb(70,65,180)] transition-colors">Proje Hakkında</button>
            <button onClick={() => scrollToSection('about')} className="hover:text-[rgb(70,65,180)] transition-colors">Biz Kimiz</button>

            {/* GITHUB LINKI */}
            <a 
              href="https://github.com/turkerklc/ChestX-Rai" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-900 transition-colors text-xl"
            >
              <FaGithub />
            </a>

            {/* ANALİZ BUTONU */}
            <button 
              onClick={() => scrollToSection('analyzer')}
              className="px-6 py-2.5 bg-[rgb(70,65,180)] hover:bg-[#38339acc] text-white rounded-full shadow-lg shadow-indigo-200 hover:shadow-indigo-300 hover:-translate-y-0.5 transition-all duration-300 flex items-center gap-2"
            >
              <span>Analiz Başlat</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path></svg>
            </button>
          </div>
        </div>
      </nav>

      {/* ANALYZER SECTION */}
      <section id="analyzer" className="pt-32 pb-20 px-4">
        {/* Kutu Genişliği */}
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-10">
            <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-4">
              Yapay Zeka Destekli <span className="text-red-600">Röntgen Analizi</span>
            </h1>
            <p className="text-lg text-gray-500 max-w-2xl mx-auto">
              Saniyeler içinde 14 farklı akciğer hastalığını tespit edin ve xAI teknolojisi ile görsel kanıtları inceleyin.
            </p>
          </div>

          <div className="grid lg:grid-cols-12 gap-6 bg-white p-2 rounded-3xl shadow-2xl border border-gray-100 overflow-hidden">
            {/* SOL PANEL (YÜKLEME) */}
            <div className="lg:col-span-5 p-6 bg-blue-50/50 flex flex-col justify-center">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FaCloudUploadAlt className="text-blue-600" /> Görüntü Yükle
              </h2>
              <div className="border-3 border-dashed border-blue-200 rounded-2xl bg-white p-4 text-center hover:border-blue-400 transition-all cursor-pointer relative group h-64 flex flex-col justify-center items-center">
                <input type="file" id="file-upload" accept=".jpg, .jpeg, .png" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="max-h-full max-w-full rounded-lg shadow-sm object-contain" />
                ) : (
                  <div className="group-hover:scale-105 transition-transform duration-300">
                    <div className="bg-blue-100 p-3 rounded-full inline-block mb-3">
                      <FaCloudUploadAlt className="text-3xl text-blue-600" />
                    </div>
                    <p className="text-gray-500 font-medium text-sm">Dosyayı buraya sürükleyin</p>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3 mt-4">
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase">Yaş</label>
                  <input type="number" value={age} onChange={handleAgeChange} min="0" max="150" className="w-full mt-1 p-2.5 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none text-sm" />
                </div>
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase">Cinsiyet</label>
                  <select value={gender} onChange={(e) => setGender(e.target.value)} className="w-full mt-1 p-2.5 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none text-sm">
                    <option value="M">Erkek</option>
                    <option value="F">Kadın</option>
                    <option value="Other">Diğer</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3 mt-5">
                <button
                  onClick={handleAnalyze}
                  disabled={loading || !selectedFile}
                  className={`flex-1 py-3 rounded-xl font-bold text-white shadow-md transition-all transform active:scale-95 text-sm ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 hover:-translate-y-0.5'
                    }`}
                >
                  {loading ? 'Analiz Ediliyor...' : 'Analizi Başlat'}
                </button>

                {(selectedFile || predictions) && (
                  <button
                    onClick={handleReset}
                    className="px-4 py-3 rounded-xl font-bold text-white bg-red-600 hover:bg-red-700 hover:-translate-y-0.5 transition-all active:scale-95 border border-gray-300 text-sm"
                  >
                    Sıfırla
                  </button>
                )}
              </div>
            </div>

            {/* SAĞ PANEL (SONUÇLAR) */}
            <div className="lg:col-span-7 p-6 flex flex-col justify-center min-h-[400px]" ref={resultsRef}>
              {!predictions && !loading && (
                <div className="text-center text-gray-400">
                  <FaArrowRight className="text-3xl opacity-20 mx-auto mb-4" />
                  <h3 className="text-lg font-medium">Sonuçlar burada görünecek</h3>
                </div>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center py-10">
                  <FaSpinner className="text-4xl text-blue-600 animate-spin" />
                  <p className="text-gray-400 mt-3 animate-pulse text-sm">Yapay zeka analiz ediyor...</p>
                </div>
              )}

              {predictions && (
                <div className="grid md:grid-cols-2 gap-6 animate-fade-in">
                  
                  {/* SONUÇ LİSTESİ */}
                  {Object.keys(predictions)[0] === 'No Finding' ? (
                    <div className="bg-green-50 border border-green-200 rounded-xl p-6 flex flex-col items-center justify-center text-center shadow-sm h-full">
                      <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-3">
                        <FaCheckCircle className="text-4xl text-green-600" />
                      </div>
                      <h3 className="text-xl font-bold text-green-800 mb-2">Bulgu Rastlanmadı</h3>
                      <p className="text-green-700 mb-4 text-xs leading-relaxed">
                        Yapay zeka analizi sonucunda bu görüntüde %{(predictions['No Finding'] * 100).toFixed(1)} oranında patolojik bir bulguya rastlanmamıştır.
                      </p>
                      <div className="w-full bg-white rounded-lg p-3 border border-green-100 text-[10px] text-gray-500">
                        <span className="font-bold block mb-1 text-green-700">Önemli Not:</span>
                        Bu sonuç kesin bir tıbbi teşhis değildir.
                      </div>
                    </div>
                  ) : (
                    <div>
                      <h3 className="text-base font-bold text-gray-800 mb-3 flex items-center gap-2">
                        Tespit Edilen Bulgular
                        <span className="text-[10px] font-normal text-gray-400">(Detay için tıklayın)</span>
                      </h3>

                      <div className="space-y-2">
                        {Object.entries(predictions)
                          .filter(([label]) => label !== 'No Finding')
                          .slice(0, 4)
                          .map(([label, score]) => (
                            <div
                              key={label}
                              className={`bg-white rounded-lg border transition-all duration-200 overflow-hidden ${expandedDisease === label ? 'border-blue-400 shadow-md ring-1 ring-blue-100' : 'border-gray-100 shadow-sm hover:border-blue-200'
                                }`}
                            >
                              <div
                                onClick={() => toggleDescription(label)}
                                className="p-2.5 cursor-pointer"
                              >
                                <div className="flex justify-between items-center mb-1.5">
                                  <span className="font-medium text-gray-700 flex items-center gap-2 text-sm">
                                    {label}
                                  </span>

                                  <div className="flex items-center gap-2">
                                    <span className={`font-bold text-sm ${score > 0.5 ? 'text-red-600' : 'text-blue-600'}`}>
                                      {(score * 100).toFixed(1)}%
                                    </span>
                                    {expandedDisease === label ?
                                      <FaChevronUp className="text-gray-400 text-[10px]" /> :
                                      <FaChevronDown className="text-gray-400 text-[10px]" />
                                    }
                                  </div>
                                </div>

                                <div className="w-full bg-gray-100 rounded-full h-1.5">
                                  <div
                                    className={`h-full rounded-full transition-all duration-500 ${score > 0.5 ? 'bg-red-500' : 'bg-blue-600'}`}
                                    style={{ width: `${score * 100}%` }}
                                  ></div>
                                </div>
                              </div>

                              {expandedDisease === label && (
                                <div className="bg-blue-50 px-3 py-2 text-[11px] text-gray-600 border-t border-blue-100 flex gap-2 items-start animate-fadeIn">
                                  <FaInfoCircle className="text-blue-500 mt-0.5 shrink-0" />
                                  <p>{DISEASE_DESCRIPTIONS[label] || "Detaylı açıklama bulunmuyor."}</p>
                                </div>
                              )}
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* xAI GÖRSEL */}
                  <div className="flex flex-col h-full">
                    <h3 className="text-base font-bold text-gray-800 mb-3">xAI Görsel Kanıt</h3>

                    {heatmapUrl ? (
                      <div className="flex flex-col gap-3">
                        <div className="relative group">
                          <img
                            src={heatmapUrl}
                            alt="xAI Heatmap"
                            className="w-full rounded-lg shadow-md border border-gray-200 object-contain max-h-64 bg-black"
                          />
                          <div className="absolute top-2 right-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded backdrop-blur-sm">
                            Grad-CAM
                          </div>
                        </div>
                        <button
                          onClick={handleDownloadHeatmap}
                          className="flex items-center justify-center gap-2 w-full py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold rounded-lg transition-all active:scale-95 border border-gray-200 text-xs"
                        >
                          <FaDownload className="text-blue-600" />
                          Görüntüyü İndir
                        </button>
                      </div>
                    ) : (
                      <div className="h-40 bg-gray-100 rounded-lg flex items-center justify-center text-gray-400 text-xs">
                        Isı haritası yüklenemedi.
                      </div>
                    )}
                  </div>

                </div>
              )}
            </div>

          </div>
        </div>
      </section>

      {/* DETAYLAR */}
      <section id="details" className="py-24 bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-10">Proje Hakkında</h2>
          <div className="grid md:grid-cols-3 gap-10">
            <div className="p-6 bg-gray-50 rounded-xl">
              <FaInfoCircle className="text-4xl text-blue-600 mx-auto mb-4" />
              <h3 className="font-bold">ResNet-50</h3>
              <p className="text-gray-600 text-sm leading-relaxed mt-2">
                ResNet-50, derin öğrenme dünyasının en güvenilir mimarilerinden biridir.
                Biz bu projede, bu mimariyi 112.000 adet göğüs röntgeni görüntüsüyle
                eğiterek, zatürre ve diğer akciğer hastalıklarını %90'a varan doğrulukla
                tespit edebilecek hale getirdik. Modelimiz, pikseller arasındaki en ince
                detayları bile yakalayabilir.
              </p>
            </div>
            <div className="p-6 bg-gray-50 rounded-xl">
              <FaMapMarked className="text-4xl text-blue-600 mx-auto mb-4" />
              <h3 className="font-bold">Grad-CAM</h3>
              <p className="text-gray-600 text-sm leading-relaxed mt-2">
                Grad-CAM, derin öğrenme modellerinin (özellikle de görüntü işleme modellerinin) nasıl karar verdiğini açıklamak için kullanılan tekniklerden biridir.
                Yapay zekanın verdiği kararı, girdi olarak aldığı görselin hangi bölgesine bakarak verdiğini göstermek için bir ısı haritası oluşturur. Bu x-AI'nın (Explainable AI)
                temel taşlarından biridir.
              </p>
            </div>
            <div className="p-6 bg-gray-50 rounded-xl">
              <FaExclamation className="text-4xl text-blue-600 mx-auto mb-4" />
              <h3 className="font-bold">Uyarı</h3>
              <p className="text-gray-600 text-sm leading-relaxed mt-2">
                Bu proje yalnızca eğitim ve akademik araştırma amaçlı geliştirilmiştir.
                Sunulan sonuçlar kesinlik taşımaz ve profesyonel bir tıbbi teşhis veya doktor muayenesi yerine geçmez.
                Elde edilen veriler tedavi amaçlı kullanılmamalıdır. Geliştiriciler, olası hatalı sonuçlardan veya bu sonuçlara dayanarak alınan kararlardan sorumlu tutulamaz.
                Herhangi bir sağlık sorununuzda lütfen uzman bir hekime başvurunuz.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* HAKKIMIZDA */}
      <section id="about" className="py-20 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Biz Kimiz</h2>
          <div className="flex justify-center gap-10">
            <div className="bg-white p-6 rounded-xl shadow-lg w-64"><h3 className="font-bold text-lg">Türker Kılıç</h3><p className="text-blue-500">AI Engineer</p></div>
            <div className="bg-white p-6 rounded-xl shadow-lg w-64"><h3 className="font-bold text-lg">Ferhat Köknar</h3><p className="text-blue-500">Frontend Developer</p></div>
          </div>
        </div>
      </section>

      <footer className="bg-gray-900 text-white py-8 text-center"><p>&copy; 2025 ChestX-Rai</p></footer>
    </div>
  );
}

export default App;