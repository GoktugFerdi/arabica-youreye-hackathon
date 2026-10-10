# Kafe Kamerası Analizi — Görev 1

Tavana monteli kafe kamerası görüntüsünde:

- **Kişi tespiti** — YOLO11m (yalnızca `person` sınıfı)
- **Takip** — ByteTrack ile her kişiye kalıcı bir ID
- **Giriş / çıkış sayımı** — kapı önündeki çizgiden geçenler; aynı ID iki kez sayılmaz
- **Cinsiyet tahmini** — open_clip ViT-L-14 ile zero-shot; ekranda yalnızca Kadın / Erkek / Belirsiz

## Sonuç

| | |
|---|---|
| Sonuç videosu | `outputs/sonuc_videosu.mp4` |
| Ekran görüntüleri | `outputs/ekran_*.jpg` |
| Sayım ve cinsiyet özeti | `outputs/sonuc_ozet.json` |

86 saniyelik videoda (7 farklı sahne) **6 giriş, 4 çıkış, toplam 10 kişi** sayıldı. Her geçiş
videodan kare kare kontrol edildi; kaçan veya yanlış sayılan geçiş bulunmadı.

> Cinsiyet tahmini model çıktısıdır, hata payı içerir. Yukarıdan bakan kamerada yüzler çoğu
> zaman görünmediği için tahmin kişinin tüm görünümüne dayanır.

## Çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Videoyu `videos/kafe.mp4` olarak koyun.

- **Arayüz:** `python app.py` → tarayıcıda http://127.0.0.1:7860 açılır. Video seçip **Başlat**'a
  basın; canlı önizleme, sayaçlar, giriş/çıkış tablosu ve sonuç videosu burada görünür.
- **Komut satırı:** `python task1_people.py --max-seconds 30` (tamamı için `--max-seconds`
  vermeyin). Tüm ayarlar için `python task1_people.py --help`.

Model ağırlıkları (YOLO, CLIP) ilk çalıştırmada otomatik indirilir. NVIDIA GPU'lu bir
bilgisayarda yaklaşık 6–8 FPS çalışır; laptop prize takılıyken daha hızlıdır.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `task1_people.py` | Tüm işleme hattı: tespit, takip, sayım, cinsiyet, çizim |
| `app.py` | Gradio arayüzü |
| `line.json` | Sayım çizgisi (kapı açıklığı) |
| `zone.json` | Kafe içi bölgesi; camın dışındaki yayalar sayılmaz |
| `pick_line.py`, `pick_zone.py` | Çizgiyi / bölgeyi fareyle yeniden seçmek için araçlar |
| `trackers/` | Takip ayarları |

## Nasıl çalışıyor

- **Karanlık bölgeler:** Tespitten önce kare CLAHE ile aydınlatılır, test-zamanı çoğaltma
  (`augment`) açıktır. Video normal parlaklıkta kaydedilir.
- **Kafe bölgesi:** Ayak noktası kafe zemininin dışında kalan kişiler (cam arkasındaki yayalar) atılır.
- **Sahne geçişleri:** Video farklı saatlerden birleştirilmiş sahnelerden oluşuyor; geçişte
  takip sıfırlanır, ID'ler sahneler arasında karışmaz.
- **Sayım:** Kişi çizginin öbür tarafında art arda 10 kare kalmadan sayılmaz (titreme ve
  kısa süreli örtülmeler sayılmaz). Çizgi yalnızca kapı açıklığını kapsar; kapının
  yanındaki tezgâhta çalışan kişi sayılmaz.
- **Cinsiyet:** Her ID için 5 karede bir, en fazla 10 kez tahmin yapılır ve ortalaması alınır;
  ortalama güven %60'ın altındaysa "Belirsiz" yazılır.

---

# Görev 2 — Araç Sayımı ve Plaka Tanıma

Trafik kamerası görüntüsünde:

- **Araç tespiti** — YOLO11m (COCO; yalnızca `car`, `motorcycle`, `bus`, `truck`)
- **Takip** — ByteTrack ile her araca kalıcı bir ID
- **Sayım** — çizgiden geçen her araç bir kez sayılır; köşede toplam, türlere göre sayılar ve FPS
- **Plaka tespiti** — her aracın kutusu kırpılıp kendi eğittiğimiz plaka modeliyle aranır
- **Plaka okuma** — EasyOCR; her araç için en iyi okuma kutunun altında gösterilir

## Sonuç

| | |
|---|---|
| Sonuç videosu | `outputs/task2_result.mp4` |
| Ekran görüntüleri | `outputs/screenshots/` |
| Geçiş tablosu | `outputs/task2_results.csv` |
| Araç başına en iyi plaka kırpımı | `outputs/plates/<ID>.jpg` |

~5 dakikalık videoda (9.184 kare) çizgiden **353 araç** geçti: 304 otomobil, 37 kamyon,
7 otobüs, 5 motosiklet. 684 araç için plaka bölgesi bulundu; çizgiyi geçen 353 aracın 156'sında
OCR bir metin okudu, ancak bu okumaların çoğu düşük güvenlidir (bkz. Kısıtlar). Tüm video
NVIDIA GPU'lu bir dizüstünde ortalama ~18 FPS işlendi.

## Çalıştırma

Kurulum Görev 1 ile aynıdır; ek olarak `pip install easyocr`. Videoyu `videos/arac.mp4`
olarak, plaka modelini `models/plate_best.pt` olarak koyun.

```bash
python task2_vehicles.py                    # tüm video -> outputs/task2_result.mp4
python task2_vehicles.py --max-seconds 30   # yalnızca ilk 30 saniye
python task2_vehicles.py --line-start 0,700 --line-end 1920,700   # sayım çizgisi
python task2_vehicles.py --model yolo11s.pt # daha hızlı, biraz daha az doğru
```

Tüm ayarlar için `python task2_vehicles.py --help`. Çizgi koordinatlarını kare üzerinde
tıklayarak almak için: `python pick_line.py videos/arac.mp4`.

## Kullanılan modeller

| Aşama | Model |
|---|---|
| Araç tespiti | YOLO11m, COCO üzerinde önceden eğitilmiş (araç sınıfları) |
| Plaka tespiti | Kendi eğittiğimiz YOLO11m plaka modeli — Roboflow Türk plaka veri seti, 5.256 görsel, Google Colab T4 GPU ile eğitildi (`models/plate_best.pt`) |
| Takip | ByteTrack |
| Plaka okuma | EasyOCR (yalnızca büyük harf ve rakam) |

## Nasıl çalışıyor

- **Plaka tespiti:** Araç kırpımları küçük olduğu için plaka modeli 320 px girişle çalışır
  (640'a göre çok daha hızlı ve daha fazla plaka buluyor), güven eşiği 0.25.
- **Plaka okuma:** Her araç için en fazla 5 kez, o ana kadarki en geniş plaka kırpımı (araç
  yaklaştıkça plaka büyür) gri tona çevrilip 3 kat büyütülerek okunur. 40 px'ten dar
  plakalar okunmaz. Sonuçtan yalnızca boşluklar ve harf/rakam dışı karakterler silinir;
  en yüksek OCR güvenli okuma o aracın plakası olur.
- **CSV:** Çizgiyi geçen her araç için geçiş saniyesi, ID, tür, araç güveni, plaka ve OCR
  güveni yazılır. Plaka, araç çizgiyi geçtikten sonra okunsa da tabloya girer.

## Kısıtlar

- Plaka modeli Türk plakalarıyla eğitildi, test videosunda ise yabancı plakalar var. Model
  plaka bölgelerini yine de tespit ediyor, ancak OCR doğruluğu düşebilir.
- Bulanık ve uzak araçlarda okuma hataları olabilir; bu videoda plakalar karede yalnızca
  ~40–55 × 12–15 piksel olduğu için okumaların çoğu güvenilir değildir.
- YOLO11m, YOLO11n'ye göre daha doğru ama daha yavaştır; daha zayıf donanımda FPS düşebilir.

## Geliştirme önerisi

- Plaka modelini çok ülkeli bir plaka veri setiyle ek eğitimden geçirmek.
