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
