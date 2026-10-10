"""YOLO11 + ByteTrack ile kişi tespiti, takibi ve giriş çizgisinden geçenlerin sayımı.

Tespit, karanlık bölgeleri açılmış (CLAHE) bir kopya üzerinde yapılır;
kutular orijinal kareye çizilir, yani kaydedilen video normal parlaklıktadır.
zone.json varsa ayak noktası (kutunun alt-ortası) bu çokgenin dışında kalan
kişiler (ör. camın dışındaki yayalar) atılır.
Video farklı saatlerden birleştirilmiş sahnelerden oluşuyor; sahne geçişinde
takipçi sıfırlanır ki yeni sahnedeki kişi, eski sahnede aynı yerde oturanın ID'sini almasın.
line.json'daki çizgiden ayak noktası geçen kişiler sayılır: kapıdan içeri (aşağı)
yürüyen Giriş, dışarı (yukarı) yürüyen Çıkış. Bir ID her yönde en fazla bir kez,
Toplam'da (çizgiden geçen farklı kişi) da en fazla bir kez sayılır.
Cinsiyet: kamera yukarıdan baktığı için yüz yerine kişinin tüm kutusu (kenar paylı)
open_clip ViT-L-14 ile erkek/kadın metin gruplarına göre zero-shot sınıflandırılır
(her grup birkaç metnin ortalama vektörü). Aynı görüntü vektörü başörtüsü / şapka /
açık baş gruplarıyla da karşılaştırılır; başörtüsü olasılığı eşiği geçen kişi Kadın
sayılır (şapka ayrı bir sınıf olduğu için şapkalı erkekler başörtülü sanılmaz).
Her ID için 5 karede bir, en fazla 10 kez tahmin yapılır ve olasılıkların ortalaması
tutulur; ortalama güven %60'ın altındaysa "Belirsiz" yazılır.

Kullanım:
    python task1_people.py --max-seconds 30
"""
import argparse
import json
import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import open_clip
import supervision as sv
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PERSON = 0
FONT_CANDIDATES = ("C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf")
GENDER_NAMES = ("Erkek", "Kadın")
DISCLAIMER = "Cinsiyet tahmini model çıktısıdır, hata payı içerir."


def find_font_path():
    for name in FONT_CANDIDATES:
        try:
            ImageFont.truetype(name, 10)
            return name
        except OSError:
            pass
    return None


def load_font(size):
    # cv2.putText Türkçe karakterleri (ş, ç, ı) çizemediği için PIL kullanılıyor
    path = find_font_path()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def draw_panel(frame, rows, font, corner="tr", margin=20, pad=18, gap=10, y_offset=0):
    """Köşeye yarı saydam bir panel içinde (metin, renk) satırları çizer.

    corner: "tr" sağ üst, "tl" sol üst, "bl" sol alt. Panelin yüksekliğini döndürür.
    """
    sizes = [font.getbbox(text) for text, _ in rows]
    line_h = max(b[3] for b in sizes)
    pw = max(b[2] for b in sizes) + 2 * pad
    ph = len(rows) * line_h + (len(rows) - 1) * gap + 2 * pad
    x0 = frame.shape[1] - pw - margin if corner == "tr" else margin
    y0 = frame.shape[0] - ph - margin if corner == "bl" else margin + y_offset

    roi = frame[y0:y0 + ph, x0:x0 + pw]
    roi[:] = (roi * 0.35).astype(np.uint8)  # arka planı karart
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for j, (text, color) in enumerate(rows):
        draw.text((pad, pad + j * (line_h + gap)), text, font=font, fill=color)
    roi[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return ph


class GenderEstimator:
    """open_clip ile kişi kırpıntısını zero-shot sınıflandırır: cinsiyet ve baş örtüsü."""

    # Her sınıf, metinlerinin ortalama vektörüyle temsil edilir (prompt ensemble).
    GENDER_PROMPTS = (  # GENDER_NAMES ile aynı sırada
        ("a photo of a man", "a man sitting in a cafe", "a photo of a man seen from above",
         "a security camera photo of a man"),
        ("a photo of a woman", "a woman sitting in a cafe", "a photo of a woman seen from above",
         "a security camera photo of a woman"),
    )
    HEAD_PROMPTS = (  # 0: başörtüsü, 1: şapka/bere, 2: açık baş, 3: kapüşon
        ("a photo of a woman wearing a hijab", "a person wearing a headscarf",
         "a head covered with a headscarf"),
        ("a person wearing a baseball cap", "a person wearing a hat", "a person wearing a beanie"),
        ("a person with uncovered hair", "a person with short hair", "a person with long hair",
         "a bald person"),
        ("a person wearing a hoodie with the hood up", "a person wearing a hooded sweatshirt",
         "a person with a hood over their head"),
    )

    def __init__(self, arch, pretrained, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(arch)
        self.gender_text = self._class_embeds(tokenizer, self.GENDER_PROMPTS)
        self.head_text = self._class_embeds(tokenizer, self.HEAD_PROMPTS)

    @torch.no_grad()
    def _class_embeds(self, tokenizer, classes):
        out = []
        for prompts in classes:
            e = self.model.encode_text(tokenizer(list(prompts)).to(self.device))
            e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
            out.append(e / e.norm())
        return torch.stack(out)

    @torch.no_grad()
    def predict(self, crops):
        """BGR kırpıntı listesi -> ((N, 2) [erkek, kadın], (N, 4) [başörtüsü, şapka, açık baş, kapüşon])."""
        batch = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops
        ]).to(self.device)
        img = self.model.encode_image(batch)
        img = img / img.norm(dim=-1, keepdim=True)
        scale = self.model.logit_scale.exp()
        gender = (scale * img @ self.gender_text.T).softmax(dim=-1)
        head = (scale * img @ self.head_text.T).softmax(dim=-1)
        return gender.cpu().numpy(), head.cpu().numpy()


_GENDER_CACHE = {}


def get_gender_estimator(arch, pretrained, device):
    """CLIP modelini bir kez yükler; arayüzde ardışık çalıştırmalarda yeniden kullanılır."""
    key = (arch, pretrained, device)
    if key not in _GENDER_CACHE:
        _GENDER_CACHE[key] = GenderEstimator(arch, pretrained, device)
    return _GENDER_CACHE[key]


class LabelDrawer:
    """Türkçe karakterli etiketleri PIL ile bir kez çizip önbellekte tutar.

    sv.RichLabelAnnotator her karede tüm kareyi PIL'e çevirdiği için yavaştı (~20 ms);
    burada yalnızca küçük etiket görüntüleri yapıştırılır.
    """

    def __init__(self, font, palette=sv.ColorPalette.DEFAULT, pad=6):
        self.font, self.palette, self.pad = font, palette, pad
        self.cache = {}

    def _render(self, text, color):
        key = (text, color.as_bgr())
        if key not in self.cache:
            l, t, r, b = self.font.getbbox(text)
            img = Image.new("RGB", (r + 2 * self.pad, b - t + 2 * self.pad), color.as_rgb())
            ImageDraw.Draw(img).text((self.pad, self.pad - t), text, font=self.font, fill="white")
            self.cache[key] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return self.cache[key]

    def annotate(self, frame, det, labels):
        H, W = frame.shape[:2]
        for (x1, y1, x2, y2), tid, text in zip(det.xyxy.astype(int), det.tracker_id, labels):
            img = self._render(text, self.palette.by_idx(int(tid)))
            h, w = img.shape[:2]
            if w > W or h > H:
                continue
            x = min(max(x1, 0), W - w)
            y = y1 - h if y1 - h >= 0 else min(max(y1, 0), H - h)  # kare dışına taşmasın
            frame[y:y + h, x:x + w] = img
        return frame


def threaded_reader(cap, total, maxsize=8):
    """Kareleri arka planda okur; GPU çalışırken disk/çözme beklenmez."""
    q = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def run():
        for _ in range(total):
            if stop.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                break
            q.put(frame)
        q.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        while (frame := q.get()) is not None:
            yield frame
    finally:  # erken kapatılırsa iş parçacığını durdur; cap.release() ondan sonra güvenli
        stop.set()
        while thread.is_alive():
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


class ThreadedWriter:
    """Kareleri arka planda video dosyasına yazar."""

    def __init__(self, writer, maxsize=8):
        self.writer, self.q = writer, queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (frame := self.q.get()) is not None:
            self.writer.write(frame)

    def write(self, frame):
        self.q.put(frame)

    def release(self):
        self.q.put(None)
        self.thread.join()
        self.writer.release()


def drop_contained(det, thresh):
    """Alanının `thresh` oranı daha büyük bir kutunun içinde kalan kutuları atar.

    augment ile aynı kişinin üst gövdesi veya bacakları için ayrı, iç içe kutular
    çıkabiliyor; NMS bunları yakalamaz çünkü IoU'ları düşüktür.
    """
    if len(det) < 2:
        return det
    xyxy = det.xyxy
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = np.ones(len(det), dtype=bool)
    for i in range(len(det)):
        for j in range(len(det)):
            if i == j or area[i] >= area[j]:
                continue
            iw = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0])
            ih = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
            if iw > 0 and ih > 0 and iw * ih / area[i] >= thresh:
                keep[i] = False
                break
    return det[keep]


def crop_with_margin(frame, xyxy, margin):
    x1, y1, x2, y2 = xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return frame[y1:y2, x1:x2]


def enhance(frame, clahe):
    """Karanlık bölgeleri aydınlatır; parlak bölgeleri (pencere) fazla patlatmaz."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def small_gray(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)


def reset_tracker(model):
    """Tüm izleri siler ama ID sayacını korur (yeni sahnede ID'ler tekrar kullanılmaz)."""
    tracker = model.predictor.trackers[0]
    ids = tracker._ids
    tracker.reset()
    tracker._ids = ids


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/kafe.mp4")
    ap.add_argument("--output", default="outputs/test1.mp4")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="NMS IoU eşiği; düşük değer çift kutuları azaltır")
    ap.add_argument("--tracker", default="trackers/bytetrack_cafe.yaml",
                    help="Alternatif: trackers/botsort_cafe.yaml (ReID'li); ReID YOLO'nun iç "
                         "özelliklerini kullandığı için yalnızca --no-augment ile çalışır")
    ap.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=True,
                    help="Tespitten önce karanlık bölgeleri aydınlat")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="Test-zamanı çoğaltma (çevirme + çoklu ölçek); karanlık köşedeki, "
                         "sandalye arkasında kalan kişilerin tespit güvenini artırır")
    ap.add_argument("--zone", default="zone.json",
                    help="Kafe içi çokgeni; boş bırakılırsa ('') filtre uygulanmaz")
    ap.add_argument("--line", default="line.json", help="Sayım çizgisi (start/end)")
    ap.add_argument("--min-cross-frames", type=int, default=10,
                    help="Sayılmak için çizginin öbür tarafında art arda kalınması gereken kare")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                    help="GPU'da FP16 çıkarım (daha hızlı)")
    ap.add_argument("--cut-thresh", type=float, default=10.0,
                    help="Ardışık kareler arası ortalama fark bunu aşarsa sahne geçişi sayılır "
                         "(normal kareler ~1-3, geçişler 16+)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Videonun yalnızca ilk N saniyesini işle")
    ap.add_argument("--clip-arch", default="ViT-L-14")
    ap.add_argument("--clip-pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--scarf-thresh", type=float, default=0.8,
                    help="Ortalama başörtüsü olasılığı bunu geçerse kişi Kadın sayılır. Kısa gri "
                         "saç ~0.65-0.7 alabildiği için yüksek tutuluyor; başörtülü kadınları "
                         "eşik altında kalsa da ana cinsiyet modeli zaten Kadın olarak bulur")
    ap.add_argument("--contain-thresh", type=float, default=0.8,
                    help="Bir kutunun alanının bu oranı daha büyük bir kutunun içindeyse "
                         "aynı kişinin kısmi kutusu sayılıp atılır")
    ap.add_argument("--gender-every", type=int, default=5,
                    help="Her ID için kaç karede bir cinsiyet tahmini yapılacağı")
    ap.add_argument("--gender-max", type=int, default=10,
                    help="Her ID için en fazla kaç tahmin yapılacağı")
    ap.add_argument("--gender-min-conf", type=float, default=0.60,
                    help="Ortalama güven bunun altındaysa 'Belirsiz' yazılır")
    ap.add_argument("--crop-margin", type=float, default=0.10,
                    help="Kırpılan kutuya her kenardan eklenecek pay (kutu boyutuna oranla)")
    return ap


def process(args, log=print):
    """Videoyu işler; her kare için bir durum sözlüğü üretir (arayüz canlı önizleme için kullanır).

    Ara kareler: {"i", "total", "frame" (işaretli BGR kare), "counts", "fps"}.
    Son öğe: {"done": True, "summary": {...}} — sayım, cinsiyet ve aşama süreleri.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gender = get_gender_estimator(args.clip_arch, args.clip_pretrained,
                                  "cuda" if device == 0 else "cpu")
    gender_sum = {}  # takip ID'si -> olasılık toplamı [erkek, kadın]
    head_sum = {}  # takip ID'si -> olasılık toplamı [başörtüsü, şapka, açık baş, kapüşon]
    gender_n = {}  # takip ID'si -> yapılan tahmin sayısı

    zone = None
    if args.zone and os.path.exists(args.zone):
        with open(args.zone, encoding="utf-8") as f:
            polygon = np.array(json.load(f)["polygon"])
        zone = sv.PolygonZone(polygon, triggering_anchors=(sv.Position.BOTTOM_CENTER,))
        log(f"Bölge filtresi: {args.zone} ({len(polygon)} köşe)")

    with open(args.line, encoding="utf-8") as f:
        line_cfg = json.load(f)
    p1, p2 = tuple(line_cfg["start"]), tuple(line_cfg["end"])
    # LineZone'da "in", çizgi yönünün soluna geçiştir. Çizgiyi sağdan sola verince
    # "in" = aşağı doğru yürümek = kapıdan kafeye girmek olur.
    left, right = sorted([p1, p2])
    line_zone = sv.LineZone(start=sv.Point(*right), end=sv.Point(*left),
                            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
                            minimum_crossing_threshold=args.min_cross_frames)
    counted_in, counted_out = set(), set()
    log(f"Sayım çizgisi: {p1} -> {p2}")
    font = load_font(56)
    small_font = load_font(24)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_seconds is not None:
        total = min(total, int(args.max_seconds * fps))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = ThreadedWriter(
        cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)))
    # YOLO kareyi zaten imgsz'ye küçültüyor; aydınlatmayı o boyutta yapmak hem hızlı hem
    # modelin gördüğü görüntüyü değiştirmiyor. Kutular sonra orijinal boyuta ölçeklenir.
    det_scale = min(1.0, args.imgsz / max(w, h))
    det_size = (round(w * det_scale), round(h * det_scale))

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_drawer = LabelDrawer(load_font(20))

    id_frames = {}  # takip ID'si -> görüldüğü kare sayısı
    scene_ids = [set()]  # her sahnede görülen ID'ler
    n_boxes = 0
    prev_gray = None
    frame_times = deque(maxlen=30)  # son karelerin zamanları (anlık FPS için)

    def gender_label(tid):
        if tid not in gender_n:
            return "Belirsiz"
        # Baş örtüsü/kapüşon sınıfları yalnızca arka planda doğruluk için kullanılır;
        # ekranda sadece cinsiyet ve güven gösterilir.
        scarf = head_sum[tid][0] / gender_n[tid]
        if scarf >= args.scarf_thresh:
            return f"Kadın %{scarf * 100:.0f}"
        avg = gender_sum[tid] / gender_n[tid]
        k = int(avg.argmax())
        if avg[k] < args.gender_min_conf:
            return "Belirsiz"
        return f"{GENDER_NAMES[k]} %{avg[k] * 100:.0f}"

    timings = {}  # aşama adı -> toplam süre (sn)
    stage_start = [0.0]

    def tick(name):
        """Önceki tick'ten bu yana geçen süreyi `name` aşamasına yazar."""
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - stage_start[0]
        stage_start[0] = now

    events = []  # (ID, yön, kare)
    start = time.time()
    i = -1
    stage_start[0] = time.perf_counter()
    frames = threaded_reader(cap, total)
    try:
        for i, frame in enumerate(frames):
            tick("okuma")

            small = cv2.resize(frame, det_size, interpolation=cv2.INTER_AREA) \
                if det_scale < 1.0 else frame
            gray = small_gray(small)
            if prev_gray is not None and np.mean(np.abs(gray - prev_gray)) > args.cut_thresh:
                reset_tracker(model)
                scene_ids.append(set())
                log(f"Sahne geçişi: kare {i} ({i / fps:.2f} sn), takipçi sıfırlandı")
            prev_gray = gray
            source = enhance(small, clahe) if args.enhance else small
            tick("ön işleme")

            result = model.track(source, persist=True, tracker=args.tracker,
                                 classes=[PERSON], imgsz=args.imgsz, iou=args.iou,
                                 augment=args.augment,
                                 quantize=16 if args.half and device != "cpu" else None,
                                 device=device, verbose=False)[0]
            tick("tespit + takip")
            det = sv.Detections.from_ultralytics(result)
            if det.tracker_id is None:  # bu karede takip edilen kimse yok
                det = sv.Detections.empty()
                det.tracker_id = np.empty(0, dtype=int)
            if det_scale < 1.0:  # küçültülmüş karedeki kutuları orijinal koordinatlara taşı
                det.xyxy = det.xyxy / det_scale
            if zone is not None:
                det = det[zone.trigger(det)]
            det = drop_contained(det, args.contain_thresh)
            for tid in det.tracker_id.tolist():
                id_frames[tid] = id_frames.get(tid, 0) + 1
                scene_ids[-1].add(tid)
            n_boxes += len(det)

            # Cinsiyet: ID'nin 1., 6., 11. ... karesinde, en fazla gender_max kez
            due = [j for j, tid in enumerate(det.tracker_id.tolist())
                   if (id_frames[tid] - 1) % args.gender_every == 0
                   and gender_n.get(tid, 0) < args.gender_max]
            crops = [crop_with_margin(frame, det.xyxy[j], args.crop_margin) for j in due]
            due = [j for j, c in zip(due, crops) if c.size > 0]
            if due:
                g_probs, h_probs = gender.predict([c for c in crops if c.size > 0])
                for j, g, hd in zip(due, g_probs, h_probs):
                    tid = int(det.tracker_id[j])
                    gender_sum[tid] = gender_sum.get(tid, np.zeros(2)) + g
                    head_sum[tid] = head_sum.get(tid, np.zeros(len(hd))) + hd
                    gender_n[tid] = gender_n.get(tid, 0) + 1
            tick("cinsiyet")

            crossed_in, crossed_out = line_zone.trigger(det)
            for tid in det.tracker_id[crossed_in].tolist():
                if tid not in counted_in:
                    counted_in.add(tid)
                    events.append((tid, "Giriş", i))
                    log(f"Giriş: #{tid} (kare {i}, {i / fps:.2f} sn)")
            for tid in det.tracker_id[crossed_out].tolist():
                if tid not in counted_out:
                    counted_out.add(tid)
                    events.append((tid, "Çıkış", i))
                    log(f"Çıkış: #{tid} (kare {i}, {i / fps:.2f} sn)")

            if zone is not None:
                cv2.polylines(frame, [zone.polygon], True, (0, 255, 0), 1)
            labels = [f"ID {tid} · {gender_label(tid)}" for tid in det.tracker_id.tolist()]
            frame = box_annotator.annotate(frame, det)
            frame = label_drawer.annotate(frame, det, labels)
            cv2.line(frame, p1, p2, (0, 255, 255), 4)
            for p in (p1, p2):
                cv2.circle(frame, p, 8, (0, 255, 255), -1)
            panel_h = draw_panel(frame, [
                (f"Giriş: {len(counted_in)}", (80, 230, 80)),
                (f"Çıkış: {len(counted_out)}", (255, 90, 90)),
                (f"Toplam: {len(counted_in | counted_out)}", (255, 255, 255)),
            ], font)
            frame_times.append(time.time())
            live_fps = 0.0
            if len(frame_times) > 1:
                live_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                draw_panel(frame, [(f"FPS: {live_fps:.1f}", (255, 255, 255))], small_font,
                           pad=10, y_offset=panel_h + 10)
            draw_panel(frame, [(DISCLAIMER, (230, 230, 230))], small_font, corner="bl", pad=10)
            tick("çizim")
            writer.write(frame)
            tick("yazma")

            yield {"i": i, "total": total, "frame": frame, "fps": live_fps,
                   "counts": (len(counted_in), len(counted_out), len(counted_in | counted_out))}
            stage_start[0] = time.perf_counter()  # arayüzün harcadığı süre aşamalara yazılmasın
    finally:  # arayüzden durdurulsa da video dosyası düzgün kapansın
        frames.close()  # önce okuma iş parçacığı dursun
        cap.release()
        writer.release()
    n = i + 1
    elapsed = time.time() - start
    yield {"done": True, "summary": {
        "frames": n,
        "elapsed": elapsed,
        "fps": n / elapsed if elapsed else 0.0,
        "avg_people": n_boxes / n if n else 0.0,
        "ids": len(id_frames),
        "short_ids": sum(1 for c in id_frames.values() if c < fps),
        "scene_ids": [len(s) for s in scene_ids],
        "counts": (len(counted_in), len(counted_out), len(counted_in | counted_out)),
        "crossings": [{"id": tid, "yön": d, "kare": k, "saniye": round(k / fps, 2),
                       "cinsiyet": gender_label(tid), "tahmin": gender_n.get(tid, 0)}
                      for tid, d, k in events],
        "timings_ms": {k: v / n * 1000 for k, v in timings.items()} if n else {},
        "output": args.output,
    }}


def main():
    args = build_parser().parse_args()
    for ev in process(args, log=lambda m: print(m, flush=True)):
        if not ev.get("done"):
            if (ev["i"] + 1) % 100 == 0:
                print(f"{ev['i'] + 1}/{ev['total']} kare", flush=True)
            continue
        s = ev["summary"]
        print(f"İşlenen kare: {s['frames']}, süre: {s['elapsed']:.1f} sn ({s['fps']:.1f} FPS)")
        print(f"Kare başına ortalama kişi: {s['avg_people']:.1f}")
        print(f"Toplam farklı takip ID'si: {s['ids']} (1 sn'den kısa görünen: {s['short_ids']})")
        print("Sahne başına ID: " + ", ".join(map(str, s["scene_ids"])))
        print("Giriş: {}  Çıkış: {}  Toplam (farklı kişi): {}".format(*s["counts"]))
        for c in s["crossings"]:
            print(f"  ID {c['id']} ({c['yön']}, {c['saniye']} sn): {c['cinsiyet']} "
                  f"({c['tahmin']} tahmin)")
        print("Kare başına aşama süreleri: " + ", ".join(
            f"{k} {v:.0f} ms" for k, v in s["timings_ms"].items()))
        print(f"Kaydedildi: {s['output']}")


if __name__ == "__main__":
    main()
