"""YOLO11m + ByteTrack ile araç tespiti, takibi, çizgi sayımı ve plaka tespiti.

COCO'daki car, motorcycle, bus, truck sınıfları tespit edilir ve ByteTrack ile takip edilir.
--line-start / --line-end çizgisinden geçen her araç (her ID bir kez) sayılır; köşede
toplam, türlere göre sayılar ve FPS gösterilir.
Her aracın kutusu kırpılıp models/plate_best.pt ile plaka aranır; bulunan plaka yeşil
kutu ve güven skoruyla çizilir. Her araç ID'si için en yüksek güvenli plaka kırpımı
outputs/plates/<ID>.jpg olarak kaydedilir.

Kullanım:
    python task2_vehicles.py --max-seconds 30
"""
import argparse
import glob
import os
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from task1_people import LabelDrawer, ThreadedWriter, draw_panel, load_font, threaded_reader

# COCO sınıf numarası -> ekranda gösterilen tür adı
VEHICLES = {2: "Otomobil", 3: "Motosiklet", 5: "Otobüs", 7: "Kamyon"}
PLATE_COLOR = (0, 255, 0)  # BGR yeşil


def parse_point(text):
    x, y = text.split(",")
    return int(x), int(y)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/arac.mp4")
    ap.add_argument("--output", default="outputs/test2.mp4")
    ap.add_argument("--model", default="yolo11m.pt",
                    help="Daha hızlı ama daha az doğru: yolo11s.pt veya yolo11n.pt")
    ap.add_argument("--plate-model", default="models/plate_best.pt")
    ap.add_argument("--plate-conf", type=float, default=0.25)
    ap.add_argument("--plate-imgsz", type=int, default=320,
                    help="Plaka modelinin giriş boyutu; araç kırpımları zaten küçük (~150-300 px), "
                         "320 hem ~10 kat hızlı hem 640'tan fazla plaka buluyor")
    ap.add_argument("--plates-dir", default="outputs/plates")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--line-start", type=parse_point, default=(0, 700),
                    help="Sayım çizgisinin başlangıcı, 'x,y' biçiminde")
    ap.add_argument("--line-end", type=parse_point, default=(1920, 700),
                    help="Sayım çizgisinin bitişi, 'x,y' biçiminde")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                    help="GPU'da FP16 çıkarım (daha hızlı)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Videonun yalnızca ilk N saniyesini işle")
    return ap


def clip_box(xyxy, w, h):
    x1, y1, x2, y2 = xyxy.astype(int)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def process(args, log=print):
    """Videoyu işler; her kare için bir durum sözlüğü üretir (task1_people.process ile aynı yapı)."""
    device = 0 if torch.cuda.is_available() else "cpu"
    quantize = 16 if args.half and device == 0 else None
    model = YOLO(args.model)
    plate_model = YOLO(args.plate_model)

    p1, p2 = tuple(args.line_start), tuple(args.line_end)
    line_zone = sv.LineZone(start=sv.Point(*p1), end=sv.Point(*p2),
                            triggering_anchors=(sv.Position.CENTER,))
    counted = {}  # takip ID'si -> tür adı (her araç bir kez sayılır)
    log(f"Sayım çizgisi: {p1} -> {p2}")

    os.makedirs(args.plates_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.plates_dir, "*.jpg")):  # önceki çalıştırmanın ID'leri
        os.remove(old)
    best_plate = {}  # takip ID'si -> kaydedilen en yüksek plaka güveni

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

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_drawer = LabelDrawer(load_font(20))
    font = load_font(40)
    small_font = load_font(24)
    frame_times = deque(maxlen=30)  # son karelerin zamanları (anlık FPS için)
    seen_ids = set()

    start = time.time()
    i = -1
    frames = threaded_reader(cap, total)
    try:
        for i, frame in enumerate(frames):
            r = model.track(frame, persist=True, tracker=args.tracker, classes=list(VEHICLES),
                            device=device, quantize=quantize, verbose=False)[0]
            det = sv.Detections.from_ultralytics(r)
            if det.tracker_id is None:
                det = sv.Detections.empty()
                det.tracker_id = np.empty(0, dtype=int)
            else:
                det = det[det.tracker_id != None]  # noqa: E711 (takipçinin henüz onaylamadığı kutular)
            seen_ids.update(det.tracker_id.tolist())

            # Plaka: her aracın kutusu kırpılıp tek seferde (batch) plaka modeline verilir
            boxes = [clip_box(b, w, h) for b in det.xyxy]
            idx = [j for j, (x1, y1, x2, y2) in enumerate(boxes) if x2 - x1 > 1 and y2 - y1 > 1]
            plates = []  # (x1, y1, x2, y2, güven) tam kare koordinatlarında
            if idx:
                crops = [frame[boxes[j][1]:boxes[j][3], boxes[j][0]:boxes[j][2]] for j in idx]
                results = plate_model.predict(crops, imgsz=args.plate_imgsz, conf=args.plate_conf,
                                              device=device, quantize=quantize, verbose=False)
                for j, pr in zip(idx, results):
                    if len(pr.boxes) == 0:
                        continue
                    k = int(pr.boxes.conf.argmax())  # bir araçta tek plaka: en güvenlisi
                    conf = float(pr.boxes.conf[k])
                    ox, oy = boxes[j][0], boxes[j][1]
                    px1, py1, px2, py2 = pr.boxes.xyxy[k].cpu().numpy().astype(int)
                    px1, py1, px2, py2 = clip_box(
                        np.array([px1 + ox, py1 + oy, px2 + ox, py2 + oy]), w, h)
                    if px2 - px1 < 2 or py2 - py1 < 2:
                        continue
                    plates.append((px1, py1, px2, py2, conf))
                    tid = int(det.tracker_id[j])
                    if conf > best_plate.get(tid, 0.0):
                        best_plate[tid] = conf
                        cv2.imwrite(os.path.join(args.plates_dir, f"{tid}.jpg"),
                                    frame[py1:py2, px1:px2])

            crossed_in, crossed_out = line_zone.trigger(det)
            for j in np.flatnonzero(crossed_in | crossed_out):
                tid = int(det.tracker_id[j])
                if tid not in counted:
                    counted[tid] = VEHICLES[int(det.class_id[j])]
                    log(f"Geçiş: #{tid} {counted[tid]} (kare {i}, {i / fps:.2f} sn)")

            labels = [f"ID {tid} · {VEHICLES[int(c)]} · {conf:.2f}"
                      for tid, c, conf in zip(det.tracker_id, det.class_id, det.confidence)]
            frame = box_annotator.annotate(frame, det)
            frame = label_drawer.annotate(frame, det, labels)
            for px1, py1, px2, py2, conf in plates:
                cv2.rectangle(frame, (px1, py1), (px2, py2), PLATE_COLOR, 2)
                cv2.putText(frame, f"{conf:.2f}", (px1, max(py1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (0, 255, 255), 4)
            for p in (p1, p2):
                cv2.circle(frame, p, 8, (0, 255, 255), -1)

            by_type = {name: 0 for name in VEHICLES.values()}
            for name in counted.values():
                by_type[name] += 1
            panel_h = draw_panel(frame, [(f"Toplam araç: {len(counted)}", (255, 255, 255))] + [
                (f"{name}: {n}", (230, 230, 120)) for name, n in by_type.items()], font)
            frame_times.append(time.time())
            live_fps = 0.0
            if len(frame_times) > 1:
                live_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                draw_panel(frame, [(f"FPS: {live_fps:.1f}", (255, 255, 255))], small_font,
                           pad=10, y_offset=panel_h + 10)
            writer.write(frame)

            yield {"i": i, "total": total, "frame": frame, "fps": live_fps,
                   "counts": (len(counted), by_type)}
    finally:  # durdurulsa da video dosyası düzgün kapansın
        frames.close()
        cap.release()
        writer.release()
    n = i + 1
    elapsed = time.time() - start
    by_type = {name: 0 for name in VEHICLES.values()}
    for name in counted.values():
        by_type[name] += 1
    yield {"done": True, "summary": {
        "frames": n,
        "elapsed": elapsed,
        "fps": n / elapsed if elapsed else 0.0,
        "ids": len(seen_ids),
        "total": len(counted),
        "by_type": by_type,
        "plates": {tid: round(c, 3) for tid, c in sorted(best_plate.items())},
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
        print(f"Toplam farklı takip ID'si: {s['ids']}")
        print(f"Çizgiden geçen araç: {s['total']}  (" + ", ".join(
            f"{k}: {v}" for k, v in s["by_type"].items()) + ")")
        print(f"Plakası kaydedilen araç: {len(s['plates'])} -> {args.plates_dir}")
        print(f"Kaydedildi: {s['output']}")


if __name__ == "__main__":
    main()
