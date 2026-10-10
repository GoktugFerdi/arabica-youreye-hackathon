"""Videonun ilk karesinde kafenin içini çokgen olarak seçer ve configs/zone.json'a kaydeder.

Sol tık: köşe ekle    Sağ tık: son köşeyi geri al
r: sıfırla    Enter / s: kaydet ve çık    q / ESC: kaydetmeden çık
"""
import json
import os
import sys

import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "videos/kafe.mp4"
OUT = "configs/zone.json"
WINDOW = "pick_zone  (sol: kose | sag: geri al | r: sifirla | Enter: kaydet | q: cik)"

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"Video okunamadı: {VIDEO}")

# Referans olsun diye giriş çizgisini de göster
line = None
if os.path.exists("configs/line.json"):
    with open("configs/line.json", encoding="utf-8") as f:
        cfg = json.load(f)
    line = (tuple(cfg["start"]), tuple(cfg["end"]))

points = []


def redraw():
    img = frame.copy()
    if len(points) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(points)], (0, 200, 0))
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    if len(points) >= 2:
        cv2.polylines(img, [np.array(points)], len(points) >= 3, (0, 255, 0), 2)
    for p in points:
        cv2.circle(img, p, 6, (0, 0, 255), -1)
    if line:
        cv2.line(img, line[0], line[1], (0, 255, 255), 3)
    cv2.imshow(WINDOW, img)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Köşe {len(points)}: ({x}, {y})", flush=True)
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        print(f"Geri alındı: {points.pop()}", flush=True)
    else:
        return
    redraw()


cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)
cv2.setMouseCallback(WINDOW, on_mouse)
redraw()

saved = False
while True:
    key = cv2.waitKey(20) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord("r"):
        points.clear()
        print("Sıfırlandı", flush=True)
        redraw()
    if key in (13, ord("s")):
        if len(points) < 3:
            print("En az 3 köşe gerekli", flush=True)
            continue
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"video": VIDEO, "polygon": [list(p) for p in points]}, f, indent=2)
        saved = True
        break

cv2.destroyAllWindows()
if saved:
    print(f"Kaydedildi: {OUT} ({len(points)} köşe): {points}")
else:
    print("Kaydedilmedi")
