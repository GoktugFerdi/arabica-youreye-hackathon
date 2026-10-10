"""Videonun ilk karesinde iki nokta seçip aralarına çizgi çizer.

Sol tık: nokta ekle (2 nokta seçilince çizgi çizilir)
r: sıfırla    q / ESC: çık
"""
import sys

import cv2

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "videos/kafe.mp4"
WINDOW = "pick_line  (sol tik: nokta | r: sifirla | q: cik)"

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"Video okunamadı: {VIDEO}")

points = []


def redraw():
    img = frame.copy()
    for p in points:
        cv2.circle(img, p, 6, (0, 0, 255), -1)
    if len(points) == 2:
        cv2.line(img, points[0], points[1], (0, 255, 255), 3)
    cv2.imshow(WINDOW, img)


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(points) == 2:
        points.clear()
    points.append((x, y))
    print(f"Nokta {len(points)}: ({x}, {y})", flush=True)
    if len(points) == 2:
        print(f"Çizgi: {points[0]} -> {points[1]}", flush=True)
    redraw()


# WINDOW_NORMAL: pencere ekrana sığacak şekilde küçültülebilir,
# tıklanan koordinatlar yine orijinal kare piksellerinde gelir.
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)
cv2.setMouseCallback(WINDOW, on_mouse)
redraw()

while True:
    key = cv2.waitKey(20) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord("r"):
        points.clear()
        print("Sıfırlandı", flush=True)
        redraw()

cv2.destroyAllWindows()
if len(points) == 2:
    print(f"Son çizgi: {points[0]} -> {points[1]}")
