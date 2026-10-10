"""Kafe kamerası analiz arayüzü (Gradio).

Çalıştırma:
    .venv\\Scripts\\python app.py
Tarayıcıda http://127.0.0.1:7860 otomatik açılır.
"""
import glob
import os
import subprocess
import time

import cv2
import gradio as gr
import imageio_ffmpeg

from task1_people import build_parser, process

PREVIEW_WIDTH = 960  # tarayıcıya gönderilen önizleme genişliği
PREVIEW_EVERY = 2  # her kaç karede bir önizleme güncellensin

CSS = """
.stat-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;}
@media (max-width: 700px) {.stat-grid {grid-template-columns: repeat(2, 1fr);}}
.stat {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 16px;
       background: var(--background-fill-secondary);}
.stat .label {font-size: 13px; color: var(--body-text-color-subdued);}
.stat .value {font-size: 32px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.2;}
.stat.in .value {color: #16a34a;} .stat.out .value {color: #dc2626;}
.progress {height: 6px; border-radius: 3px; background: var(--background-fill-secondary);
           margin-top: 12px; overflow: hidden;}
.progress > div {height: 100%; background: var(--color-accent);}
.note {font-size: 13px; color: var(--body-text-color-subdued);}
"""


def stats_html(counts=(0, 0, 0), fps=0.0, done=0, total=0, status="Hazır"):
    pct = 100 * done / total if total else 0
    cards = [("Giriş", counts[0], "in"), ("Çıkış", counts[1], "out"),
             ("Toplam", counts[2], ""), ("FPS", f"{fps:.1f}", "")]
    html = '<div class="stat-grid">' + "".join(
        f'<div class="stat {cls}"><div class="label">{k}</div><div class="value">{v}</div></div>'
        for k, v, cls in cards) + "</div>"
    html += f'<div class="progress"><div style="width:{pct:.1f}%"></div></div>'
    html += f'<div class="note">{status} · kare {done}/{total} ({pct:.0f}%)</div>'
    return html


def list_videos():
    """videos/ içindeki girdi videoları; sonuç videoları ve tarayıcı kopyaları listelenmez."""
    files = sorted(glob.glob("videos/*.mp4") + glob.glob("videos/*.avi"))
    files = [f for f in files if "_Sonuç" not in f and not f.endswith("_web.mp4")]
    return sorted(files, key=lambda f: "kafe" not in f)  # kafe videosu önce


def to_browser_mp4(src):
    """mp4v kodlu çıktıyı tarayıcıların oynatabildiği H.264'e çevirir."""
    dst = src.replace(".mp4", "_web.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)
    return dst


def run(video_choice, upload, max_seconds, model, imgsz, augment):
    video = upload or video_choice
    if not video:
        raise gr.Error("Bir video seçin veya yükleyin.")

    args = build_parser().parse_args([])
    args.video = video
    args.model = model
    args.imgsz = int(imgsz)
    args.augment = augment
    args.max_seconds = max_seconds if max_seconds > 0 else None
    os.makedirs("outputs", exist_ok=True)
    args.output = f"outputs/arayuz_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    logs = []

    def log(msg):
        logs.append(msg)

    yield (None, stats_html(status="Modeller yükleniyor…"), "", None, None, None)
    for ev in process(args, log=log):
        if ev.get("done"):
            s = ev["summary"]
            break
        if ev["i"] % PREVIEW_EVERY:
            continue
        f = ev["frame"]
        h, w = f.shape[:2]
        preview = cv2.cvtColor(cv2.resize(f, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w))),
                               cv2.COLOR_BGR2RGB)
        yield (preview, stats_html(ev["counts"], ev["fps"], ev["i"] + 1, ev["total"], "İşleniyor"),
               "\n".join(logs[-200:]), gr.skip(), gr.skip(), gr.skip())

    yield (gr.skip(), stats_html(s["counts"], s["fps"], s["frames"], s["frames"],
                                 "Video tarayıcı için dönüştürülüyor…"),
           "\n".join(logs), None, None, None)
    web_video = to_browser_mp4(s["output"])

    crossings = [[c["id"], c["yön"], c["saniye"], c["cinsiyet"]] for c in s["crossings"]]
    details = [
        ["İşlenen kare", s["frames"]],
        ["Süre", f"{s['elapsed']:.1f} sn"],
        ["Ortalama FPS", f"{s['fps']:.1f}"],
        ["Kare başına ortalama kişi", f"{s['avg_people']:.1f}"],
        ["Sahne sayısı", len(s["scene_ids"])],
        ["Farklı takip ID'si", s["ids"]],
    ] + [[f"Süre · {k}", f"{v:.0f} ms/kare"] for k, v in s["timings_ms"].items()]
    yield (gr.skip(), stats_html(s["counts"], s["fps"], s["frames"], s["frames"], "Tamamlandı"),
           "\n".join(logs), web_video, crossings, details)


with gr.Blocks(title="Kafe Kamerası Analizi") as demo:
    gr.Markdown("# Kafe Kamerası Analizi\n"
                "Kişi tespiti, takip, kapıdan giriş/çıkış sayımı ve cinsiyet tahmini. "
                "Sayım çizgisi ve kafe bölgesi bu kamera için ayarlıdır "
                "(`configs/line.json`, `configs/zone.json`).")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            videos = list_videos()
            video_choice = gr.Dropdown(videos, value=videos[0] if videos else None,
                                       label="Video (videos/ klasörü)")
            upload = gr.File(label="…veya video yükle", file_types=["video"], type="filepath")
            max_seconds = gr.Slider(0, 120, value=30, step=5,
                                    label="İşlenecek süre (sn) · 0 = videonun tamamı")
            with gr.Accordion("Gelişmiş ayarlar", open=False):
                model = gr.Dropdown(
                    [("yolo11n.pt", "yolo11n.pt"), ("yolo11s.pt", "yolo11s.pt"),
                     ("yolo11m.pt", "models/yolo11m.pt")],
                    value="models/yolo11m.pt", label="YOLO modeli (küçük = hızlı, büyük = doğru)")
                imgsz = gr.Dropdown([960, 1280], value=1280, label="Tespit çözünürlüğü")
                augment = gr.Checkbox(True, label="Augment (karanlık köşedeki kişiler için; "
                                                  "daha yavaş)")
            with gr.Row():
                start = gr.Button("Başlat", variant="primary")
                stop = gr.Button("Durdur", variant="stop")
            gr.Markdown("Cinsiyet tahmini model çıktısıdır, hata payı içerir.",
                        elem_classes="note")
        with gr.Column(scale=2):
            stats = gr.HTML(stats_html())
            preview = gr.Image(label="Canlı önizleme", type="numpy", interactive=False)
    with gr.Tabs():
        with gr.Tab("Sonuç videosu"):
            result_video = gr.Video(label="İşaretli video", interactive=False)
        with gr.Tab("Giriş / Çıkış"):
            crossings = gr.Dataframe(headers=["ID", "Yön", "Saniye", "Cinsiyet"],
                                     interactive=False)
        with gr.Tab("Ayrıntılar"):
            details = gr.Dataframe(headers=["Ölçüm", "Değer"], interactive=False)
        with gr.Tab("Olay günlüğü"):
            log_box = gr.Textbox(lines=14, max_lines=14, show_label=False)

    run_event = start.click(run, [video_choice, upload, max_seconds, model, imgsz, augment],
                            [preview, stats, log_box, result_video, crossings, details])
    stop.click(None, cancels=[run_event])


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
