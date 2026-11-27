"""
Adds title, stage labels and footer to the Blender frames and writes the mp4.
  pip install pillow imageio imageio-ffmpeg numpy
  python compose_mp4.py              (uses ./frames)
  python compose_mp4.py frames_test  (preview frames)
"""
import sys, glob, os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "frames"
OUT = "day27_focsani_to_the_mountains.mp4" if FOLDER == "frames" else "day27_preview.mp4"
FPS = 30 if FOLDER == "frames" else 1
HOLD = 30 if FOLDER == "frames" else 0  # hold the last frame for 1 s

files = sorted(glob.glob(os.path.join(FOLDER, "*.png")))
if not files:
    sys.exit(f"No PNG frames in {FOLDER}")
W, H = Image.open(files[0]).size


def font(px, bold=True):
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, px)
        except OSError:
            pass
    return ImageFont.load_default()


k = W / 1080.0  # sizes follow your layout standard (26 / 14 / 8 pt at 100 dpi)
static = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(static)
d.text((0.06 * W, (1 - 0.93) * H), "Focșani to the Mountains", font=font(int(36 * k)),
       fill=(255, 255, 255, 255), anchor="ls")
d.text((0.06 * W, (1 - 0.895) * H), "From the plain to the Vrancea Carpathians",
       font=font(int(19 * k), False), fill=(0, 229, 255, 255), anchor="ls")
d.text((0.06 * W, (1 - 0.062) * H), "#30DayMapChallenge | Day 27: New tool | Sara AHARON",
       font=font(int(11 * k), False), fill=(138, 147, 176, 255), anchor="ls")
d.text((0.06 * W, (1 - 0.04) * H), "Data: SRTM 1 arc-second (USGS) via GEE | Buildings: Overture Maps (heights partly estimated) | 3D: Blender",
       font=font(int(11 * k), False), fill=(138, 147, 176, 255), anchor="ls")

# stage labels: (start, end, text, color) as fractions of the clip
STAGES = [
    (0.00, 0.30, "FOCȘANI · 55 m", (255, 64, 166)),
    (0.30, 0.62, "THE SUBCARPATHIAN HILLS", (170, 100, 255)),
    (0.66, 1.00, "VRANCEA MOUNTAINS · up to 1,775 m", (0, 229, 255)),
]
FADE = 0.04
lab_font = font(int(24 * k))


# ---------------- bloom (glow) ----------------
BLOOM_THRESH = 190   # was 150, so only the brightest edges glow
BLOOM_GAIN = 0.4     # was 0.8
BLOOM_R = 8 * k      # was 12 * k, a tighter halo


def bloom(img):
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    lum = a.max(axis=2)
    mask = np.clip((lum - BLOOM_THRESH) / (255.0 - BLOOM_THRESH), 0, 1)[..., None]
    bright = Image.fromarray((a * mask).astype(np.uint8))
    tight = np.asarray(bright.filter(ImageFilter.GaussianBlur(BLOOM_R))).astype(np.float32)
    wide = np.asarray(bright.filter(ImageFilter.GaussianBlur(BLOOM_R * 3))).astype(np.float32)
    out = a + BLOOM_GAIN * tight + 0.3 * BLOOM_GAIN * wide
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")
    
def label_layer(u):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for a, b, text, col in STAGES:
        if a <= u <= b:
            alpha = min(1.0, (u - a) / FADE, (b - u) / FADE if b < 1.0 else 1.0)
            ImageDraw.Draw(layer).text((0.06 * W, 0.90 * H), text, font=lab_font,
                                       fill=col + (int(255 * max(alpha, 0)),), anchor="ls")
    return layer


PREVIEW = FOLDER != "frames"  # preview mode saves PNGs instead of an mp4
writer = None if PREVIEW else imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
last = None
for i, f in enumerate(files):
    u = i / max(len(files) - 1, 1)
    im = bloom(Image.open(f))
    im = Image.alpha_composite(im, label_layer(u))
    im = Image.alpha_composite(im, static)
    if PREVIEW:
        im.convert("RGB").save(f"preview_{i}.png")
        print("saved", f"preview_{i}.png")
        continue
    last = np.array(im.convert("RGB"))
    last = last[: last.shape[0] // 2 * 2, : last.shape[1] // 2 * 2]  # yuv420p needs even sizes
    writer.append_data(last)
    if i % 30 == 0:
        print(f"{i + 1}/{len(files)}")
if not PREVIEW:
    for _ in range(HOLD):
        writer.append_data(last)
    writer.close()
    print("Saved", OUT)
