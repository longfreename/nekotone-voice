"""Generate Voicekit's app icon: a minimalist soundwave mark on the app's
own dark/indigo gradient (matches styles.css --bg/--accent). No mascot,
no cat -- a professional, sleek glyph suitable for the app window, the
taskbar, and the installer.

Run from the repo root: python packaging/gen_icon.py
Writes app/src-tauri/icons/{32x32,128x128,128x128@2x,icon}.png/.ico and
app-icon-source.png (a large master copy, for future re-export).
"""
from PIL import Image, ImageDraw


def rounded_gradient_tile(size, radius_frac=0.22):
    """Dark-to-indigo diagonal gradient tile with rounded corners,
    matching the app's dark theme (--bg #0a1020 -> --accent #7a8cff)."""
    w = h = size
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    top_left = (10, 16, 32)       # #0a1020 --bg
    bottom_right = (122, 140, 255)  # #7a8cff --accent
    grad = Image.new("RGB", (w, h))
    px = grad.load()
    for y in range(h):
        for x in range(w):
            t = (x / (w - 1) + y / (h - 1)) / 2.0
            # ease so the accent only glows in, keeping most of the tile dark/sleek
            t = t ** 1.6
            r = int(top_left[0] + (bottom_right[0] - top_left[0]) * t)
            g = int(top_left[1] + (bottom_right[1] - top_left[1]) * t)
            b = int(top_left[2] + (bottom_right[2] - top_left[2]) * t)
            px[x, y] = (r, g, b)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    radius = int(w * radius_frac)
    mdraw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    base.paste(grad, (0, 0), mask)
    return base


def draw_waveform(img, size):
    """Five rounded bars of varying height, centered -- a simple,
    unambiguous "voice/audio" glyph, drawn crisp at any resolution."""
    draw = ImageDraw.Draw(img)
    n = 5
    # relative heights (middle tallest), as a fraction of the glyph box
    heights = [0.40, 0.68, 1.0, 0.68, 0.40]
    box = size * 0.62
    bar_w = box / (n * 1.8)
    gap = bar_w * 0.8
    total_w = n * bar_w + (n - 1) * gap
    x0 = (size - total_w) / 2
    cy = size / 2
    color = (255, 255, 255, 255)
    for i, hf in enumerate(heights):
        bh = box * hf
        x1 = x0 + i * (bar_w + gap)
        x2 = x1 + bar_w
        y1 = cy - bh / 2
        y2 = cy + bh / 2
        draw.rounded_rectangle([x1, y1, x2, y2], radius=bar_w / 2, fill=color)
    return img


def make(size):
    img = rounded_gradient_tile(size)
    return draw_waveform(img, size)


if __name__ == "__main__":
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "app", "src-tauri", "icons")
    os.makedirs(out_dir, exist_ok=True)

    master = make(1024)
    master.save(os.path.join(out_dir, "app-icon-source.png"))

    sizes = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "icon.png": 512,
    }
    for name, sz in sizes.items():
        master.resize((sz, sz), Image.LANCZOS).save(os.path.join(out_dir, name))

    # Multi-resolution .ico (Windows picks the best for each context:
    # taskbar, Explorer details, Alt+Tab, jumbo thumbnails).
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    master.save(
        os.path.join(out_dir, "icon.ico"),
        sizes=[(s, s) for s in ico_sizes],
    )

    # Small tray icon (used by Tauri's tray/notification area if enabled).
    master.resize((32, 32), Image.LANCZOS).save(os.path.join(out_dir, "tray.png"))

    print("wrote icons to", out_dir)
