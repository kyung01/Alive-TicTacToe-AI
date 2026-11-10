"""YouTube thumbnail: the generated art (ChatGPT/thumb-impossible.png) cropped to 16:9 with the headline set in a real font.

    python tools/make_thumbnail.py   -> presentation/video/export/thumbnail_1280x720.jpg (+ a 1920x1080 PNG)

The headline is drawn here rather than by the image model, so the letters are crisp and spelled right.
"""
import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ChatGPT", "thumb-impossible.png")   # the art (the prompt file keeps its first name)
OUT = os.path.join(ROOT, "presentation", "video", "export")
FONT = r"C:\Windows\Fonts\seguibl.ttf"   # Segoe UI Black
W, H = 1920, 1080


def fit(text, max_w, start):
    size = start
    while size > 40:
        f = ImageFont.truetype(FONT, size)
        l, t, r, b = f.getbbox(text)
        if r - l <= max_w:
            return f
        size -= 4
    return ImageFont.truetype(FONT, size)


def layer_text(text, font, fill, stroke, sw):
    l, t, r, b = font.getbbox(text, stroke_width=sw)
    im = Image.new("RGBA", (r - l + 40, b - t + 40), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((20 - l, 20 - t), text, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)
    return im


def shadowed(im, off=(10, 14), blur=10, alpha=170):
    sh = Image.new("RGBA", (im.width + 80, im.height + 80), (0, 0, 0, 0))
    a = im.split()[3].point(lambda v: v * alpha // 255)
    sh.paste((5, 7, 31, 255), (40 + off[0], 40 + off[1]), a)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    sh.alpha_composite(im, (40, 40))
    return sh


def main():
    # the picture is fitted to the full height and pushed right; the strip it leaves on the left is filled with a blurred
    # mirror of its own edge, which leaves a wide calm area for big headline letters
    art = Image.open(SRC).convert("RGB")
    s = H / art.height
    art = art.resize((round(art.width * s), H), Image.LANCZOS)
    shift = W - art.width
    placed = Image.new("RGB", (W, H)); placed.paste(art, (shift, 0))
    fill = Image.new("RGB", (W, H)); strip = art.crop((0, 0, shift + 200, H)).transpose(Image.FLIP_LEFT_RIGHT)
    fill.paste(strip, (0, 0)); fill = fill.filter(ImageFilter.GaussianBlur(45))
    mask = Image.new("L", (W, H), 0); dm = ImageDraw.Draw(mask)
    dm.rectangle([0, 0, shift, H], fill=255)
    for x in range(160):   # a wide soft seam
        dm.line([(shift + x, 0), (shift + x, H)], fill=int(255 * (1 - x / 160) ** 1.5))
    img = Image.composite(fill, placed, mask).convert("RGBA")

    # a soft dark wash behind the headline so it reads on any background
    wash = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(wash)
    for x in range(0, 1100, 6):
        a = int(120 * max(0, 1 - x / 1100) ** 1.4)
        d.rectangle([x, 0, x + 6, H], fill=(5, 7, 31, a))
    img.alpha_composite(wash)

    # "FLAWED AI": one style, one size: tall condensed white letters with a thick dark outline, on two lines
    font = ImageFont.truetype(r"C:\Windows\Fonts\impact.ttf", 10)
    for size in range(420, 80, -4):
        font = ImageFont.truetype(r"C:\Windows\Fonts\impact.ttf", size)
        if font.getbbox("FLAWED")[2] - font.getbbox("FLAWED")[0] <= 840:
            break
    fill, stroke = (255, 255, 255, 255), (5, 7, 31, 255)
    l1, l2 = layer_text("FLAWED", font, fill, stroke, 14), layer_text("AI", font, fill, stroke, 14)
    block = Image.new("RGBA", (max(l1.width, l2.width), l1.height + l2.height - 50), (0, 0, 0, 0))
    block.alpha_composite(l1, (0, 0)); block.alpha_composite(l2, (0, l1.height - 50))
    block = shadowed(block)
    img.alpha_composite(block, (0, max(0, (H - block.height) // 2 + 10)))

    os.makedirs(OUT, exist_ok=True)
    img = img.convert("RGB")
    img.save(os.path.join(OUT, "thumbnail_1920x1080.png"), optimize=True)
    small = img.resize((1280, 720), Image.LANCZOS)
    q = 93
    p = os.path.join(OUT, "thumbnail_1280x720.jpg")
    while True:   # YouTube's limit is 2 MB
        small.save(p, quality=q, optimize=True, progressive=True)
        if os.path.getsize(p) < 1_900_000 or q < 70:
            break
        q -= 5
    print("wrote", p, os.path.getsize(p) // 1024, "KB")


if __name__ == "__main__":
    main()
