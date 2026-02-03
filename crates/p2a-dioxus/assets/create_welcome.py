#!/usr/bin/env python3
"""
Create static welcome images with multilingual greetings.
Generates variants for light and dark themes with transparent backgrounds.
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
WIDTH = 360
HEIGHT = 180

# Theme colors
THEMES = {
    "light": {
        "text": (40, 40, 40),        # Dark text for light background
        "text_light": (120, 120, 120),
        "dots": (180, 180, 180),
    },
    "dark": {
        "text": (240, 240, 240),     # Light text for dark background
        "text_light": (160, 160, 160),
        "dots": (100, 100, 100),
    },
}

def load_fonts():
    """Try to load appropriate fonts."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]

    chinese_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]

    font_large = None
    font_medium = None

    for path in font_paths:
        try:
            font_large = ImageFont.truetype(path, 48)
            font_medium = ImageFont.truetype(path.replace("-Bold", "").replace("-B", "-R"), 16)
            break
        except:
            continue

    if font_large is None:
        font_large = ImageFont.load_default()
        font_medium = font_large

    # Load CJK font
    font_cjk = None
    for path in chinese_paths:
        try:
            font_cjk = ImageFont.truetype(path, 16)
            break
        except:
            continue

    if font_cjk is None:
        font_cjk = font_medium

    return font_large, font_medium, font_cjk

def create_welcome_image(theme_name, colors):
    """Create the welcome image for a specific theme."""
    # RGBA for transparency
    img = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_large, font_medium, font_cjk = load_fonts()

    text_color = colors["text"]
    text_light = colors["text_light"]
    dot_color = colors["dots"]

    # Main greeting: "Grüezi" - largest, centered
    main_text = "Grüezi"
    bbox = draw.textbbox((0, 0), main_text, font=font_large)
    text_width = bbox[2] - bbox[0]
    x = (WIDTH - text_width) // 2
    y = 35
    draw.text((x, y), main_text, font=font_large, fill=text_color)

    # Add subtle decorative dots below main text
    dot_y = 95
    for i in range(5):
        dot_x = WIDTH // 2 - 32 + i * 16
        draw.ellipse([dot_x, dot_y, dot_x + 4, dot_y + 4], fill=dot_color)

    # Row 1: Welcome, Bienvenue, 欢迎
    row1 = [
        ("Welcome", font_medium, text_color),
        ("Bienvenue", font_medium, text_light),
        ("欢迎", font_cjk, text_light),
    ]

    spacing = 18
    total_width = sum(draw.textbbox((0, 0), t, font=f)[2] for t, f, _ in row1) + spacing * (len(row1) - 1)
    x = (WIDTH - total_width) // 2
    y = 115

    for text, font, color in row1:
        draw.text((x, y), text, font=font, fill=color)
        bbox = draw.textbbox((0, 0), text, font=font)
        x += bbox[2] - bbox[0] + spacing

    # Row 2: Willkommen, Benvenuto
    row2 = [
        ("Willkommen", font_medium, text_light),
        ("Benvenuto", font_medium, text_light),
    ]

    total_width = sum(draw.textbbox((0, 0), t, font=f)[2] for t, f, _ in row2) + spacing * (len(row2) - 1)
    x = (WIDTH - total_width) // 2
    y = 145

    for text, font, color in row2:
        draw.text((x, y), text, font=font, fill=color)
        bbox = draw.textbbox((0, 0), text, font=font)
        x += bbox[2] - bbox[0] + spacing

    return img

def main():
    base_path = os.path.dirname(__file__)

    for theme_name, colors in THEMES.items():
        print(f"Creating {theme_name} theme...")
        img = create_welcome_image(theme_name, colors)

        filename = f"welcome-{theme_name}.png"
        output_path = os.path.join(base_path, filename)
        img.save(output_path, "PNG", optimize=True)
        print(f"  Saved: {filename}")

    print(f"\nDone! Size: {WIDTH}x{HEIGHT}")

if __name__ == "__main__":
    main()
