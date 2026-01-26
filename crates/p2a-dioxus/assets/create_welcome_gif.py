#!/usr/bin/env python3
"""
Create a pixel art welcome GIF with multilingual greetings.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import math

# Configuration
WIDTH = 320
HEIGHT = 160
FRAMES = 30
DURATION = 100  # ms per frame

# Colors (matching p2a theme)
BG_COLOR = (250, 250, 250)  # Light gray background
PINK = (219, 39, 119)       # Magenta/pink for MCP
TEAL = (20, 184, 166)       # Teal for LLM
ORANGE = (249, 115, 22)     # Orange for Rust

def create_frame(frame_num, total_frames, fonts):
    """Create a single frame of the animation."""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font_large, font_medium, font_chinese = fonts

    # Animation progress (0 to 1, looping)
    progress = frame_num / total_frames

    # Subtle floating animation for main text
    float_offset = math.sin(progress * 2 * math.pi) * 3

    # Draw decorative pixel corners (animated)
    pixel_size = 4
    for i in range(6):
        phase = (progress * 2 + i * 0.15) % 1
        alpha_factor = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(phase * 2 * math.pi))

        # Top-left teal pixels
        x = i * pixel_size + int(math.sin(progress * 2 * math.pi + i) * 2)
        y = i * pixel_size
        color = tuple(int(c * alpha_factor + BG_COLOR[j] * (1 - alpha_factor)) for j, c in enumerate(TEAL))
        draw.rectangle([x, y, x + pixel_size - 1, y + pixel_size - 1], fill=color)

        # Top-right pink pixels
        x = WIDTH - (i + 1) * pixel_size - int(math.sin(progress * 2 * math.pi + i) * 2)
        color = tuple(int(c * alpha_factor + BG_COLOR[j] * (1 - alpha_factor)) for j, c in enumerate(PINK))
        draw.rectangle([x, y, x + pixel_size - 1, y + pixel_size - 1], fill=color)

        # Bottom-left orange pixels
        y = HEIGHT - (i + 1) * pixel_size
        x = i * pixel_size
        color = tuple(int(c * alpha_factor + BG_COLOR[j] * (1 - alpha_factor)) for j, c in enumerate(ORANGE))
        draw.rectangle([x, y, x + pixel_size - 1, y + pixel_size - 1], fill=color)

        # Bottom-right mixed pixels
        x = WIDTH - (i + 1) * pixel_size
        colors = [PINK, TEAL, ORANGE]
        color = tuple(int(c * alpha_factor + BG_COLOR[j] * (1 - alpha_factor)) for j, c in enumerate(colors[i % 3]))
        draw.rectangle([x, y, x + pixel_size - 1, y + pixel_size - 1], fill=color)

    # Main text: "Grüezi" - largest, center, pink
    gruezi_text = "Grüezi"
    bbox = draw.textbbox((0, 0), gruezi_text, font=font_large)
    text_width = bbox[2] - bbox[0]
    x = (WIDTH - text_width) // 2
    y = 35 + int(float_offset)

    # Shadow
    draw.text((x + 2, y + 2), gruezi_text, font=font_large, fill=(200, 200, 200))
    # Main text
    draw.text((x, y), gruezi_text, font=font_large, fill=PINK)

    # Chinese: "欢迎" - left side, teal
    chinese_text = "欢迎"
    bbox = draw.textbbox((0, 0), chinese_text, font=font_chinese)
    text_width = bbox[2] - bbox[0]
    x = WIDTH // 4 - text_width // 2 + 10
    y = 100 + int(float_offset * 0.5)

    draw.text((x + 1, y + 1), chinese_text, font=font_chinese, fill=(200, 200, 200))
    draw.text((x, y), chinese_text, font=font_chinese, fill=TEAL)

    # "Welcome" - right side, orange
    welcome_text = "Welcome"
    bbox = draw.textbbox((0, 0), welcome_text, font=font_medium)
    text_width = bbox[2] - bbox[0]
    x = 3 * WIDTH // 4 - text_width // 2 - 10
    y = 100 + int(float_offset * 0.5)

    draw.text((x + 1, y + 1), welcome_text, font=font_medium, fill=(200, 200, 200))
    draw.text((x, y), welcome_text, font=font_medium, fill=ORANGE)

    # Add sparkle pixels
    sparkle_positions = [
        (45, 25), (275, 30), (160, 130), (25, 100), (295, 95),
        (75, 70), (245, 65), (130, 20), (190, 135), (50, 135), (270, 130)
    ]
    for i, (sx, sy) in enumerate(sparkle_positions):
        phase = (progress + i * 0.1) % 1
        if phase < 0.3:
            brightness = phase / 0.3
        elif phase < 0.5:
            brightness = 1.0
        else:
            brightness = 0

        if brightness > 0:
            size = 2 + int(brightness * 2)
            colors = [PINK, TEAL, ORANGE]
            base_color = colors[i % 3]
            color = tuple(int(c * brightness + BG_COLOR[j] * (1 - brightness)) for j, c in enumerate(base_color))
            draw.rectangle([sx, sy, sx + size, sy + size], fill=color)

    return img

def load_fonts():
    """Try to load appropriate fonts."""
    font_paths = [
        # Common Linux paths
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
         "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
         "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]

    for large_path, medium_path, chinese_path in font_paths:
        try:
            font_large = ImageFont.truetype(large_path, 48)
            font_medium = ImageFont.truetype(medium_path, 22)
            try:
                font_chinese = ImageFont.truetype(chinese_path, 26)
            except:
                font_chinese = font_medium
            return font_large, font_medium, font_chinese
        except:
            continue

    # Fallback to default
    print("Warning: Using default font, text may not look ideal")
    default = ImageFont.load_default()
    return default, default, default

def pixelate(img, pixel_size=2):
    """Apply subtle pixelation effect."""
    small = img.resize(
        (img.width // pixel_size, img.height // pixel_size),
        Image.Resampling.NEAREST
    )
    return small.resize(img.size, Image.Resampling.NEAREST)

def main():
    fonts = load_fonts()
    frames = []

    print("Generating frames...")
    for i in range(FRAMES):
        frame = create_frame(i, FRAMES, fonts)
        # Subtle pixelation for retro feel
        frame = pixelate(frame, 2)
        frames.append(frame)
        print(f"  Frame {i+1}/{FRAMES}")

    # Save as GIF
    output_path = os.path.join(os.path.dirname(__file__), "welcome.gif")

    print(f"Saving to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0,
        optimize=True
    )

    print(f"Done! Created {output_path}")
    print(f"  Size: {WIDTH}x{HEIGHT}")
    print(f"  Frames: {FRAMES}")
    print(f"  Duration: {DURATION}ms per frame ({1000/DURATION:.1f} fps)")

if __name__ == "__main__":
    main()
