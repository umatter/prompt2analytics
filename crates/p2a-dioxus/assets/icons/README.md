# App Icons

This directory contains the app icons for prompt2analytics.

The logo is drawn on a grid of square cells: p, 2 and a, with the 2 in orange
(#FF4F00). The icon is the 2 alone, black on an orange tile. Every SVG here is
drawn from the cell paths in `src/components/logo.rs`; the PNG and ICO files are
rendered from `favicon.svg` (`rsvg-convert -w N -h N favicon.svg`).

## Current Files

- `favicon.svg`, `icon.svg`, `p2a-icon.svg`, `p2a-icon-minimal.svg` - the orange tile
- `p2a-wordmark.svg`, `p2a-wordmark-dark.svg` - the p2a wordmark for light and dark backgrounds
- `p2a-icon-{16,32,64,128,256,512}.png`, `p2a-icon-minimal-{16,32}.png` - the tile as PNG
- `32x32.png`, `128x128.png`, `256x256.png`, `512x512.png` - the tile at bundle sizes
- `p2a-icon.ico` - Windows icon (16, 32, 64, 256)

The README header lockup (mark plus "Prompt 2 Analytics" in Archivo, outlined) is in `docs/assets/`.

## Optional Files (for better platform support)

For proper cross-platform bundling, you can optionally add:

| File | Size | Platform |
|------|------|----------|
| `32x32.png` | 32x32 px | All |
| `128x128.png` | 128x128 px | All |
| `256x256.png` | 256x256 px | All |
| `512x512.png` | 512x512 px | macOS, Linux |
| `icon.ico` | Multi-size | Windows |
| `icon.icns` | Multi-size | macOS |

## Generating Icons

### From a source image (e.g., 1024x1024 PNG)

```bash
# Install ImageMagick
sudo apt install imagemagick  # Linux
brew install imagemagick      # macOS

# Generate PNG sizes
convert source.png -resize 32x32 32x32.png
convert source.png -resize 128x128 128x128.png
convert source.png -resize 256x256 256x256.png
convert source.png -resize 512x512 512x512.png

# Generate Windows ICO (multi-size)
convert source.png -define icon:auto-resize=256,128,64,48,32,16 icon.ico

# Generate macOS ICNS
# Option 1: Use iconutil (macOS only)
mkdir icon.iconset
cp 512x512.png icon.iconset/icon_512x512.png
cp 256x256.png icon.iconset/icon_256x256.png
cp 128x128.png icon.iconset/icon_128x128.png
cp 32x32.png icon.iconset/icon_32x32.png
iconutil -c icns icon.iconset -o icon.icns

# Option 2: Use png2icns (Linux)
sudo apt install icnsutils
png2icns icon.icns 512x512.png 256x256.png 128x128.png 32x32.png
```

## After Adding Icons

Uncomment the `icon` line in `Dioxus.toml`:

```toml
[bundle]
icon = ["assets/icons/32x32.png", "assets/icons/128x128.png", "assets/icons/256x256.png", "assets/icons/icon.icns", "assets/icons/icon.ico"]
```
