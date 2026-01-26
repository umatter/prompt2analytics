# p2a Logo Assets

Production-ready logo files for prompt2analytics.

## Logo System Overview

| File | Usage | Format |
|------|-------|--------|
| `icon/p2a-icon.svg` | App icon, taskbar, dock | Square, scalable |
| `icon/p2a-icon-minimal.svg` | Favicon, very small contexts | Wave only, no text |
| `wordmark/p2a-wordmark.svg` | Headers, navigation, docs | Horizontal |
| `badge/p2a-badge.svg` | Splash, about, marketing | Pill shape |

## Directory Structure

```
p2a-assets/
├── icon/
│   ├── p2a-icon.svg           # Main icon (7J)
│   ├── p2a-icon-minimal.svg   # Wave only for small sizes
│   ├── p2a-icon.ico           # Windows icon (multi-size)
│   ├── p2a-icon-16.png
│   ├── p2a-icon-32.png
│   ├── p2a-icon-64.png
│   ├── p2a-icon-128.png
│   ├── p2a-icon-256.png
│   ├── p2a-icon-512.png
│   ├── p2a-icon-minimal-16.png
│   └── p2a-icon-minimal-32.png
├── wordmark/
│   └── p2a-wordmark.svg       # Horizontal logo (7K)
├── badge/
│   └── p2a-badge.svg          # Pill badge (7L)
├── logo.rs                    # Dioxus components
└── README.md
```

## Dioxus Integration

Copy `logo.rs` to your `src/components/` directory.

### Usage in RSX

```rust
use crate::components::logo::{P2aIcon, P2aIconMinimal, P2aWordmark, P2aBadge};

fn Header() -> Element {
    rsx! {
        nav {
            class: "header",
            P2aWordmark { width: 120.0 }
            // ... nav items
        }
    }
}

fn SplashScreen() -> Element {
    rsx! {
        div {
            class: "splash",
            P2aBadge { width: 200.0 }
            p { "Loading..." }
        }
    }
}
```

### Setting Window Icon (Dioxus Desktop)

```rust
use dioxus::desktop::{Config, WindowBuilder};

fn main() {
    // Load icon from embedded bytes
    let icon_bytes = include_bytes!("../assets/icon/p2a-icon-256.png");
    let icon = image::load_from_memory(icon_bytes)
        .expect("Failed to load icon")
        .to_rgba8();
    let (width, height) = icon.dimensions();
    let icon = tao::window::Icon::from_rgba(icon.into_raw(), width, height)
        .expect("Failed to create icon");

    let config = Config::new()
        .with_window(
            WindowBuilder::new()
                .with_title("prompt2analytics")
                .with_window_icon(Some(icon))
        );

    dioxus::LaunchBuilder::desktop()
        .with_cfg(config)
        .launch(App);
}
```

### Raw SVG Strings

For cases where you need the SVG as a string:

```rust
use crate::components::logo::raw;

// Write to file, set as HTML content, etc.
let svg_string = raw::ICON;
```

## Brand Colors

| Color | Hex | Usage |
|-------|-----|-------|
| Orange | `#FF6B35` | Gradient start (prompt/input) |
| Cyan | `#00B4D8` | Gradient end (analytics/output) |

## macOS .icns Generation

On macOS, create an iconset and convert:

```bash
mkdir p2a-icon.iconset
cp p2a-icon-16.png p2a-icon.iconset/icon_16x16.png
cp p2a-icon-32.png p2a-icon.iconset/icon_16x16@2x.png
cp p2a-icon-32.png p2a-icon.iconset/icon_32x32.png
cp p2a-icon-64.png p2a-icon.iconset/icon_32x32@2x.png
cp p2a-icon-128.png p2a-icon.iconset/icon_128x128.png
cp p2a-icon-256.png p2a-icon.iconset/icon_128x128@2x.png
cp p2a-icon-256.png p2a-icon.iconset/icon_256x256.png
cp p2a-icon-512.png p2a-icon.iconset/icon_256x256@2x.png
cp p2a-icon-512.png p2a-icon.iconset/icon_512x512.png
iconutil -c icns p2a-icon.iconset
```

## Regenerating PNGs

If you modify the SVGs, regenerate PNGs with:

```bash
pip install cairosvg pillow

python3 << 'EOF'
import cairosvg
from PIL import Image

sizes = [16, 32, 64, 128, 256, 512]
for size in sizes:
    cairosvg.svg2png(
        url='icon/p2a-icon.svg',
        write_to=f'icon/p2a-icon-{size}.png',
        output_width=size,
        output_height=size
    )
EOF
```
