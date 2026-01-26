//! p2a Logo Components for Dioxus
//!
//! This module provides the p2a logo system as reusable Dioxus components.
//!
//! # Usage
//! ```rust
//! use crate::components::logo::{P2aIcon, P2aIconMinimal, P2aWordmark, P2aBadge};
//!
//! fn MyComponent() -> Element {
//!     rsx! {
//!         // In your app header
//!         P2aWordmark { width: 120.0 }
//!
//!         // As an icon
//!         P2aIcon { size: 48.0 }
//!
//!         // For favicon/small contexts
//!         P2aIconMinimal { size: 32.0 }
//!
//!         // For splash screens
//!         P2aBadge { width: 200.0 }
//!     }
//! }
//! ```

use dioxus::prelude::*;

/// Brand colors
pub mod colors {
    pub const ORANGE: &str = "#FF6B35";
    pub const CYAN: &str = "#00B4D8";
}

/// Main square icon with "p2a" text (7J)
/// Best for: app icons, taskbar, dock, avatars
#[component]
pub fn P2aIcon(
    /// Size in pixels (width = height)
    #[props(default = 120.0)]
    size: f64,
    /// Optional CSS class
    #[props(default = "")]
    class: &'static str,
) -> Element {
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            view_box: "0 0 120 120",
            width: "{size}",
            height: "{size}",
            class: "{class}",
            defs {
                linearGradient {
                    id: "p2a-icon-grad",
                    x1: "0%",
                    y1: "0%",
                    x2: "100%",
                    y2: "100%",
                    stop { offset: "0%", stop_color: "{colors::ORANGE}" }
                    stop { offset: "100%", stop_color: "{colors::CYAN}" }
                }
            }
            rect {
                x: "10",
                y: "10",
                width: "100",
                height: "100",
                rx: "22",
                fill: "url(#p2a-icon-grad)"
            }
            path {
                d: "M25 45 C38 28 52 28 60 42 C68 56 82 56 95 40",
                stroke: "#fff",
                stroke_width: "4",
                fill: "none",
                stroke_linecap: "round"
            }
            text {
                x: "60",
                y: "85",
                text_anchor: "middle",
                font_family: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
                font_size: "28",
                font_weight: "700",
                fill: "#fff",
                "p2a"
            }
        }
    }
}

/// Minimal icon with wave only, no text (for small sizes)
/// Best for: favicon, 16-24px contexts
#[component]
pub fn P2aIconMinimal(
    /// Size in pixels (width = height)
    #[props(default = 32.0)]
    size: f64,
    /// Optional CSS class
    #[props(default = "")]
    class: &'static str,
) -> Element {
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            view_box: "0 0 120 120",
            width: "{size}",
            height: "{size}",
            class: "{class}",
            defs {
                linearGradient {
                    id: "p2a-minimal-grad",
                    x1: "0%",
                    y1: "0%",
                    x2: "100%",
                    y2: "100%",
                    stop { offset: "0%", stop_color: "{colors::ORANGE}" }
                    stop { offset: "100%", stop_color: "{colors::CYAN}" }
                }
            }
            rect {
                x: "10",
                y: "10",
                width: "100",
                height: "100",
                rx: "22",
                fill: "url(#p2a-minimal-grad)"
            }
            path {
                d: "M25 60 C40 38 55 38 60 55 C65 72 80 72 95 50",
                stroke: "#fff",
                stroke_width: "6",
                fill: "none",
                stroke_linecap: "round"
            }
        }
    }
}

/// Horizontal wordmark with wave accent (7K)
/// Best for: headers, navigation bars, documentation
#[component]
pub fn P2aWordmark(
    /// Width in pixels (height auto-calculated)
    #[props(default = 180.0)]
    width: f64,
    /// Optional CSS class
    #[props(default = "")]
    class: &'static str,
) -> Element {
    let height = width * (63.0 / 180.0);
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            view_box: "0 0 180 63",
            width: "{width}",
            height: "{height}",
            class: "{class}",
            defs {
                linearGradient {
                    id: "p2a-wordmark-grad",
                    x1: "0%",
                    y1: "0%",
                    x2: "100%",
                    y2: "0%",
                    stop { offset: "0%", stop_color: "{colors::ORANGE}" }
                    stop { offset: "100%", stop_color: "{colors::CYAN}" }
                }
            }
            text {
                x: "90",
                y: "48",
                text_anchor: "middle",
                font_family: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
                font_size: "52",
                font_weight: "800",
                fill: "url(#p2a-wordmark-grad)",
                "p2a"
            }
            path {
                d: "M15 32 C40 15 70 15 90 32 C110 49 140 49 165 32",
                stroke: "url(#p2a-wordmark-grad)",
                stroke_width: "4",
                fill: "none",
                stroke_linecap: "round",
                opacity: "0.3"
            }
        }
    }
}

/// Pill-shaped badge with wave and text (7L)
/// Best for: splash screens, about dialogs, marketing
#[component]
pub fn P2aBadge(
    /// Width in pixels (height auto-calculated)
    #[props(default = 160.0)]
    width: f64,
    /// Optional CSS class
    #[props(default = "")]
    class: &'static str,
) -> Element {
    let height = width * 0.5;
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            view_box: "0 0 160 80",
            width: "{width}",
            height: "{height}",
            class: "{class}",
            defs {
                linearGradient {
                    id: "p2a-badge-grad",
                    x1: "0%",
                    y1: "0%",
                    x2: "100%",
                    y2: "100%",
                    stop { offset: "0%", stop_color: "{colors::ORANGE}" }
                    stop { offset: "100%", stop_color: "{colors::CYAN}" }
                }
            }
            rect {
                x: "5",
                y: "5",
                width: "150",
                height: "70",
                rx: "35",
                fill: "url(#p2a-badge-grad)"
            }
            path {
                d: "M25 40 C35 28 45 28 55 40 C65 52 75 52 85 40",
                stroke: "#fff",
                stroke_width: "3.5",
                fill: "none",
                stroke_linecap: "round"
            }
            text {
                x: "118",
                y: "50",
                text_anchor: "middle",
                font_family: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
                font_size: "28",
                font_weight: "700",
                fill: "#fff",
                "p2a"
            }
        }
    }
}

/// Raw SVG strings for use cases where you need the SVG as a string
/// (e.g., writing to files, setting window icons)
///
/// Note: Adjust paths based on your project structure.
/// These assume: src/components/logo.rs and assets/ at project root.
pub mod raw {
    /// Main icon SVG (7J) - square with wave and "p2a" text
    pub const ICON: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><defs><linearGradient id="p2a-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#FF6B35"/><stop offset="100%" stop-color="#00B4D8"/></linearGradient></defs><rect x="10" y="10" width="100" height="100" rx="22" fill="url(#p2a-grad)"/><path d="M25 45 C38 28 52 28 60 42 C68 56 82 56 95 40" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round"/><text x="60" y="85" text-anchor="middle" font-family="system-ui,-apple-system,sans-serif" font-size="28" font-weight="700" fill="#fff">p2a</text></svg>"##;

    /// Minimal icon SVG - wave only, for small sizes (favicon)
    pub const ICON_MINIMAL: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><defs><linearGradient id="p2a-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#FF6B35"/><stop offset="100%" stop-color="#00B4D8"/></linearGradient></defs><rect x="10" y="10" width="100" height="100" rx="22" fill="url(#p2a-grad)"/><path d="M25 60 C40 38 55 38 60 55 C65 72 80 72 95 50" stroke="#fff" stroke-width="6" fill="none" stroke-linecap="round"/></svg>"##;

    /// Wordmark SVG (7K) - horizontal "p2a" with wave accent
    pub const WORDMARK: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 63"><defs><linearGradient id="p2a-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#FF6B35"/><stop offset="100%" stop-color="#00B4D8"/></linearGradient></defs><text x="90" y="48" text-anchor="middle" font-family="system-ui,-apple-system,sans-serif" font-size="52" font-weight="800" fill="url(#p2a-grad)">p2a</text><path d="M15 32 C40 15 70 15 90 32 C110 49 140 49 165 32" stroke="url(#p2a-grad)" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.3"/></svg>"##;

    /// Badge SVG (7L) - pill shape with wave and "p2a"
    pub const BADGE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 80"><defs><linearGradient id="p2a-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#FF6B35"/><stop offset="100%" stop-color="#00B4D8"/></linearGradient></defs><rect x="5" y="5" width="150" height="70" rx="35" fill="url(#p2a-grad)"/><path d="M25 40 C35 28 45 28 55 40 C65 52 75 52 85 40" stroke="#fff" stroke-width="3.5" fill="none" stroke-linecap="round"/><text x="118" y="50" text-anchor="middle" font-family="system-ui,-apple-system,sans-serif" font-size="28" font-weight="700" fill="#fff">p2a</text></svg>"##;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colors() {
        assert_eq!(colors::ORANGE, "#FF6B35");
        assert_eq!(colors::CYAN, "#00B4D8");
    }
}
