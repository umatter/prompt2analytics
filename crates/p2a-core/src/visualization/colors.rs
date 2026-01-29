//! Brand color palette for p2a visualizations.
//!
//! This module defines a consistent color scheme based on the project's brand identity:
//! - **Primary gradient**: Orange (#FF6B35) to Cyan (#00B4D8)
//! - **Accent colors**: Teal and orange variants from the UI palette

use plotters::style::RGBColor;

/// Brand orange - primary color from logo (#FF6B35)
pub const BRAND_ORANGE: RGBColor = RGBColor(255, 107, 53);

/// Brand cyan - secondary color from logo (#00B4D8)
pub const BRAND_CYAN: RGBColor = RGBColor(0, 180, 216);

/// Teal accent from UI palette (#14b8a6)
pub const BRAND_TEAL: RGBColor = RGBColor(20, 184, 166);

/// Orange accent - lighter variant (#f97316)
pub const BRAND_ORANGE_LIGHT: RGBColor = RGBColor(249, 115, 22);

/// Darker teal for contrast (#0d9488)
pub const BRAND_TEAL_DARK: RGBColor = RGBColor(13, 148, 136);

/// Darker orange for contrast (#ea580c)
pub const BRAND_ORANGE_DARK: RGBColor = RGBColor(234, 88, 12);

/// Slate gray for neutral elements (#64748b)
pub const BRAND_SLATE: RGBColor = RGBColor(100, 116, 139);

/// Default color palette for multi-series charts.
///
/// Order optimized for visual distinction:
/// 1. Brand orange (primary)
/// 2. Brand cyan (secondary, high contrast with orange)
/// 3. Teal (between orange and cyan)
/// 4. Light orange (variant of primary)
/// 5. Dark teal (variant of teal)
/// 6. Dark orange (variant of primary)
pub const CHART_PALETTE: [RGBColor; 6] = [
    BRAND_ORANGE,
    BRAND_CYAN,
    BRAND_TEAL,
    BRAND_ORANGE_LIGHT,
    BRAND_TEAL_DARK,
    BRAND_ORANGE_DARK,
];

/// RGB tuple versions for plotlars/Plotly integration.
///
/// Same colors as `CHART_PALETTE` but as (u8, u8, u8) tuples
/// compatible with the plotlars `Rgb` type.
pub const PLOTLY_PALETTE: [(u8, u8, u8); 6] = [
    (255, 107, 53), // Brand orange
    (0, 180, 216),  // Brand cyan
    (20, 184, 166), // Teal
    (249, 115, 22), // Orange light
    (13, 148, 136), // Teal dark
    (234, 88, 12),  // Orange dark
];

/// Default color for single-series charts (brand orange).
pub const DEFAULT_SERIES_COLOR: RGBColor = BRAND_ORANGE;

/// Secondary color for emphasis or contrast elements (brand cyan).
pub const SECONDARY_COLOR: RGBColor = BRAND_CYAN;

/// Color for outliers and anomalies (dark orange for visibility).
pub const OUTLIER_COLOR: RGBColor = BRAND_ORANGE_DARK;

/// Color for trend lines and fitted curves (dark teal for contrast).
pub const TREND_LINE_COLOR: RGBColor = BRAND_TEAL_DARK;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_has_six_colors() {
        assert_eq!(CHART_PALETTE.len(), 6);
        assert_eq!(PLOTLY_PALETTE.len(), 6);
    }

    #[test]
    fn test_plotly_palette_matches_chart_palette() {
        for (i, rgb) in CHART_PALETTE.iter().enumerate() {
            let (r, g, b) = PLOTLY_PALETTE[i];
            assert_eq!(rgb.0, r);
            assert_eq!(rgb.1, g);
            assert_eq!(rgb.2, b);
        }
    }
}
