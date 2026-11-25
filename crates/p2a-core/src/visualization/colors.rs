//! Colors of the p2a charts.
//!
//! The first is the app's orange. The order is fixed: neighbouring series colors stay apart for
//! readers with protanopia, deuteranopia or tritanopia, and every color has a contrast of at
//! least 3:1 on white.

use plotters::style::RGBColor;

/// Orange, the app's orange (#F74C00)
pub const BRAND_ORANGE: RGBColor = RGBColor(247, 76, 0);

/// Blue (#1769AA)
pub const BRAND_BLUE: RGBColor = RGBColor(23, 105, 170);

/// Teal (#14A38B)
pub const BRAND_TEAL: RGBColor = RGBColor(20, 163, 139);

/// Purple (#7A3FB0)
pub const BRAND_PURPLE: RGBColor = RGBColor(122, 63, 176);

/// Ochre (#B58900)
pub const BRAND_OCHRE: RGBColor = RGBColor(181, 137, 0);

/// Magenta (#B0286B)
pub const BRAND_MAGENTA: RGBColor = RGBColor(176, 40, 107);

/// Slate gray for neutral elements (#64748b)
pub const BRAND_SLATE: RGBColor = RGBColor(100, 116, 139);

/// Default color palette for multi-series charts, in order of use.
pub const CHART_PALETTE: [RGBColor; 6] = [
    BRAND_ORANGE,
    BRAND_BLUE,
    BRAND_TEAL,
    BRAND_PURPLE,
    BRAND_OCHRE,
    BRAND_MAGENTA,
];

/// [`CHART_PALETTE`] as (u8, u8, u8) tuples, for the interactive (Plotly) charts.
pub const PLOTLY_PALETTE: [(u8, u8, u8); 6] = [
    (247, 76, 0),   // Orange
    (23, 105, 170), // Blue
    (20, 163, 139), // Teal
    (122, 63, 176), // Purple
    (181, 137, 0),  // Ochre
    (176, 40, 107), // Magenta
];

/// Default color for single-series charts.
pub const DEFAULT_SERIES_COLOR: RGBColor = BRAND_ORANGE;

/// Secondary color for emphasis or contrast elements.
pub const SECONDARY_COLOR: RGBColor = BRAND_BLUE;

/// Color for outliers and anomalies.
pub const OUTLIER_COLOR: RGBColor = BRAND_MAGENTA;

/// Color for trend lines and fitted curves.
pub const TREND_LINE_COLOR: RGBColor = BRAND_BLUE;

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
