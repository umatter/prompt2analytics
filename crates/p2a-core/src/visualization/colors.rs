//! Colors of the p2a charts.
//!
//! The first is the app's orange. The order is fixed: neighbouring series colors stay apart for
//! readers with protanopia, deuteranopia or tritanopia, and every color has a contrast of at
//! least 3:1 on white. The order (orange, purple, teal, magenta, blue, ochre) maximises the
//! smallest CIELAB difference among the first three, then four, colors over normal and
//! simulated (Machado 2009) protan, deutan and tritan vision, on the light page and through
//! the dark theme's `invert(.9) hue-rotate(180deg)` filter: 38.6 among three, where orange,
//! blue, teal (blue and teal close in the dark theme) gave 12.3.

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
    BRAND_PURPLE,
    BRAND_TEAL,
    BRAND_MAGENTA,
    BRAND_BLUE,
    BRAND_OCHRE,
];

/// [`CHART_PALETTE`] as (u8, u8, u8) tuples, for the interactive (Plotly) charts.
pub const PLOTLY_PALETTE: [(u8, u8, u8); 6] = [
    (247, 76, 0),   // Orange
    (122, 63, 176), // Purple
    (20, 163, 139), // Teal
    (176, 40, 107), // Magenta
    (23, 105, 170), // Blue
    (181, 137, 0),  // Ochre
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

    /// CIELAB (D65) of an sRGB color.
    fn lab(c: (f64, f64, f64)) -> [f64; 3] {
        let lin = |v: f64| {
            let v = v / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        let (r, g, b) = (lin(c.0), lin(c.1), lin(c.2));
        let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089;
        let f = |t: f64| {
            if t > 0.008856 {
                t.cbrt()
            } else {
                7.787 * t + 16.0 / 116.0
            }
        };
        [
            116.0 * f(y) - 16.0,
            500.0 * (f(x) - f(y)),
            200.0 * (f(y) - f(z)),
        ]
    }

    /// The dark theme's CSS filter on a chart image: invert(.9), then hue-rotate(180deg).
    fn dark(c: (u8, u8, u8)) -> (f64, f64, f64) {
        let v = [c.0, c.1, c.2].map(|x| 0.9 - 0.8 * f64::from(x) / 255.0);
        let m = [
            [-0.574, 1.430, 0.144],
            [0.426, 0.430, 0.144],
            [0.426, 1.430, -0.856],
        ];
        let px = |i: usize| {
            (255.0 * (m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])).clamp(0.0, 255.0)
        };
        (px(0), px(1), px(2))
    }

    /// The first three series colors stay apart on the light page and in the dark theme
    /// (blue and teal, the old second and third, were 46 apart after the dark filter).
    #[test]
    fn the_first_three_colors_stay_apart_in_both_themes() {
        let light = |c: (u8, u8, u8)| (f64::from(c.0), f64::from(c.1), f64::from(c.2));
        for (i, a) in PLOTLY_PALETTE[..3].iter().enumerate() {
            for b in &PLOTLY_PALETTE[i + 1..3] {
                for view in [light as fn((u8, u8, u8)) -> (f64, f64, f64), dark] {
                    let (p, q) = (lab(view(*a)), lab(view(*b)));
                    let de =
                        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                            .sqrt();
                    assert!(de > 60.0, "{a:?} vs {b:?}: {de}");
                }
            }
        }
    }
}
