//! Heatmap visualization for correlation matrices and other 2D data.

use super::charts::{encode_rgb_to_png_base64, rgb_buffer};
use super::vector::{FigureBackend, SvgRecorder};
use super::{DEFAULT_HEIGHT, DEFAULT_WIDTH, VisualizationError};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

/// Result of heatmap generation.
pub struct HeatmapResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Vector version of the image (SVG); None when it would be larger than
    /// [`MAX_SVG_BYTES`](super::MAX_SVG_BYTES)
    pub svg: Option<String>,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row labels
    pub row_labels: Vec<String>,
    /// Column labels
    pub col_labels: Vec<String>,
    /// Notes on the drawing ("labels hidden: cells are below 10 px").
    pub warnings: Vec<String>,
}

impl std::fmt::Display for HeatmapResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Heatmap")?;
        writeln!(f, "=======")?;
        writeln!(f, "Dimensions: {} x {}", self.n_rows, self.n_cols)?;
        for w in &self.warnings {
            writeln!(f, "Warning: {w}")?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Top band for the title.
const TITLE_BAND: u32 = 60;
/// Bottom margin.
const BOTTOM_MARGIN: u32 = 20;
/// Left margin before the row labels.
const LEFT_MARGIN: u32 = 20;
/// Right band for the colour bar: a 20 px gap, the 20 px bar and its labels.
const COLORBAR_BAND: u32 = 100;
/// The label band when labels are hidden.
const HIDDEN_LABEL_MARGIN: u32 = 10;
/// Labels are drawn only on cells of at least this size.
const MIN_LABELLED_CELL: u32 = 10;
/// The smallest cell drawn; a matrix that needs smaller cells is refused.
const MIN_CELL: u32 = 4;

/// Where the matrix goes on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// Side of a (square) cell in pixels.
    cell: u32,
    /// Width of the row-label band left of the matrix and height of the (rotated)
    /// column-label band above it.
    label_margin: u32,
    /// Top-left corner of the matrix.
    x_offset: u32,
    y_offset: u32,
    /// Labels are drawn (cells of at least 10 px).
    labels: bool,
}

/// The cell size that fits `n_rows` x `n_cols` cells with label bands of `label_margin`.
fn cell_size(n_rows: usize, n_cols: usize, label_margin: u32, width: u32, height: u32) -> u32 {
    let avail_w = width.saturating_sub(LEFT_MARGIN + label_margin + COLORBAR_BAND);
    let avail_h = height.saturating_sub(TITLE_BAND + label_margin + BOTTOM_MARGIN);
    let per = |avail: u32, n: usize| u32::try_from(avail as usize / n.max(1)).unwrap_or(u32::MAX);
    per(avail_w, n_cols).min(per(avail_h, n_rows))
}

/// The layout of an `n_rows` x `n_cols` matrix whose longest label has `max_label_len`
/// characters: square cells as large as fit; labels drawn when the cells keep at least 10 px
/// with the label bands, otherwise hidden and the bands shrunk to 10 px; refused when the cells
/// would fall below 4 px.
fn layout(
    n_rows: usize,
    n_cols: usize,
    max_label_len: usize,
    width: u32,
    height: u32,
) -> Result<Layout, VisualizationError> {
    let label_margin = u32::try_from(max_label_len.saturating_mul(8).clamp(8, 150)).unwrap_or(150);
    let with_labels = cell_size(n_rows, n_cols, label_margin, width, height);
    let (cell, label_margin, labels) = if with_labels >= MIN_LABELLED_CELL {
        (with_labels, label_margin, true)
    } else {
        let cell = cell_size(n_rows, n_cols, HIDDEN_LABEL_MARGIN, width, height);
        (cell, HIDDEN_LABEL_MARGIN, false)
    };
    if cell < MIN_CELL {
        let capacity = heatmap_capacity(Some(width), Some(height));
        return Err(VisualizationError::InvalidData(format!(
            "a {n_rows} x {n_cols} heatmap does not fit a {width} x {height} image: at most \
             {capacity} variables fit (cells of at least {MIN_CELL} px); use a larger image or \
             fewer variables"
        )));
    }
    Ok(Layout {
        cell,
        label_margin,
        x_offset: LEFT_MARGIN + label_margin,
        y_offset: TITLE_BAND + label_margin,
        labels,
    })
}

/// The largest k for which a k x k heatmap fits a `width` x `height` image (default 800 x
/// 600), with 4 px cells and hidden labels: 127 at the default size.
pub fn heatmap_capacity(width: Option<u32>, height: Option<u32>) -> usize {
    let width = width.unwrap_or(DEFAULT_WIDTH);
    let height = height.unwrap_or(DEFAULT_HEIGHT);
    let avail_w = width.saturating_sub(LEFT_MARGIN + HIDDEN_LABEL_MARGIN + COLORBAR_BAND);
    let avail_h = height.saturating_sub(TITLE_BAND + HIDDEN_LABEL_MARGIN + BOTTOM_MARGIN);
    (avail_w.min(avail_h) / MIN_CELL) as usize
}

/// Generate a heatmap from a correlation matrix or other 2D data.
///
/// # Arguments
/// * `matrix` - 2D data as row-major Vec<Vec<f64>>
/// * `row_labels` - Labels for rows
/// * `col_labels` - Labels for columns
/// * `title` - Chart title
/// * `width` - Image width (default: 800)
/// * `height` - Image height (default: 600)
///
/// Square cells shrink to fit the image; row and (rotated) column labels are drawn while the
/// cells keep at least 10 px and hidden below that, with a warning in the result. A matrix that
/// would need cells below 4 px is refused naming [`heatmap_capacity`].
pub fn correlation_heatmap(
    matrix: &[Vec<f64>],
    row_labels: &[String],
    col_labels: &[String],
    title: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<HeatmapResult, VisualizationError> {
    if matrix.is_empty() {
        return Err(VisualizationError::InvalidData("Empty matrix".to_string()));
    }

    let n_rows = matrix.len();
    let n_cols = matrix[0].len();

    if row_labels.len() != n_rows || col_labels.len() != n_cols {
        return Err(VisualizationError::InvalidData(format!(
            "Label count mismatch: {}x{} matrix but {} row labels and {} col labels",
            n_rows,
            n_cols,
            row_labels.len(),
            col_labels.len()
        )));
    }

    let width = width.unwrap_or(DEFAULT_WIDTH);
    let height = height.unwrap_or(DEFAULT_HEIGHT);
    let title = title.unwrap_or("Correlation Matrix");

    // Find min/max for color scaling
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for row in matrix {
        for &val in row {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    // For correlation matrices, typically -1 to 1
    let is_correlation = min_val >= -1.0 - 1e-6 && max_val <= 1.0 + 1e-6;
    let (color_min, color_max) = if is_correlation {
        (-1.0, 1.0)
    } else {
        (min_val, max_val)
    };

    let max_label_len = row_labels
        .iter()
        .chain(col_labels.iter())
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(5);
    let lay = layout(n_rows, n_cols, max_label_len, width, height)?;
    let mut warnings = Vec::new();
    if !lay.labels {
        warnings.push(format!(
            "labels hidden: cells are below {MIN_LABELLED_CELL} px ({} px for {} x {} \
             variables); use a larger image or fewer variables to show them",
            lay.cell, n_rows, n_cols
        ));
    }

    // Create image buffer
    let mut buffer = rgb_buffer(width, height)?;
    let mut recorder = SvgRecorder::new(width, height);

    {
        let root =
            FigureBackend::new(&mut buffer, &mut recorder, (width, height)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let plot_area = root.margin(10, 10, 10, 10);

        // Title
        plot_area
            .titled(title, ("sans-serif", 24))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let cell = lay.cell;
        let heatmap_width = cell * n_cols as u32;
        let heatmap_height = cell * n_rows as u32;
        let (x_offset, y_offset) = (lay.x_offset, lay.y_offset);

        // Draw heatmap cells
        for (i, row) in matrix.iter().enumerate() {
            crate::resource::checkpoint()
                .map_err(|e| VisualizationError::InvalidData(e.to_string()))?;
            for (j, &val) in row.iter().enumerate() {
                let x = x_offset + j as u32 * cell;
                let y = y_offset + i as u32 * cell;

                let color = value_to_color(val, color_min, color_max);

                root.draw(&Rectangle::new(
                    [(x as i32, y as i32), ((x + cell) as i32, (y + cell) as i32)],
                    color.filled(),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Cell borders only where they leave the colour visible.
                if cell >= MIN_LABELLED_CELL {
                    root.draw(&Rectangle::new(
                        [(x as i32, y as i32), ((x + cell) as i32, (y + cell) as i32)],
                        BLACK.stroke_width(1),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                }

                // Draw value text if cells are large enough
                if cell >= 35 {
                    let text_color = if val.abs() > 0.5 { WHITE } else { BLACK };
                    // A value that could not be computed (R's NA) shows as NA on its gray cell.
                    let val_text = if val.is_finite() {
                        format!("{:.2}", val)
                    } else {
                        "NA".to_string()
                    };
                    root.draw(&Text::new(
                        val_text,
                        ((x + cell / 2) as i32, (y + cell / 2) as i32),
                        ("sans-serif", 12)
                            .into_font()
                            .color(&text_color)
                            .pos(Pos::new(HPos::Center, VPos::Center)),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                }
            }
        }

        if lay.labels {
            // Labels shortened by characters (a byte cut panics inside a multi-byte
            // character) to what the label band holds at about 8 px a character.
            let max_chars = (lay.label_margin / 8).max(1) as usize;
            let font_size = f64::from(cell.min(11));
            // Row labels, right-aligned against the matrix.
            for (i, label) in row_labels.iter().enumerate() {
                let y = y_offset + i as u32 * cell + cell / 2;
                root.draw(&Text::new(
                    p2a_text::ellipsize(label, max_chars).into_owned(),
                    ((x_offset - 4) as i32, y as i32),
                    ("sans-serif", font_size)
                        .into_font()
                        .color(&BLACK)
                        .pos(Pos::new(HPos::Right, VPos::Center)),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
            // Column labels, rotated, in the band above the matrix, reading upwards.
            for (j, label) in col_labels.iter().enumerate() {
                let x = x_offset + j as u32 * cell + cell / 2;
                root.draw(&Text::new(
                    p2a_text::ellipsize(label, max_chars).into_owned(),
                    (x as i32, (y_offset - 4) as i32),
                    ("sans-serif", font_size)
                        .into_font()
                        .transform(FontTransform::Rotate270)
                        .color(&BLACK)
                        .pos(Pos::new(HPos::Left, VPos::Center)),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        }

        // Draw colorbar
        let colorbar_x = x_offset + heatmap_width + 20;
        let colorbar_width = 20u32;
        let colorbar_height = heatmap_height.min(200);
        let colorbar_y = y_offset + (heatmap_height - colorbar_height) / 2;

        for i in 0..colorbar_height {
            let val = color_max - (color_max - color_min) * i as f64 / colorbar_height as f64;
            let color = value_to_color(val, color_min, color_max);
            let y = colorbar_y + i;
            root.draw(&Rectangle::new(
                [
                    (colorbar_x as i32, y as i32),
                    ((colorbar_x + colorbar_width) as i32, (y + 1) as i32),
                ],
                color.filled(),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        // Colorbar labels
        root.draw(&Text::new(
            format!("{:.1}", color_max),
            (
                (colorbar_x + colorbar_width + 5) as i32,
                (colorbar_y + 5) as i32,
            ),
            ("sans-serif", 11).into_font().color(&BLACK),
        ))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.draw(&Text::new(
            format!("{:.1}", color_min),
            (
                (colorbar_x + colorbar_width + 5) as i32,
                (colorbar_y + colorbar_height) as i32,
            ),
            ("sans-serif", 11).into_font().color(&BLACK),
        ))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, width, height)?;

    Ok(HeatmapResult {
        image_base64,
        svg: recorder.finish(),
        n_rows,
        n_cols,
        row_labels: row_labels.to_vec(),
        col_labels: col_labels.to_vec(),
        warnings,
    })
}

/// Convert a value to a color using a diverging cyan-white-orange colormap.
///
/// Uses brand colors:
/// - Negative values: Brand Cyan (#00B4D8)
/// - Zero/neutral: White
/// - Positive values: Brand Orange (#FF6B35)
fn value_to_color(val: f64, min: f64, max: f64) -> RGBColor {
    if !val.is_finite() {
        return RGBColor(128, 128, 128); // Gray for NaN
    }

    // Normalize to 0-1
    let range = max - min;
    let normalized = if range.abs() < 1e-10 {
        0.5
    } else {
        (val - min) / range
    };

    // Brand colors for endpoints
    // Cyan: (0, 180, 216) - #00B4D8
    // Orange: (255, 107, 53) - #FF6B35

    // Diverging colormap: cyan (0) -> white (0.5) -> orange (1)
    let (r, g, b) = if normalized <= 0.5 {
        // Cyan to white
        let t = normalized * 2.0;
        (
            (0.0 + t * 255.0) as u8,
            (180.0 + t * (255.0 - 180.0)) as u8,
            (216.0 + t * (255.0 - 216.0)) as u8,
        )
    } else {
        // White to orange
        let t = (normalized - 0.5) * 2.0;
        (
            255,
            (255.0 - t * (255.0 - 107.0)) as u8,
            (255.0 - t * (255.0 - 53.0)) as u8,
        )
    };

    RGBColor(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_heatmap() {
        let matrix = vec![
            vec![1.0, 0.5, -0.3],
            vec![0.5, 1.0, 0.2],
            vec![-0.3, 0.2, 1.0],
        ];
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = correlation_heatmap(&matrix, &labels, &labels, None, None, None).unwrap();
        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols, 3);
        assert!(!result.image_base64.is_empty());
    }

    #[test]
    fn small_canvas_does_not_underflow_layout() {
        let m = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let l = vec!["A".to_string(), "B".to_string()];
        // Too small for any cell: refused (not a panic), naming the capacity.
        let e = correlation_heatmap(&m, &l, &l, None, Some(50), Some(50))
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("at most 0 variables fit"), "{e}");
        assert!(correlation_heatmap(&m, &l, &l, None, Some(0), Some(50)).is_err());
        assert!(correlation_heatmap(&m, &l, &l, None, Some(160), Some(110)).is_ok());
    }

    fn labels(k: usize) -> Vec<String> {
        (0..k).map(|i| format!("var{i:02}")).collect()
    }

    /// A k x k matrix with values in [-1, 1].
    fn matrix(k: usize) -> Vec<Vec<f64>> {
        (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if i == j {
                            1.0
                        } else {
                            ((i * 7 + j * 3) % 19) as f64 / 9.5 - 1.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// DATA-63: cells had a 20 px minimum on the fixed canvas, so wide matrices were cut off.
    #[test]
    fn test_heatmap_layout_fits_the_canvas() {
        let (w, h) = (DEFAULT_WIDTH, DEFAULT_HEIGHT);
        for k in [2, 10, 27, 40, 100, 127] {
            for len in [5, 20] {
                let lay = layout(k, k, len, w, h).unwrap();
                let side = lay.cell * k as u32;
                assert!(lay.cell >= MIN_CELL, "{k} {len}");
                // The matrix, then the colour bar (gap, bar, labels) inside the canvas.
                assert!(
                    lay.x_offset + side + COLORBAR_BAND <= w,
                    "{k} {len}: {lay:?}"
                );
                assert!(lay.y_offset + side <= h, "{k} {len}: {lay:?}");
                // The label bands lie inside the canvas, left of and above the matrix.
                assert!(lay.x_offset >= lay.label_margin, "{k} {len}: {lay:?}");
                assert!(lay.y_offset >= TITLE_BAND + lay.label_margin, "{k} {len}");
                // Labels are drawn iff the cells keep at least 10 px with the label bands.
                let margin = (len as u32 * 8).clamp(8, 150);
                let labelled = cell_size(k, k, margin, w, h) >= MIN_LABELLED_CELL;
                assert_eq!(lay.labels, labelled, "{k} {len}: {lay:?}");
                if lay.labels {
                    assert!(lay.cell >= MIN_LABELLED_CELL);
                }
            }
        }
        // Labels of five characters fit up to 48 variables, of 19 or more up to 37.
        assert!(layout(48, 48, 5, w, h).unwrap().labels);
        assert!(!layout(49, 49, 5, w, h).unwrap().labels);
        assert!(layout(37, 37, 19, w, h).unwrap().labels);
        assert!(!layout(38, 38, 19, w, h).unwrap().labels);
        let e = layout(128, 128, 5, w, h).unwrap_err().to_string();
        assert!(
            e.contains("at most 127 variables") && e.contains("a larger image or fewer variables"),
            "{e}"
        );
        assert_eq!(heatmap_capacity(None, None), 127);
        assert_eq!(heatmap_capacity(Some(1200), Some(1000)), 227);
        let r = correlation_heatmap(&matrix(128), &labels(128), &labels(128), None, None, None);
        assert!(r.is_err());
    }

    /// The cell's colour at the centre of cell (39, 39) and a coloured bar: at HEAD row 39 and
    /// the colour bar were off the canvas.
    #[test]
    fn test_heatmap_draws_every_cell() {
        use base64::Engine;
        let k = 40;
        let (m, l) = (matrix(k), labels(k));
        let r = correlation_heatmap(&m, &l, &l, None, None, None).unwrap();
        assert_eq!((r.n_rows, r.n_cols), (k, k));
        let lay = layout(k, k, 5, DEFAULT_WIDTH, DEFAULT_HEIGHT).unwrap();
        let png = base64::engine::general_purpose::STANDARD
            .decode(&r.image_base64)
            .unwrap();
        let img = image::load_from_memory(&png).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (DEFAULT_WIDTH, DEFAULT_HEIGHT));
        let centre = |i: usize, j: usize| {
            let x = lay.x_offset + j as u32 * lay.cell + lay.cell / 2;
            let y = lay.y_offset + i as u32 * lay.cell + lay.cell / 2;
            img.get_pixel(x, y).0
        };
        for (i, j) in [(0, 0), (39, 39), (39, 0), (0, 39), (17, 23)] {
            let RGBColor(r, g, b) = value_to_color(m[i][j], -1.0, 1.0);
            assert_eq!(centre(i, j), [r, g, b], "cell ({i}, {j})");
        }
        // The colour bar's top (+1) and bottom (-1) ends.
        let bar_x = lay.x_offset + lay.cell * k as u32 + 30;
        let bar_h = (lay.cell * k as u32).min(200);
        let bar_y = lay.y_offset + (lay.cell * k as u32 - bar_h) / 2;
        let top = img.get_pixel(bar_x, bar_y + 1).0;
        let bottom = img.get_pixel(bar_x, bar_y + bar_h - 2).0;
        assert_ne!(top, [255, 255, 255]);
        assert_ne!(bottom, [255, 255, 255]);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        // Past 48 variables the labels are hidden and the result says so.
        let r =
            correlation_heatmap(&matrix(60), &labels(60), &labels(60), None, None, None).unwrap();
        assert!(r.warnings[0].starts_with("labels hidden: cells are below 10 px"));
    }

    /// DATA-26 (column site): labels over 8 bytes were cut with &label[..5]; 'ß' spans bytes 4..6 of
    /// "Größe_kg" and '入' bytes 3..6 of "収入合計額". All three labels are <= 15 bytes, so the row site is not hit.
    #[test]
    fn test_correlation_heatmap_non_ascii_column_labels() {
        let m = vec![
            vec![1.0, 0.42, -0.17],
            vec![0.42, 1.0, 0.31],
            vec![-0.17, 0.31, 1.0],
        ];
        let labels: Vec<String> = ["Größe_kg", "収入合計額", "R²_adj"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let r = correlation_heatmap(&m, &labels, &labels, None, None, None).unwrap();
        assert_eq!((r.n_rows, r.n_cols), (3, 3));
        assert_eq!(r.col_labels, labels); // the result keeps the full names
        assert!(!r.image_base64.is_empty());
    }

    /// DATA-26 (row site): labels over 15 bytes were cut with &label[..12]; 'ö' spans bytes 11..13.
    #[test]
    fn test_correlation_heatmap_non_ascii_row_labels() {
        let m = vec![
            vec![1.0, 0.27, 0.05],
            vec![0.27, 1.0, -0.38],
            vec![0.05, -0.38, 1.0],
        ];
        let labels: Vec<String> = ["Haushaltsgrößenklasse", "Einkommen", "Alter"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let r = correlation_heatmap(&m, &labels, &labels, Some("Korrelation"), None, None).unwrap();
        assert_eq!(r.row_labels, labels);
        assert!(!r.image_base64.is_empty());
    }
}
