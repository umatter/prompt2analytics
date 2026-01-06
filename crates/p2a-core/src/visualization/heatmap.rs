//! Heatmap visualization for correlation matrices and other 2D data.

use super::{VisualizationError, DEFAULT_WIDTH, DEFAULT_HEIGHT};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use plotters::prelude::*;

/// Result of heatmap generation.
pub struct HeatmapResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row labels
    pub row_labels: Vec<String>,
    /// Column labels
    pub col_labels: Vec<String>,
}

impl std::fmt::Display for HeatmapResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Heatmap")?;
        writeln!(f, "=======")?;
        writeln!(f, "Dimensions: {} x {}", self.n_rows, self.n_cols)?;
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
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
        return Err(VisualizationError::InvalidData(
            format!("Label count mismatch: {}x{} matrix but {} row labels and {} col labels",
                n_rows, n_cols, row_labels.len(), col_labels.len())
        ));
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

    // Create image buffer
    let mut buffer = vec![0u8; (width * height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width, height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Calculate layout - need space for labels
        let max_label_len = row_labels.iter().chain(col_labels.iter())
            .map(|s| s.len())
            .max()
            .unwrap_or(5);
        let label_margin = (max_label_len * 8).min(150) as u32;

        let plot_area = root.margin(10, 10, 10, 10);

        // Title
        plot_area.titled(title, ("sans-serif", 24))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Calculate cell size
        let available_width = width - 2 * label_margin - 60;  // Extra for colorbar
        let available_height = height - label_margin - 80;  // Title + margin
        let cell_width = (available_width / n_cols as u32).max(20);
        let cell_height = (available_height / n_rows as u32).max(20);

        let heatmap_width = cell_width * n_cols as u32;
        let heatmap_height = cell_height * n_rows as u32;

        let x_offset = label_margin + 20;
        let y_offset = 60u32;

        // Draw heatmap cells
        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let x = x_offset + j as u32 * cell_width;
                let y = y_offset + i as u32 * cell_height;

                let color = value_to_color(val, color_min, color_max);

                root.draw(&Rectangle::new(
                    [(x as i32, y as i32), ((x + cell_width) as i32, (y + cell_height) as i32)],
                    color.filled(),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw cell border
                root.draw(&Rectangle::new(
                    [(x as i32, y as i32), ((x + cell_width) as i32, (y + cell_height) as i32)],
                    BLACK.stroke_width(1),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw value text if cells are large enough
                if cell_width >= 35 && cell_height >= 20 {
                    let text_color = if val.abs() > 0.5 { WHITE } else { BLACK };
                    let val_text = format!("{:.2}", val);
                    root.draw(&Text::new(
                        val_text,
                        ((x + cell_width / 2) as i32, (y + cell_height / 2 + 4) as i32),
                        ("sans-serif", 12).into_font().color(&text_color),
                    )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                }
            }
        }

        // Draw row labels (left side)
        for (i, label) in row_labels.iter().enumerate() {
            let y = y_offset + i as u32 * cell_height + cell_height / 2;
            let truncated = if label.len() > 15 {
                format!("{}...", &label[..12])
            } else {
                label.clone()
            };
            root.draw(&Text::new(
                truncated,
                ((x_offset - 5) as i32, (y + 4) as i32),
                ("sans-serif", 11).into_font().color(&BLACK).transform(FontTransform::None),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        // Draw column labels (top, rotated would be ideal but we'll use abbreviated)
        for (j, label) in col_labels.iter().enumerate() {
            let x = x_offset + j as u32 * cell_width + cell_width / 2;
            let truncated = if label.len() > 8 {
                format!("{}...", &label[..5])
            } else {
                label.clone()
            };
            root.draw(&Text::new(
                truncated,
                (x as i32, (y_offset - 5) as i32),
                ("sans-serif", 10).into_font().color(&BLACK),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
                [(colorbar_x as i32, y as i32), ((colorbar_x + colorbar_width) as i32, (y + 1) as i32)],
                color.filled(),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        // Colorbar labels
        root.draw(&Text::new(
            format!("{:.1}", color_max),
            ((colorbar_x + colorbar_width + 5) as i32, (colorbar_y + 5) as i32),
            ("sans-serif", 11).into_font().color(&BLACK),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.draw(&Text::new(
            format!("{:.1}", color_min),
            ((colorbar_x + colorbar_width + 5) as i32, (colorbar_y + colorbar_height) as i32),
            ("sans-serif", 11).into_font().color(&BLACK),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, width, height)?;

    Ok(HeatmapResult {
        image_base64,
        n_rows,
        n_cols,
        row_labels: row_labels.to_vec(),
        col_labels: col_labels.to_vec(),
    })
}

/// Convert a value to a color using a diverging blue-white-red colormap.
fn value_to_color(val: f64, min: f64, max: f64) -> RGBColor {
    if !val.is_finite() {
        return RGBColor(128, 128, 128);  // Gray for NaN
    }

    // Normalize to 0-1
    let range = max - min;
    let normalized = if range.abs() < 1e-10 {
        0.5
    } else {
        (val - min) / range
    };

    // Diverging colormap: blue (0) -> white (0.5) -> red (1)
    let (r, g, b) = if normalized <= 0.5 {
        // Blue to white
        let t = normalized * 2.0;
        (
            (66.0 + t * (255.0 - 66.0)) as u8,
            (136.0 + t * (255.0 - 136.0)) as u8,
            (181.0 + t * (255.0 - 181.0)) as u8,
        )
    } else {
        // White to red
        let t = (normalized - 0.5) * 2.0;
        (
            255,
            (255.0 - t * (255.0 - 77.0)) as u8,
            (255.0 - t * (255.0 - 77.0)) as u8,
        )
    };

    RGBColor(r, g, b)
}

fn encode_rgb_to_png_base64(rgb_buffer: &[u8], width: u32, height: u32) -> Result<String, VisualizationError> {
    use image::{ImageBuffer, Rgb, ImageEncoder, codecs::png::PngEncoder, ColorType};
    use std::io::Cursor;

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_buffer.to_vec())
        .ok_or_else(|| VisualizationError::EncodingError("Failed to create image buffer".to_string()))?;

    let mut png_data = Vec::new();
    let encoder = PngEncoder::new(Cursor::new(&mut png_data));
    encoder.write_image(
        img.as_raw(),
        width,
        height,
        ColorType::Rgb8,
    ).map_err(|e: image::ImageError| VisualizationError::EncodingError(e.to_string()))?;

    Ok(BASE64.encode(&png_data))
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
}
