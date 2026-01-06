//! Chart generation: histograms, scatter plots, box plots, line charts.

use super::{VisualizationError, DEFAULT_WIDTH, DEFAULT_HEIGHT};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use plotters::prelude::*;

/// Configuration for chart generation.
#[derive(Debug, Clone)]
pub struct ChartConfig {
    pub width: u32,
    pub height: u32,
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            title: None,
            x_label: None,
            y_label: None,
        }
    }
}

/// Result of histogram generation.
pub struct HistogramResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of bins used
    pub bins: usize,
    /// Minimum value in data
    pub min: f64,
    /// Maximum value in data
    pub max: f64,
    /// Mean value
    pub mean: f64,
}

impl std::fmt::Display for HistogramResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Histogram")?;
        writeln!(f, "=========")?;
        writeln!(f, "Bins: {}", self.bins)?;
        writeln!(f, "Range: [{:.4}, {:.4}]", self.min, self.max)?;
        writeln!(f, "Mean: {:.4}", self.mean)?;
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a histogram from numeric data.
///
/// # Arguments
/// * `data` - Numeric values to plot
/// * `bins` - Number of bins (default: auto-calculated)
/// * `config` - Chart configuration
pub fn histogram(
    data: &[f64],
    bins: Option<usize>,
    config: ChartConfig,
) -> Result<HistogramResult, VisualizationError> {
    if data.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data array".to_string()));
    }

    // Filter out NaN values
    let clean_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if clean_data.is_empty() {
        return Err(VisualizationError::InvalidData("No finite values in data".to_string()));
    }

    let min_val = clean_data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = clean_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_val: f64 = clean_data.iter().sum::<f64>() / clean_data.len() as f64;

    // Auto-calculate bins using Sturges' rule if not specified
    let n_bins = bins.unwrap_or_else(|| {
        let n = clean_data.len() as f64;
        (1.0 + 3.322 * n.log10()).ceil() as usize
    }).max(1);

    // Calculate bin edges and counts
    let bin_width = if (max_val - min_val).abs() < 1e-10 {
        1.0
    } else {
        (max_val - min_val) / n_bins as f64
    };

    let mut counts = vec![0u32; n_bins];

    for &val in &clean_data {
        let bin_idx = if bin_width > 0.0 {
            ((val - min_val) / bin_width).floor() as usize
        } else {
            0
        };
        let bin_idx = bin_idx.min(n_bins - 1);
        counts[bin_idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Histogram");
        let x_label = config.x_label.as_deref().unwrap_or("Value");
        let y_label = config.y_label.as_deref().unwrap_or("Frequency");

        let x_range = min_val..(max_val + bin_width * 0.01);
        let y_range = 0u32..(max_count + max_count / 10 + 1);

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw histogram bars as rectangles
        for (i, &count) in counts.iter().enumerate() {
            let x0 = min_val + i as f64 * bin_width;
            let x1 = x0 + bin_width;

            chart.draw_series(std::iter::once(
                Rectangle::new([(x0, 0), (x1, count)], BLUE.mix(0.7).filled())
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    // Encode as PNG then base64
    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(HistogramResult {
        image_base64,
        bins: n_bins,
        min: min_val,
        max: max_val,
        mean: mean_val,
    })
}

/// Result of scatter plot generation.
pub struct ScatterResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of points plotted
    pub n_points: usize,
    /// Correlation coefficient (if calculated)
    pub correlation: Option<f64>,
}

impl std::fmt::Display for ScatterResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Scatter Plot")?;
        writeln!(f, "============")?;
        writeln!(f, "Points: {}", self.n_points)?;
        if let Some(r) = self.correlation {
            writeln!(f, "Correlation: {:.4}", r)?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a scatter plot from two numeric arrays.
///
/// # Arguments
/// * `x` - X-axis values
/// * `y` - Y-axis values
/// * `config` - Chart configuration
pub fn scatter_plot(
    x: &[f64],
    y: &[f64],
    config: ChartConfig,
) -> Result<ScatterResult, VisualizationError> {
    if x.len() != y.len() {
        return Err(VisualizationError::InvalidData(
            format!("X and Y must have same length: {} vs {}", x.len(), y.len())
        ));
    }

    if x.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data arrays".to_string()));
    }

    // Filter out pairs with NaN
    let points: Vec<(f64, f64)> = x.iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(xi, yi)| (*xi, *yi))
        .collect();

    if points.is_empty() {
        return Err(VisualizationError::InvalidData("No finite value pairs".to_string()));
    }

    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    // Add 5% padding
    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.05;

    // Calculate correlation
    let correlation = calculate_correlation(&points);

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Scatter Plot");
        let x_label = config.x_label.as_deref().unwrap_or("X");
        let y_label = config.y_label.as_deref().unwrap_or("Y");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw points
        chart
            .draw_series(
                points.iter().map(|(xi, yi)| {
                    Circle::new((*xi, *yi), 4, BLUE.mix(0.7).filled())
                }),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(ScatterResult {
        image_base64,
        n_points: points.len(),
        correlation,
    })
}

/// Result of box plot generation.
pub struct BoxPlotResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Statistics for each group
    pub statistics: Vec<BoxPlotStats>,
}

/// Statistics for a single box in the box plot.
#[derive(Debug, Clone)]
pub struct BoxPlotStats {
    pub label: String,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
}

impl std::fmt::Display for BoxPlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Box Plot")?;
        writeln!(f, "========")?;
        for stat in &self.statistics {
            writeln!(f, "{}: min={:.2}, Q1={:.2}, med={:.2}, Q3={:.2}, max={:.2}",
                stat.label, stat.min, stat.q1, stat.median, stat.q3, stat.max)?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a box plot from multiple data groups.
///
/// # Arguments
/// * `groups` - Vector of (label, data) pairs
/// * `config` - Chart configuration
pub fn box_plot(
    groups: &[(String, Vec<f64>)],
    config: ChartConfig,
) -> Result<BoxPlotResult, VisualizationError> {
    if groups.is_empty() {
        return Err(VisualizationError::InvalidData("No groups provided".to_string()));
    }

    // Calculate statistics for each group
    let mut statistics = Vec::new();
    let mut all_values = Vec::new();

    for (label, data) in groups {
        let clean: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if clean.is_empty() {
            continue;
        }

        let stats = calculate_box_stats(label.clone(), &clean);
        all_values.extend(clean.iter().copied());
        statistics.push(stats);
    }

    if statistics.is_empty() {
        return Err(VisualizationError::InvalidData("No valid data in groups".to_string()));
    }

    let y_min = all_values.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Box Plot");
        let y_label = config.y_label.as_deref().unwrap_or("Value");

        // Build x-axis with continuous coordinates (0 to n_groups)
        let labels: Vec<String> = statistics.iter().map(|s| s.label.clone()).collect();
        let n_groups = labels.len() as f64;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                -0.5..(n_groups - 0.5),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .y_desc(y_label)
            .x_label_formatter(&|x| {
                let idx = x.round() as usize;
                labels.get(idx).cloned().unwrap_or_default()
            })
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw box plots manually
        for (i, stat) in statistics.iter().enumerate() {
            let box_half_width = 0.3;
            let x_center = i as f64;
            let x_left = x_center - box_half_width;
            let x_right = x_center + box_half_width;

            // Draw box (Q1 to Q3)
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_left, stat.q1), (x_right, stat.q3)],
                BLUE.mix(0.5).filled(),
            ))).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw box border
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_left, stat.q1), (x_right, stat.q3)],
                BLACK.stroke_width(1),
            ))).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw median line
            chart.draw_series(LineSeries::new(
                vec![(x_left, stat.median), (x_right, stat.median)],
                BLACK.stroke_width(2),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw whiskers
            // Lower whisker (vertical line)
            chart.draw_series(LineSeries::new(
                vec![(x_center, stat.min), (x_center, stat.q1)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            // Lower whisker cap
            chart.draw_series(LineSeries::new(
                vec![(x_left + 0.1, stat.min), (x_right - 0.1, stat.min)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Upper whisker (vertical line)
            chart.draw_series(LineSeries::new(
                vec![(x_center, stat.q3), (x_center, stat.max)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            // Upper whisker cap
            chart.draw_series(LineSeries::new(
                vec![(x_left + 0.1, stat.max), (x_right - 0.1, stat.max)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw outliers
            chart.draw_series(
                stat.outliers.iter().map(|&outlier| {
                    Circle::new((x_center, outlier), 3, RED.filled())
                })
            ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(BoxPlotResult {
        image_base64,
        statistics,
    })
}

/// Result of line chart generation.
pub struct LineChartResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of data points
    pub n_points: usize,
    /// Number of series
    pub n_series: usize,
}

impl std::fmt::Display for LineChartResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Line Chart")?;
        writeln!(f, "==========")?;
        writeln!(f, "Series: {}", self.n_series)?;
        writeln!(f, "Points: {}", self.n_points)?;
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a line chart from one or more time series.
///
/// # Arguments
/// * `series` - Vector of (label, x_values, y_values) tuples
/// * `config` - Chart configuration
pub fn line_chart(
    series: &[(String, Vec<f64>, Vec<f64>)],
    config: ChartConfig,
) -> Result<LineChartResult, VisualizationError> {
    if series.is_empty() {
        return Err(VisualizationError::InvalidData("No series provided".to_string()));
    }

    // Find global bounds
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut total_points = 0;

    for (_, x_vals, y_vals) in series {
        for (xi, yi) in x_vals.iter().zip(y_vals.iter()) {
            if xi.is_finite() && yi.is_finite() {
                x_min = x_min.min(*xi);
                x_max = x_max.max(*xi);
                y_min = y_min.min(*yi);
                y_max = y_max.max(*yi);
                total_points += 1;
            }
        }
    }

    if total_points == 0 {
        return Err(VisualizationError::InvalidData("No valid data points".to_string()));
    }

    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.05;

    // Colors for multiple series
    let colors = [BLUE, RED, GREEN, MAGENTA, CYAN, BLACK];

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Line Chart");
        let x_label = config.x_label.as_deref().unwrap_or("X");
        let y_label = config.y_label.as_deref().unwrap_or("Y");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw each series
        for (idx, (label, x_vals, y_vals)) in series.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let points: Vec<(f64, f64)> = x_vals.iter()
                .zip(y_vals.iter())
                .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
                .map(|(xi, yi)| (*xi, *yi))
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));
        }

        // Draw legend if multiple series
        if series.len() > 1 {
            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(LineChartResult {
        image_base64,
        n_points: total_points,
        n_series: series.len(),
    })
}

// Helper functions

fn calculate_correlation(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|p| p.0).sum();
    let sum_y: f64 = points.iter().map(|p| p.1).sum();
    let sum_xy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let sum_x2: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sum_y2: f64 = points.iter().map(|p| p.1 * p.1).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator.abs() < 1e-10 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn calculate_box_stats(label: String, data: &[f64]) -> BoxPlotStats {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let q1_idx = n / 4;
    let median_idx = n / 2;
    let q3_idx = 3 * n / 4;

    let q1 = sorted[q1_idx];
    let median = if n % 2 == 0 && n > 1 {
        (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
    } else {
        sorted[median_idx]
    };
    let q3 = sorted[q3_idx.min(n - 1)];

    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let min = sorted.iter().copied().find(|&x| x >= lower_fence).unwrap_or(sorted[0]);
    let max = sorted.iter().rev().copied().find(|&x| x <= upper_fence).unwrap_or(sorted[n - 1]);

    let outliers: Vec<f64> = sorted.iter()
        .copied()
        .filter(|&x| x < lower_fence || x > upper_fence)
        .collect();

    BoxPlotStats { label, min, q1, median, q3, max, outliers }
}

fn encode_rgb_to_png_base64(rgb_buffer: &[u8], width: u32, height: u32) -> Result<String, VisualizationError> {
    use image::{ImageBuffer, Rgb, ImageEncoder, codecs::png::PngEncoder, ColorType};
    use std::io::Cursor;

    // Create image from RGB buffer
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_buffer.to_vec())
        .ok_or_else(|| VisualizationError::EncodingError("Failed to create image buffer".to_string()))?;

    // Encode to PNG
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
    fn test_histogram() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = histogram(&data, Some(10), ChartConfig::default()).unwrap();
        assert_eq!(result.bins, 10);
        assert!(!result.image_base64.is_empty());
    }

    #[test]
    fn test_scatter() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + 1.0).collect();
        let result = scatter_plot(&x, &y, ChartConfig::default()).unwrap();
        assert_eq!(result.n_points, 50);
        assert!(result.correlation.unwrap() > 0.99);
    }

    #[test]
    fn test_line_chart() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let series = vec![("sin(x)".to_string(), x, y)];
        let result = line_chart(&series, ChartConfig::default()).unwrap();
        assert_eq!(result.n_series, 1);
    }
}
