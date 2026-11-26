//! Vector versions of the static charts.
//!
//! [`FigureBackend`] hands every drawing call to plotters' bitmap backend, so the PNG is the
//! one the bitmap backend alone draws, and records the shapes it can as a small SVG document
//! in [`SvgRecorder`]. Coordinates are plotters' integer pixels.
//!
//! The SVG is written for one reader besides browsers: the app's PDF export parses exactly
//! what is emitted here (`g`, `rect`, `circle`, `path`, `text`; see `p2a-dioxus`,
//! `figure_pdf.rs`). Keep the two in step.

use plotters::backend::BitMapBackend;
use plotters_backend::{
    BackendColor, BackendCoord, BackendStyle, BackendTextStyle, DrawingBackend, DrawingErrorKind,
    FontTransform,
    text_anchor::{HPos, VPos},
};
use std::fmt::Write as _;

/// Largest SVG kept, in bytes; a chart that needs more has only its PNG.
pub const MAX_SVG_BYTES: usize = 1024 * 1024;

/// Font families of the chart text, metric-compatible ones first.
const FONT_FAMILY: &str = "Arimo, 'Liberation Sans', Arial, Helvetica, sans-serif";

/// An SVG document under construction.
pub(super) struct SvgRecorder {
    svg: String,
    /// Paint of the open `g`, which shapes in one style share.
    group: Option<Paint>,
    /// False once the document is too large or a call could not be recorded.
    complete: bool,
}

/// How a shape is painted: filled, or stroked with a width.
#[derive(Clone, Copy, PartialEq)]
struct Paint {
    color: BackendColorKey,
    stroke: Option<u32>,
}

/// A colour: its components and opacity.
type BackendColorKey = ((u8, u8, u8), f64);

impl Paint {
    fn fill(c: BackendColor) -> Self {
        Self {
            color: (c.rgb, c.alpha.clamp(0.0, 1.0)),
            stroke: None,
        }
    }

    fn stroke(c: BackendColor, width: u32) -> Self {
        Self {
            stroke: Some(width.max(1)),
            ..Self::fill(c)
        }
    }

    /// The paint of a shape that is filled or only outlined.
    fn of<S: BackendStyle>(style: &S, fill: bool) -> Self {
        if fill {
            Self::fill(style.color())
        } else {
            Self::stroke(style.color(), style.stroke_width())
        }
    }

    /// Open a `g` with the paint's attributes.
    fn open(&self, svg: &mut String) {
        let ((r, g, b), alpha) = self.color;
        let role = if self.stroke.is_some() {
            svg.push_str("<g fill=\"none\" ");
            "stroke"
        } else {
            svg.push_str("<g ");
            "fill"
        };
        let _ = write!(svg, "{role}=\"#{r:02x}{g:02x}{b:02x}\"");
        if alpha < 1.0 {
            let _ = write!(svg, " {role}-opacity=\"{alpha:.3}\"");
        }
        if let Some(width) = self.stroke {
            let _ = write!(svg, " stroke-width=\"{width}\"");
        }
        svg.push('>');
    }
}

impl SvgRecorder {
    pub(super) fn new(width: u32, height: u32) -> Self {
        let mut svg = String::new();
        let _ = write!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" \
             viewBox=\"0 0 {width} {height}\" font-family=\"{FONT_FAMILY}\" \
             stroke-linejoin=\"round\">"
        );
        Self {
            svg,
            group: None,
            complete: true,
        }
    }

    /// The finished document, or None when the chart could not be recorded whole.
    pub(super) fn finish(mut self) -> Option<String> {
        if !self.complete {
            return None;
        }
        if self.group.is_some() {
            self.svg.push_str("</g>");
        }
        self.svg.push_str("</svg>");
        Some(self.svg)
    }

    /// Give up on the vector version and free what was recorded.
    fn abandon(&mut self) {
        self.complete = false;
        self.svg = String::new();
    }

    fn shape(&mut self, paint: Paint, write: impl FnOnce(&mut String)) {
        if !self.complete {
            return;
        }
        if self.group != Some(paint) {
            if self.group.is_some() {
                self.svg.push_str("</g>");
            }
            paint.open(&mut self.svg);
            self.group = Some(paint);
        }
        write(&mut self.svg);
        if self.svg.len() > MAX_SVG_BYTES {
            self.abandon();
        }
    }

    fn path(&mut self, paint: Paint, points: impl IntoIterator<Item = BackendCoord>, closed: bool) {
        self.shape(paint, |svg| {
            svg.push_str("<path d=\"");
            for (i, (x, y)) in points.into_iter().enumerate() {
                let _ = write!(svg, "{}{x} {y}", if i == 0 { 'M' } else { 'L' });
                if svg.len() > MAX_SVG_BYTES {
                    break;
                }
            }
            svg.push_str(if closed { "Z\"/>" } else { "\"/>" });
        });
    }

    fn rect(&mut self, paint: Paint, a: BackendCoord, b: BackendCoord) {
        let (x, y) = (a.0.min(b.0), a.1.min(b.1));
        let (w, h) = (a.0.abs_diff(b.0), a.1.abs_diff(b.1));
        self.shape(paint, |svg| {
            let _ = write!(
                svg,
                "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\"/>"
            );
        });
    }

    fn text<S: BackendTextStyle>(&mut self, text: &str, style: &S, pos: BackendCoord) {
        let (x, y) = pos;
        // plotters' font sizes are line heights; its own SVG backend divides by 1.24 too.
        let size = style.size() / 1.24;
        let anchor = match style.anchor().h_pos {
            HPos::Left => "start",
            HPos::Center => "middle",
            HPos::Right => "end",
        };
        // The baseline below the anchor point, in shares of the font size (an SVG `dy`
        // written out, because the PDF export places glyphs on the baseline).
        let baseline = match style.anchor().v_pos {
            VPos::Top => 0.80,
            VPos::Center => 0.34,
            VPos::Bottom => -0.22,
        };
        let angle = match style.transform() {
            FontTransform::None => 0,
            FontTransform::Rotate90 => 90,
            FontTransform::Rotate180 => 180,
            FontTransform::Rotate270 => 270,
        };
        self.shape(Paint::fill(style.color()), |svg| {
            let _ = write!(
                svg,
                "<text x=\"{x}\" y=\"{:.2}\" font-size=\"{size:.2}\" text-anchor=\"{anchor}\"",
                f64::from(y) + baseline * size
            );
            if angle != 0 {
                let _ = write!(svg, " transform=\"rotate({angle} {x} {y})\"");
            }
            svg.push('>');
            for c in text.chars() {
                match c {
                    '&' => svg.push_str("&amp;"),
                    '<' => svg.push_str("&lt;"),
                    '>' => svg.push_str("&gt;"),
                    '"' => svg.push_str("&quot;"),
                    // Not allowed in XML 1.0, or a line break the PDF reader does not know.
                    c if c.is_control() || c == '\u{fffe}' || c == '\u{ffff}' => svg.push(' '),
                    c => svg.push(c),
                }
            }
            svg.push_str("</text>");
        });
    }
}

/// The bitmap backend, with a recording of what it draws.
pub(super) struct FigureBackend<'a> {
    bitmap: BitMapBackend<'a>,
    svg: &'a mut SvgRecorder,
}

impl<'a> FigureBackend<'a> {
    pub(super) fn new(buffer: &'a mut [u8], svg: &'a mut SvgRecorder, size: (u32, u32)) -> Self {
        Self {
            bitmap: BitMapBackend::with_buffer(buffer, size),
            svg,
        }
    }
}

type Drawn<'a> = Result<(), DrawingErrorKind<<BitMapBackend<'a> as DrawingBackend>::ErrorType>>;

impl<'a> DrawingBackend for FigureBackend<'a> {
    type ErrorType = <BitMapBackend<'a> as DrawingBackend>::ErrorType;

    fn get_size(&self) -> (u32, u32) {
        self.bitmap.get_size()
    }

    fn ensure_prepared(&mut self) -> Drawn<'a> {
        self.bitmap.ensure_prepared()
    }

    fn present(&mut self) -> Drawn<'a> {
        self.bitmap.present()
    }

    fn draw_pixel(&mut self, point: BackendCoord, color: BackendColor) -> Drawn<'a> {
        if color.alpha > 0.0 {
            self.svg
                .rect(Paint::fill(color), point, (point.0 + 1, point.1 + 1));
        }
        self.bitmap.draw_pixel(point, color)
    }

    fn draw_line<S: BackendStyle>(
        &mut self,
        from: BackendCoord,
        to: BackendCoord,
        style: &S,
    ) -> Drawn<'a> {
        let color = style.color();
        if color.alpha > 0.0 {
            self.svg.path(
                Paint::stroke(color, style.stroke_width()),
                [from, to],
                false,
            );
        }
        self.bitmap.draw_line(from, to, style)
    }

    fn draw_rect<S: BackendStyle>(
        &mut self,
        upper_left: BackendCoord,
        bottom_right: BackendCoord,
        style: &S,
        fill: bool,
    ) -> Drawn<'a> {
        let color = style.color();
        if color.alpha > 0.0 {
            self.svg
                .rect(Paint::of(style, fill), upper_left, bottom_right);
        }
        self.bitmap.draw_rect(upper_left, bottom_right, style, fill)
    }

    fn draw_path<S: BackendStyle, I: IntoIterator<Item = BackendCoord>>(
        &mut self,
        path: I,
        style: &S,
    ) -> Drawn<'a> {
        let color = style.color();
        if color.alpha <= 0.0 {
            return self.bitmap.draw_path(path, style);
        }
        let points: Vec<BackendCoord> = path.into_iter().collect();
        self.svg.path(
            Paint::stroke(color, style.stroke_width()),
            points.iter().copied(),
            false,
        );
        self.bitmap.draw_path(points, style)
    }

    fn draw_circle<S: BackendStyle>(
        &mut self,
        center: BackendCoord,
        radius: u32,
        style: &S,
        fill: bool,
    ) -> Drawn<'a> {
        let color = style.color();
        if color.alpha > 0.0 {
            let (x, y) = center;
            // The bitmap's discs take in the pixels at their radius.
            let half = if fill { ".5" } else { "" };
            self.svg.shape(Paint::of(style, fill), |svg| {
                let _ = write!(svg, "<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius}{half}\"/>");
            });
        }
        self.bitmap.draw_circle(center, radius, style, fill)
    }

    fn fill_polygon<S: BackendStyle, I: IntoIterator<Item = BackendCoord>>(
        &mut self,
        vert: I,
        style: &S,
    ) -> Drawn<'a> {
        let color = style.color();
        let points: Vec<BackendCoord> = vert.into_iter().collect();
        if color.alpha > 0.0 {
            self.svg
                .path(Paint::fill(color), points.iter().copied(), true);
        }
        self.bitmap.fill_polygon(points, style)
    }

    fn draw_text<S: BackendTextStyle>(
        &mut self,
        text: &str,
        style: &S,
        pos: BackendCoord,
    ) -> Drawn<'a> {
        if style.color().alpha > 0.0 {
            self.svg.text(text, style, pos);
        }
        self.bitmap.draw_text(text, style, pos)
    }

    fn estimate_text_size<S: BackendTextStyle>(
        &self,
        text: &str,
        style: &S,
    ) -> Result<(u32, u32), DrawingErrorKind<Self::ErrorType>> {
        self.bitmap.estimate_text_size(text, style)
    }

    fn blit_bitmap(&mut self, pos: BackendCoord, size: (u32, u32), src: &[u8]) -> Drawn<'a> {
        // No chart draws one; a vector version without it would be wrong.
        self.svg.abandon();
        self.bitmap.blit_bitmap(pos, size, src)
    }
}
