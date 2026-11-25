//! Figures as PDF files, written in the browser or the app.
//!
//! [`from_svg`] turns the SVG a chart tool returns into a vector page: text stays text, in an
//! embedded font. It reads only what p2a writes (`p2a-core`, `visualization/vector.rs`) and
//! answers None for anything else, such as a character the font has no glyph for; the caller
//! then uses [`from_png`], which puts the PNG on a page unchanged.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The font of the text (`assets/fonts/README.md`).
const FONT: &[u8] = include_bytes!("../assets/fonts/arimo-regular.ttf");

/// Points per image pixel: an 800 px chart is 600 pt wide.
const SCALE: f64 = 0.75;

/// Bezier handle length of a quarter circle, as a share of the radius.
const ARC: f64 = 0.552_284_75;

/// A vector PDF page of a chart's SVG, or None when it holds something this reader does not
/// know or the font cannot show.
pub fn from_svg(svg: &str) -> Option<Vec<u8>> {
    let face = ttf_parser::Face::parse(FONT, 0).ok()?;
    let mut page = Page {
        face: &face,
        content: String::new(),
        opacities: Vec::new(),
        glyphs: BTreeMap::new(),
    };
    let (width, height) = page.read(svg)?;
    let Page {
        content,
        opacities,
        glyphs,
        ..
    } = page;

    let mut pdf = Pdf::new();
    let mut resources = String::new();
    if !opacities.is_empty() {
        resources.push_str("/ExtGState <<");
        for (i, o) in opacities.iter().enumerate() {
            let _ = write!(resources, " /G{i} << /ca {o} /CA {o} >>");
        }
        resources.push_str(" >> ");
    }
    if !glyphs.is_empty() {
        let font = pdf.font(&face, &glyphs);
        let _ = write!(resources, "/Font << /F0 {font} 0 R >> ");
    }
    let content = format!(
        "{} 0 0 {} 0 {} cm 1 j\n{content}",
        num(SCALE),
        num(-SCALE),
        num(height * SCALE)
    );
    Some(pdf.finish(
        width * SCALE,
        height * SCALE,
        &resources,
        content.as_bytes(),
    ))
}

/// A PDF page with a PNG on it, pixel for pixel. Reads 8-bit RGB without interlacing, which
/// is what the chart tools write; None for any other PNG.
pub fn from_png(png: &[u8]) -> Option<Vec<u8>> {
    let mut rest = png.strip_prefix(b"\x89PNG\r\n\x1a\n")?;
    let (mut size, mut data) = (None, Vec::new());
    while rest.len() >= 12 {
        let len = u32::from_be_bytes(rest[..4].try_into().ok()?) as usize;
        let (kind, body) = (&rest[4..8], rest.get(8..8usize.checked_add(len)?)?);
        match kind {
            b"IHDR" => {
                // 8 bits, RGB, deflate, adaptive filters, not interlaced.
                if body.get(8..13)? != [8, 2, 0, 0, 0] {
                    return None;
                }
                let w = u32::from_be_bytes(body[..4].try_into().ok()?);
                let h = u32::from_be_bytes(body[4..8].try_into().ok()?);
                size = Some((w, h));
            }
            b"IDAT" => data.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        rest = rest.get(len + 12..)?;
    }
    let (w, h) = size.filter(|&(w, h)| w > 0 && h > 0)?;
    if data.is_empty() {
        return None;
    }
    let mut pdf = Pdf::new();
    let image = pdf.stream(
        &format!(
            "/Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB \
             /BitsPerComponent 8 /Filter /FlateDecode /DecodeParms << /Predictor 15 /Colors 3 \
             /BitsPerComponent 8 /Columns {w} >>"
        ),
        &data,
    );
    let (pw, ph) = (f64::from(w) * SCALE, f64::from(h) * SCALE);
    let content = format!("{} 0 0 {} 0 0 cm /Im0 Do", num(pw), num(ph));
    let resources = format!("/XObject << /Im0 {image} 0 R >> ");
    Some(pdf.finish(pw, ph, &resources, content.as_bytes()))
}

/// A number as PDF writes it.
fn num(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// One tag of the SVG and the text after it, up to the next tag.
struct Tag<'a> {
    name: &'a str,
    attrs: &'a str,
    closing: bool,
    text: &'a str,
}

/// The tags of `svg`, in order; None when it is not well formed.
fn tags(svg: &str) -> Option<Vec<Tag<'_>>> {
    let mut out = Vec::new();
    let mut rest = svg.trim();
    while !rest.is_empty() {
        let (tag, after) = rest.strip_prefix('<')?.split_once('>')?;
        let next = after.find('<').unwrap_or(after.len());
        let (text, following) = after.split_at(next);
        let (tag, closing) = match tag.strip_prefix('/') {
            Some(t) => (t, true),
            None => (tag.strip_suffix('/').unwrap_or(tag), false),
        };
        let (name, attrs) = tag.split_once(' ').unwrap_or((tag, ""));
        out.push(Tag {
            name,
            attrs,
            closing,
            text,
        });
        rest = following;
    }
    Some(out)
}

impl<'a> Tag<'a> {
    fn attr(&self, name: &str) -> Option<&'a str> {
        let mut rest = self.attrs;
        while let Some((key, after)) = rest.split_once("=\"") {
            let (value, next) = after.split_once('"')?;
            if key.trim() == name {
                return Some(value);
            }
            rest = next;
        }
        None
    }

    fn number(&self, name: &str) -> Option<f64> {
        self.attr(name)?
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
    }
}

/// `#rrggbb` as PDF colour components.
fn color(value: &str) -> Option<String> {
    let hex = value.strip_prefix('#').filter(|h| h.len() == 6)?;
    let mut out = String::new();
    for i in [0, 2, 4] {
        let part = u8::from_str_radix(hex.get(i..i + 2)?, 16).ok()?;
        out.push_str(&num(f64::from(part) / 255.0));
        out.push(' ');
    }
    Some(out)
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

/// The content of the page under construction.
struct Page<'f> {
    face: &'f ttf_parser::Face<'static>,
    content: String,
    /// Opacities used, each an ExtGState.
    opacities: Vec<String>,
    /// Glyphs used and the character each shows.
    glyphs: BTreeMap<u16, char>,
}

impl Page<'_> {
    /// Read the document and return its size.
    fn read(&mut self, svg: &str) -> Option<(f64, f64)> {
        let mut size = None;
        // How a shape of the open group is painted: fill or stroke.
        let mut paint: Option<&str> = None;
        for tag in tags(svg)? {
            match (tag.name, tag.closing) {
                ("svg", false) => {
                    let (w, h) = (tag.number("width")?, tag.number("height")?);
                    size = (w > 0.0 && h > 0.0).then_some((w, h));
                }
                ("g", false) => {
                    self.content.push_str("q ");
                    paint = Some(self.group(&tag)?);
                }
                ("g", true) => {
                    self.content.push_str("Q\n");
                    paint = None;
                }
                ("text", false) => {
                    paint?;
                    self.text(&tag)?;
                }
                ("rect" | "circle" | "path", false) => {
                    let op = paint?;
                    self.shape(&tag)?;
                    self.content.push_str(op);
                    self.content.push('\n');
                }
                ("svg" | "text", true) => {}
                _ => return None,
            }
        }
        size
    }

    /// Set the group's colour, opacity and line width; the operator that paints its shapes.
    fn group(&mut self, tag: &Tag) -> Option<&'static str> {
        let fill = tag.attr("fill")?;
        let (op, opacity) = if fill == "none" {
            let rgb = color(tag.attr("stroke")?)?;
            let _ = write!(
                self.content,
                "{rgb}RG {} w ",
                num(tag.number("stroke-width")?)
            );
            ("S", tag.attr("stroke-opacity"))
        } else {
            let rgb = color(fill)?;
            let _ = write!(self.content, "{rgb}rg ");
            ("f", tag.attr("fill-opacity"))
        };
        if let Some(o) = opacity {
            let value = num(o.parse::<f64>().ok().filter(|v| (0.0..=1.0).contains(v))?);
            let i = match self.opacities.iter().position(|v| *v == value) {
                Some(i) => i,
                None => {
                    self.opacities.push(value);
                    self.opacities.len() - 1
                }
            };
            let _ = write!(self.content, "/G{i} gs ");
        }
        Some(op)
    }

    fn shape(&mut self, tag: &Tag) -> Option<()> {
        let c = &mut self.content;
        match tag.name {
            "rect" => {
                for name in ["x", "y", "width", "height"] {
                    let _ = write!(c, "{} ", num(tag.number(name)?));
                }
                c.push_str("re ");
            }
            "circle" => {
                let (x, y, r) = (tag.number("cx")?, tag.number("cy")?, tag.number("r")?);
                let k = r * ARC;
                let _ = write!(c, "{} {} m ", num(x + r), num(y));
                for [x1, y1, x2, y2, x3, y3] in [
                    [x + r, y + k, x + k, y + r, x, y + r],
                    [x - k, y + r, x - r, y + k, x - r, y],
                    [x - r, y - k, x - k, y - r, x, y - r],
                    [x + k, y - r, x + r, y - k, x + r, y],
                ] {
                    for v in [x1, y1, x2, y2, x3, y3] {
                        let _ = write!(c, "{} ", num(v));
                    }
                    c.push_str("c ");
                }
                c.push_str("h ");
            }
            _ => {
                let d = tag.attr("d")?;
                let (points, closed) = match d.strip_suffix('Z') {
                    Some(p) => (p, true),
                    None => (d, false),
                };
                for (i, point) in points.strip_prefix('M')?.split('L').enumerate() {
                    let (x, y) = point.split_once(' ')?;
                    let (x, y) = (x.parse::<i32>().ok()?, y.parse::<i32>().ok()?);
                    let _ = write!(c, "{x} {y} {} ", if i == 0 { 'm' } else { 'l' });
                }
                if closed {
                    c.push_str("h ");
                }
            }
        }
        Some(())
    }

    fn text(&mut self, tag: &Tag) -> Option<()> {
        let (x, y) = (tag.number("x")?, tag.number("y")?);
        let size = tag.number("font-size").filter(|s| *s > 0.0)?;
        let units = f64::from(self.face.units_per_em());
        let (mut hex, mut advance) = (String::new(), 0.0);
        for ch in unescape(tag.text).chars() {
            let glyph = self.face.glyph_index(ch)?;
            advance += f64::from(self.face.glyph_hor_advance(glyph)?) / units * size;
            self.glyphs.entry(glyph.0).or_insert(ch);
            let _ = write!(hex, "{:04X}", glyph.0);
        }
        if hex.is_empty() {
            return Some(());
        }
        let shift = match tag.attr("text-anchor")? {
            "start" => 0.0,
            "middle" => -advance / 2.0,
            "end" => -advance,
            _ => return None,
        };
        // The start of the baseline, turned about the anchor with the text.
        let (cos, sin, ax, ay) = match tag.attr("transform") {
            None => (1.0, 0.0, x, y),
            Some(t) => {
                let args: Vec<f64> = t
                    .strip_prefix("rotate(")?
                    .strip_suffix(')')?
                    .split(' ')
                    .map(|v| v.parse().ok())
                    .collect::<Option<_>>()?;
                let [angle, ax, ay] = args[..] else {
                    return None;
                };
                let (cos, sin) = match angle as i32 {
                    90 => (0.0, 1.0),
                    180 => (-1.0, 0.0),
                    270 => (0.0, -1.0),
                    _ => return None,
                };
                (cos, sin, ax, ay)
            }
        };
        let (dx, dy) = (x + shift - ax, y - ay);
        let (px, py) = (ax + cos * dx - sin * dy, ay + sin * dx + cos * dy);
        let _ = writeln!(
            self.content,
            "BT /F0 {} Tf {} {} {} {} {} {} Tm <{hex}> Tj ET",
            num(size),
            num(cos),
            num(sin),
            num(sin),
            num(-cos),
            num(px),
            num(py)
        );
        Some(())
    }
}

/// A PDF file under construction: its numbered objects. 1 is the catalog, 2 the page tree and
/// 3 the page, written last.
struct Pdf {
    objects: Vec<Vec<u8>>,
}

impl Pdf {
    fn new() -> Self {
        Self {
            objects: vec![Vec::new(); 3],
        }
    }

    fn object(&mut self, body: impl Into<Vec<u8>>) -> usize {
        self.objects.push(body.into());
        self.objects.len()
    }

    fn stream(&mut self, dict: &str, data: &[u8]) -> usize {
        let mut body = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
        body.extend_from_slice(data);
        body.extend_from_slice(b"\nendstream");
        self.object(body)
    }

    fn deflated(&mut self, dict: &str, data: &[u8]) -> usize {
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(data, 6);
        self.stream(&format!("{dict} /Filter /FlateDecode"), &packed)
    }

    /// The font, embedded, with the widths and characters of the glyphs used.
    fn font(&mut self, face: &ttf_parser::Face, glyphs: &BTreeMap<u16, char>) -> usize {
        let per_mille = |v: f64| num(v * 1000.0 / f64::from(face.units_per_em()));
        let file = self.deflated(&format!("/Length1 {}", FONT.len()), FONT);
        let b = face.global_bounding_box();
        let ascent = per_mille(f64::from(face.ascender()));
        let descriptor = self.object(format!(
            "<< /Type /FontDescriptor /FontName /PTOAAA+Arimo /Flags 32 /FontBBox [{} {} {} {}] \
             /ItalicAngle 0 /Ascent {ascent} /Descent {} /CapHeight 716 /StemV 80 \
             /FontFile2 {file} 0 R >>",
            per_mille(f64::from(b.x_min)),
            per_mille(f64::from(b.y_min)),
            per_mille(f64::from(b.x_max)),
            per_mille(f64::from(b.y_max)),
            per_mille(f64::from(face.descender())),
        ));
        let mut widths = String::new();
        let mut chars = String::new();
        for (&glyph, ch) in glyphs {
            let advance = face
                .glyph_hor_advance(ttf_parser::GlyphId(glyph))
                .unwrap_or_default();
            let _ = write!(widths, "{glyph} [{}] ", per_mille(f64::from(advance)));
            let mut units = [0u16; 2];
            let _ = write!(chars, "1 beginbfchar\n<{glyph:04X}> <");
            for unit in ch.encode_utf16(&mut units) {
                let _ = write!(chars, "{unit:04X}");
            }
            chars.push_str(">\nendbfchar\n");
        }
        let descendant = self.object(format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /PTOAAA+Arimo /CIDSystemInfo \
             << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor {descriptor} 0 R /CIDToGIDMap /Identity /W [{widths}] >>"
        ));
        let to_unicode = self.deflated(
            "",
            format!(
                "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
                 /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
                 /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
                 1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n{chars}\
                 endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
            )
            .as_bytes(),
        );
        self.object(format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /PTOAAA+Arimo /Encoding /Identity-H \
             /DescendantFonts [{descendant} 0 R] /ToUnicode {to_unicode} 0 R >>"
        ))
    }

    /// The file: one page of `content`.
    fn finish(mut self, width: f64, height: f64, resources: &str, content: &[u8]) -> Vec<u8> {
        let content = self.deflated("", content);
        self.objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
        self.objects[1] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
        self.objects[2] = format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources << {resources}>> \
             /Contents {content} 0 R >>",
            num(width),
            num(height)
        )
        .into_bytes();

        let mut out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
        let mut offsets = Vec::with_capacity(self.objects.len());
        for (i, body) in self.objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(body);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        let count = self.objects.len() + 1;
        let mut tail = format!("xref\n0 {count}\n0000000000 65535 f \n");
        for offset in offsets {
            let _ = writeln!(tail, "{offset:010} 00000 n ");
        }
        let _ = write!(
            tail,
            "trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        );
        out.extend_from_slice(tail.as_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    /// What p2a-core's recorder writes for a small chart.
    const SVG: &str = concat!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60" viewBox="0 0 80 60" font-family="Arimo, 'Liberation Sans', Arial, Helvetica, sans-serif" stroke-linejoin="round">"#,
        r##"<g fill="#ffffff"><rect x="0" y="0" width="79" height="59"/></g>"##,
        r##"<g fill="#ff4f00" fill-opacity="0.700"><circle cx="10" cy="20" r="4"/><circle cx="30" cy="20" r="4"/></g>"##,
        r##"<g fill="none" stroke="#000000" stroke-width="2"><path d="M1 2L-3 40L9 9"/></g>"##,
        r##"<g fill="#000000"><path d="M1 2L3 4L9 9Z"/><text x="40" y="12.50" font-size="12.00" text-anchor="middle">a &lt; b &amp; √β</text>"##,
        r##"<text x="5" y="30.00" font-size="12.00" text-anchor="end" transform="rotate(270 5 28)">y</text></g></svg>"##,
    );

    fn text_of(pdf: &[u8]) -> String {
        String::from_utf8_lossy(pdf).into_owned()
    }

    /// The streams of the file, inflated.
    fn streams(pdf: &[u8]) -> Vec<Vec<u8>> {
        let find = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).position(|w| w == needle);
        let (mut out, mut rest) = (Vec::new(), pdf);
        while let Some(at) = find(rest, b">>\nstream\n") {
            rest = &rest[at + 10..];
            let end = find(rest, b"\nendstream").unwrap();
            out.push(miniz_oxide::inflate::decompress_to_vec_zlib(&rest[..end]).unwrap());
            rest = &rest[end..];
        }
        out
    }

    #[test]
    fn a_chart_becomes_a_vector_page_with_text() {
        let pdf = from_svg(SVG).unwrap();
        assert!(pdf.starts_with(b"%PDF-1.4"));
        assert!(pdf.ends_with(b"%%EOF\n"));
        let text = text_of(&pdf);
        assert!(text.contains("/MediaBox [0 0 60 45]"));
        assert!(text.contains("/Encoding /Identity-H"));
        assert!(text.contains("/ca 0.7"));
        assert!(!text.contains("/Subtype /Image"));
        let streams = streams(&pdf);
        assert_eq!(streams[0], FONT);
        let content = String::from_utf8(streams.last().unwrap().clone()).unwrap();
        assert!(content.contains("1 2 m -3 40 l 9 9 l S"), "{content}");
        assert!(content.contains("1 2 m 3 4 l 9 9 l h f"), "{content}");
        assert_eq!(content.matches("/G0 gs").count(), 1);
        assert_eq!(content.matches("Tj").count(), 2);
        let cmap = String::from_utf8(streams[1].clone()).unwrap();
        for c in ["<003C>", "<0026>", "<221A>", "<03B2>"] {
            assert!(cmap.contains(c), "{c} in {cmap}");
        }
    }

    /// Every cross-reference entry points at its object.
    #[test]
    fn the_cross_reference_table_is_right() {
        let pdf = from_svg(SVG).unwrap();
        let text = text_of(&pdf);
        let start: usize = text
            .rsplit("startxref\n")
            .next()
            .and_then(|t| t.lines().next())
            .unwrap()
            .parse()
            .unwrap();
        assert!(pdf[start..].starts_with(b"xref\n"));
        let table = String::from_utf8(pdf[start..].to_vec()).unwrap();
        for (i, line) in table.lines().skip(3).enumerate() {
            if line.starts_with("trailer") {
                break;
            }
            let at: usize = line[..10].parse().unwrap();
            assert!(pdf[at..].starts_with(format!("{} 0 obj", i + 1).as_bytes()));
        }
    }

    #[test]
    fn what_it_cannot_show_is_refused() {
        for bad in [
            "",
            "not svg",
            "<svg width=\"0\" height=\"5\"></svg>",
            "<svg width=\"9\" height=\"5\"><script>1</script></svg>",
            "<svg width=\"9\" height=\"5\"><rect x=\"1\" y=\"1\" width=\"1\" height=\"1\"/></svg>",
            "<svg width=\"9\" height=\"5\"><g fill=\"red\"></g></svg>",
            "<svg width=\"9\" height=\"5\"><g fill=\"#000000\"><path d=\"M1 NaN\"/></g></svg>",
            "<svg width=\"9\" height=\"5\"><g fill=\"#ééé\"></g></svg>",
        ] {
            assert!(from_svg(bad).is_none(), "{bad}");
        }
        // A character the font has no glyph for.
        assert!(from_svg(&SVG.replace('√', "漢")).is_none());
    }

    #[test]
    fn a_png_is_placed_unchanged() {
        // 1 x 1, 8-bit RGB.
        let png = STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
            .unwrap();
        let pdf = from_png(&png).unwrap();
        let text = text_of(&pdf);
        assert!(text.contains("/Subtype /Image /Width 1 /Height 1"));
        assert_eq!(streams(&pdf)[0], [0, 255, 0, 0]);
        for bad in [&png[..20], b"GIF89a".as_slice(), &[]] {
            assert!(from_png(bad).is_none());
        }
    }
}
