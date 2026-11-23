//! The p2a logo as Dioxus components.
//!
//! The logo is drawn on a grid of square cells: the letters p, 2 and a, each
//! built from half-block cells, with the 2 in orange. Drawing it as SVG
//! rectangles keeps it sharp at any size and needs no font.
//!
//! - [`P2aWordmark`]: the three letters, for the header and the welcome screen.
//!   The letters take the current text colour, so they follow light and dark mode.
//! - [`P2aIconMinimal`]: the orange tile with a black 2, as in the favicon and
//!   the app icons (`assets/icons/`), for small places such as sidebar headers.
//!
//! The SVG and PNG files under `assets/icons/` and `docs/assets/` are drawn from
//! the same cell paths.

use dioxus::prelude::*;

/// Brand colours.
pub mod colors {
    /// International orange, the colour of the 2.
    pub const ORANGE: &str = "#FF4F00";
}

/// Cell paths of the logo on its 14 x 6 grid: the black p and a.
const LETTERS_PATH: &str = "M0 0h3v1h-3zM11 0h3v1h-3zM0 1h1v1h-1zM3 1h1v1h-1zM10 1h1v1h-1zM13 1h1v1h-1zM0 2h1v1h-1zM3 2h1v1h-1zM10 2h1v1h-1zM13 2h1v1h-1zM0 3h3v1h-3zM10 3h4v1h-4zM0 4h1v1h-1zM10 4h1v1h-1zM13 4h1v1h-1zM0 5h1v1h-1z";
/// Cell paths of the orange 2 on the same grid.
const TWO_PATH: &str = "M5 0h4v1h-4zM8 1h1v1h-1zM5 2h4v1h-4zM5 3h1v1h-1zM5 4h4v1h-4z";
/// The 2 alone, on its own 4 x 5 grid.
const TWO_ALONE_PATH: &str = "M0 0h4v1h-4zM3 1h1v1h-1zM0 2h4v1h-4zM0 3h1v1h-1zM0 4h4v1h-4z";

/// The p2a wordmark: p and a in the current text colour, the 2 in orange.
#[component]
pub fn P2aWordmark(
    /// Width in pixels; the height follows from the 14 x 6 grid.
    #[props(default = 140.0)]
    width: f64,
    /// Extra CSS classes.
    #[props(default = "")]
    class: &'static str,
) -> Element {
    let height = width * 6.0 / 14.0;
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            view_box: "0 0 14 6",
            width: "{width}",
            height: "{height}",
            shape_rendering: "crispEdges",
            role: "img",
            "aria-label": "p2a",
            class: "text-black dark:text-white {class}",
            path { fill: "currentColor", d: LETTERS_PATH }
            path { fill: colors::ORANGE, d: TWO_PATH }
        }
    }
}

/// The long logo: the wordmark, a rule, and "Prompt 2 / Analytics" in
/// Archivo extra-wide, as in the README header.
#[component]
pub fn P2aLockup(
    /// Cell size of the wordmark in pixels; the type scales with it.
    #[props(default = 10.0)]
    cell: f64,
) -> Element {
    let font = cell * 4.6;
    let gap = cell * 2.2;
    rsx! {
        div {
            class: "inline-flex items-center text-black dark:text-white",
            style: "gap: {gap}px;",
            role: "img",
            "aria-label": "p2a, Prompt 2 Analytics",
            P2aWordmark { width: cell * 14.0, class: "shrink-0" }
            span { class: "self-stretch w-[2px] bg-current", "aria-hidden": "true" }
            span {
                class: "font-extrabold leading-[0.95] tracking-[-0.015em]",
                style: "font-size: {font}px; font-stretch: 125%;",
                "aria-hidden": "true",
                "Prompt "
                span { style: "color: {colors::ORANGE};", "2" }
                br {}
                "Analytics"
            }
        }
    }
}

/// The orange tile with a black 2, as in the favicon.
#[component]
pub fn P2aIconMinimal(
    /// Size in pixels (width = height).
    #[props(default = 32.0)]
    size: f64,
    /// Extra CSS classes.
    #[props(default = "")]
    class: &'static str,
) -> Element {
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            view_box: "0 0 8 8",
            width: "{size}",
            height: "{size}",
            shape_rendering: "crispEdges",
            role: "img",
            "aria-label": "p2a",
            class: "{class}",
            rect { width: "8", height: "8", fill: colors::ORANGE }
            path { fill: "#000", transform: "translate(2 1.5)", d: TWO_ALONE_PATH }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cells covered by a path made of `Mx yhwv1h-wz` runs.
    fn cells(path: &str) -> Vec<(u32, u32)> {
        path.split('M')
            .filter(|s| !s.is_empty())
            .flat_map(|run| {
                let (xy, rest) = run.split_once('h').unwrap();
                let (x, y) = xy.split_once(' ').unwrap();
                let w: u32 = rest.split_once('v').unwrap().0.parse().unwrap();
                let (x, y): (u32, u32) = (x.parse().unwrap(), y.parse().unwrap());
                (x..x + w).map(move |c| (c, y))
            })
            .collect()
    }

    #[test]
    fn the_two_alone_is_the_wordmarks_two_moved_to_the_origin() {
        let mut moved: Vec<_> = cells(TWO_PATH)
            .into_iter()
            .map(|(x, y)| (x - 5, y))
            .collect();
        let mut alone = cells(TWO_ALONE_PATH);
        moved.sort_unstable();
        alone.sort_unstable();
        assert_eq!(moved, alone);
    }

    #[test]
    fn letters_and_two_stay_on_the_grid_and_do_not_overlap() {
        let letters = cells(LETTERS_PATH);
        let two = cells(TWO_PATH);
        assert!(letters.iter().chain(&two).all(|&(x, y)| x < 14 && y < 6));
        assert!(two.iter().all(|c| !letters.contains(c)));
        assert!(two.iter().all(|&(x, _)| (5..=8).contains(&x)));
    }
}
