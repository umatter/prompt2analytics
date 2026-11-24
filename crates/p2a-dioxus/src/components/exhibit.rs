//! Exhibits in the transcript and the viewer that enlarges them.
//!
//! A figure shows its caption with "expand" and "png"; a table in an answer
//! gets "expand", "csv", "latex" and "copy" under it. "Expand" (or a click
//! on a figure) opens the viewer: the exhibit over the whole window, zoomable
//! when it is a figure, with the same exports.

use dioxus::prelude::*;

use crate::exhibit::{
    Exhibit, ExhibitViewer, TableData, TableFormat, copy_table, copy_table_as, export_figure,
    export_table,
};

/// Zoom steps of the viewer, in percent of the viewer's width ("fit" shows
/// the whole figure as large as the viewer allows).
const ZOOM_STEPS: [u32; 7] = [50, 75, 100, 150, 200, 300, 400];

/// The next zoom step above (`up`) or below the current zoom (`None` = fit,
/// taken as 100).
fn step_zoom(current: Option<u32>, up: bool) -> Option<u32> {
    let z = current.unwrap_or(100);
    let next = if up {
        ZOOM_STEPS.iter().copied().find(|&s| s > z)
    } else {
        ZOOM_STEPS.iter().rev().copied().find(|&s| s < z)
    };
    Some(next.unwrap_or(z))
}

/// A plot in an answer: the image, a click to enlarge it, and its caption
/// with the actions.
#[component]
pub fn FigureExhibit(image: String, label: String) -> Element {
    let viewer = try_use_context::<ExhibitViewer>();
    let open = {
        let (image, label) = (image.clone(), label.clone());
        move |_| {
            if let Some(v) = viewer {
                let mut o = v.open;
                o.set(Some(Exhibit::Figure {
                    label: label.clone(),
                    png: image.clone(),
                }));
            }
        }
    };
    let open2 = open.clone();
    let save = {
        let (image, label) = (image.clone(), label.clone());
        move |_| {
            let note = export_figure(&image, &label);
            if let Some(v) = viewer {
                v.flash(note);
            }
        }
    };
    rsx! {
        figure { class: "k-plot k-exhibit",
            button {
                class: "k-zoomable",
                title: "Open {label} in the viewer",
                aria_label: "Open {label} in the viewer",
                onclick: open,
                img { src: "data:image/png;base64,{image}", alt: "{label}" }
            }
            figcaption { class: "k-exhibit-bar",
                span { "{label}" }
                span { class: "k-exhibit-acts",
                    button { class: "k-xbtn", onclick: open2, "expand" }
                    button { class: "k-xbtn", title: "Download as PNG", onclick: save, "png" }
                }
            }
        }
    }
}

/// The actions under a table in an answer. A plain function rather than a
/// component, because the Markdown renderer builds it.
pub fn table_toolbar(data: TableData, viewer: ExhibitViewer) -> Element {
    let label = "Table".to_string();
    let size = format!("{} × {}", data.rows.len(), data.width());
    let (d1, d2, d3, d4) = (data.clone(), data.clone(), data.clone(), data);
    let (l1, l2, l3) = (label.clone(), label.clone(), label.clone());
    rsx! {
        div { class: "k-exhibit-bar",
            span { "{label} " span { class: "k-mono", "{size}" } }
            span { class: "k-exhibit-acts",
                button {
                    class: "k-xbtn",
                    onclick: move |_| {
                        let mut o = viewer.open;
                        o.set(Some(Exhibit::Table { label: l1.clone(), data: d1.clone() }));
                    },
                    "expand"
                }
                button {
                    class: "k-xbtn",
                    title: "Download as CSV",
                    onclick: move |_| viewer.flash(export_table(&d2, &l2, TableFormat::Csv)),
                    "csv"
                }
                button {
                    class: "k-xbtn",
                    title: "Download as a LaTeX table (booktabs)",
                    onclick: move |_| viewer.flash(export_table(&d3, &l3, TableFormat::Latex)),
                    "latex"
                }
                button {
                    class: "k-xbtn",
                    title: "Copy, tab-separated, for a spreadsheet",
                    onclick: move |_| viewer.flash(copy_table(&d4)),
                    "copy"
                }
            }
        }
    }
}

/// The viewer: the open exhibit over the whole window. Mounted once in the
/// app; it also shows the last export notice.
#[component]
pub fn ExhibitViewerModal() -> Element {
    let viewer = use_context::<ExhibitViewer>();
    let mut open = viewer.open;
    let mut zoom = use_signal(|| None::<u32>);
    // How a table is shown: as a table (None) or as the source of a format.
    let mut view = use_signal(|| None::<TableFormat>);
    // A newly opened exhibit starts fitted to the window, a table as a table.
    use_effect(move || {
        let _ = open.read();
        zoom.set(None);
        view.set(None);
    });
    let is_open = open.read().is_some();
    crate::app::use_escape_close(is_open, Callback::new(move |_| open.set(None)));

    let note = viewer.note.read().clone();
    let toast = rsx! {
        if let Some(text) = note {
            div { class: "k-toast", role: "status", "{text}" }
        }
    };
    let Some(exhibit) = open.read().clone() else {
        return toast;
    };

    let (label, sub) = match &exhibit {
        Exhibit::Figure { label, .. } => (
            label.clone(),
            "Click the image to zoom to 200% and back".to_string(),
        ),
        Exhibit::Table { label, data } => (
            label.clone(),
            format!("{} rows, {} columns", data.rows.len(), data.width()),
        ),
    };

    let actions = match exhibit.clone() {
        Exhibit::Figure { label, png } => rsx! {
            div { class: "k-zoom", role: "group", aria_label: "Zoom",
                button {
                    class: "k-xbtn",
                    aria_label: "Zoom out",
                    onclick: move |_| zoom.set(step_zoom(zoom(), false)),
                    "−"
                }
                span { class: "k-mono",
                    match zoom() {
                        None => "fit".to_string(),
                        Some(z) => format!("{z}%"),
                    }
                }
                button {
                    class: "k-xbtn",
                    aria_label: "Zoom in",
                    onclick: move |_| zoom.set(step_zoom(zoom(), true)),
                    "+"
                }
                if zoom().is_some() {
                    button { class: "k-xbtn", onclick: move |_| zoom.set(None), "fit" }
                }
            }
            button {
                class: "k-btn-pri",
                onclick: move |_| viewer.flash(export_figure(&png, &label)),
                "Download PNG"
            }
        },
        Exhibit::Table { label, data } => {
            let d1 = data.clone();
            let current = view();
            let download = current.unwrap_or(TableFormat::Csv);
            rsx! {
                button {
                    class: "k-btn",
                    title: match current {
                        None => "Copy, tab-separated, for a spreadsheet".to_string(),
                        Some(f) => format!("Copy the {} source", f.name()),
                    },
                    onclick: move |_| {
                        viewer
                            .flash(
                                match view() {
                                    None => copy_table(&d1),
                                    Some(f) => copy_table_as(&d1, f),
                                },
                            )
                    },
                    "Copy"
                }
                button {
                    class: "k-btn-pri",
                    onclick: move |_| viewer.flash(export_table(&data, &label, download)),
                    "Download {download.name()}"
                }
            }
        }
    };

    // The table's views, as tabs under the header.
    let tabs = match &exhibit {
        Exhibit::Figure { .. } => rsx! {},
        Exhibit::Table { .. } => {
            let options = [
                (None, "Table"),
                (Some(TableFormat::Markdown), "Markdown"),
                (Some(TableFormat::Latex), "LaTeX"),
                (Some(TableFormat::Csv), "CSV"),
            ];
            rsx! {
                div { class: "k-tabs", role: "tablist", aria_label: "Show the table as",
                    for (format , name) in options {
                        button {
                            key: "{name}",
                            class: if view() == format { "k-tab on" } else { "k-tab" },
                            role: "tab",
                            aria_selected: view() == format,
                            onclick: move |_| view.set(format),
                            "{name}"
                        }
                    }
                }
            }
        }
    };

    let body = match exhibit {
        Exhibit::Figure { label, png } => {
            let style = match zoom() {
                None => "width: 100%; height: 100%; object-fit: contain;".to_string(),
                Some(z) => format!("width: {z}%; max-width: none; margin: 0 auto;"),
            };
            let fitted = zoom().is_none();
            rsx! {
                div { class: if fitted { "k-viewer-fig fit" } else { "k-viewer-fig" },
                    img {
                        src: "data:image/png;base64,{png}",
                        alt: "{label}",
                        style: "{style}",
                        class: if fitted { "cursor-zoom-in" } else { "cursor-zoom-out" },
                        onclick: move |_| zoom.set(if zoom().is_none() { Some(200) } else { None }),
                    }
                }
            }
        }
        Exhibit::Table { data, .. } if view().is_some() => {
            let source = data.render(view().unwrap_or(TableFormat::Csv));
            rsx! {
                pre { class: "k-source", "{source}" }
            }
        }
        Exhibit::Table { data, .. } => rsx! {
            table { class: "k-xtable",
                if !data.header.is_empty() {
                    thead {
                        tr {
                            for h in data.header.iter() {
                                th { "{h}" }
                            }
                        }
                    }
                }
                tbody {
                    for row in data.rows.iter() {
                        tr {
                            for c in row.iter() {
                                td { "{c}" }
                            }
                        }
                    }
                }
            }
        },
    };

    rsx! {
        div { class: "fixed inset-0 z-[70] flex p-3 md:p-6",
            div { class: "k-backdrop", onclick: move |_| open.set(None) }
            div {
                class: "k-modal k-viewer",
                role: "dialog",
                aria_modal: "true",
                aria_label: "{label}",
                div { class: "k-modal-head",
                    div { class: "min-w-0",
                        h2 { class: "k-modal-title", "{label}" }
                        p { class: "k-modal-sub", "{sub}" }
                    }
                    div { class: "flex flex-wrap items-center justify-end gap-2",
                        {actions}
                        button {
                            class: "k-icon-btn",
                            title: "Close (Esc)",
                            aria_label: "Close",
                            onclick: move |_| open.set(None),
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "square",
                                    stroke_width: "2",
                                    d: "M6 18L18 6M6 6l12 12"
                                }
                            }
                        }
                    }
                }
                {tabs}
                div { class: "k-modal-body k-viewer-body", {body} }
            }
        }
        {toast}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_steps_from_fit_and_stop_at_the_ends() {
        assert_eq!(step_zoom(None, true), Some(150));
        assert_eq!(step_zoom(None, false), Some(75));
        assert_eq!(step_zoom(Some(400), true), Some(400));
        assert_eq!(step_zoom(Some(50), false), Some(50));
        assert_eq!(step_zoom(Some(150), false), Some(100));
    }
}
