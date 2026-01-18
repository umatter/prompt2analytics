//! Markdown rendering utilities
//!
//! Converts markdown text to Dioxus RSX elements using pulldown-cmark

use dioxus::prelude::*;
use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

/// Render markdown text to RSX elements with proper styling
pub fn render_markdown(text: &str) -> Element {
    let options = Options::all();
    let parser = Parser::new_ext(text, options);
    let events: Vec<Event> = parser.collect();

    let elements = render_events(&events);

    rsx! {
        div { class: "prose prose-sm max-w-none dark:prose-invert",
            {elements.into_iter()}
        }
    }
}

/// Recursively render markdown events into elements
fn render_events(events: &[Event]) -> Vec<Element> {
    let mut elements: Vec<Element> = Vec::new();
    let mut idx = 0;

    while idx < events.len() {
        match &events[idx] {
            Event::Start(tag) => {
                let (element, consumed) = render_tag(tag, &events[idx..]);
                elements.push(element);
                idx += consumed;
            }
            Event::Text(text) => {
                let t = text.to_string();
                elements.push(rsx! { "{t}" });
                idx += 1;
            }
            Event::Code(code) => {
                let c = code.to_string();
                elements.push(rsx! {
                    code { class: "px-1.5 py-0.5 text-sm bg-gray-100 dark:bg-gray-700 text-pink-600 dark:text-pink-400 rounded font-mono",
                        "{c}"
                    }
                });
                idx += 1;
            }
            Event::SoftBreak => {
                elements.push(rsx! { " " });
                idx += 1;
            }
            Event::HardBreak => {
                elements.push(rsx! { br {} });
                idx += 1;
            }
            Event::Rule => {
                elements.push(rsx! {
                    hr { class: "my-4 border-gray-300 dark:border-gray-600" }
                });
                idx += 1;
            }
            _ => {
                idx += 1;
            }
        }
    }

    elements
}

/// Render a tag and its contents, returning the element and number of events consumed
fn render_tag(tag: &Tag, events: &[Event]) -> (Element, usize) {
    // Find the matching end tag
    let end_idx = find_end_tag(events);
    let inner_events = &events[1..end_idx];

    match tag {
        Tag::Paragraph => {
            let children = render_events(inner_events);
            (
                rsx! {
                    p { class: "my-3 leading-relaxed text-gray-800 dark:text-gray-200",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Heading { level, .. } => {
            let children = render_events(inner_events);
            let element = match level {
                pulldown_cmark::HeadingLevel::H1 => rsx! {
                    h1 { class: "text-2xl font-bold mt-6 mb-3 text-gray-900 dark:text-white",
                        {children.into_iter()}
                    }
                },
                pulldown_cmark::HeadingLevel::H2 => rsx! {
                    h2 { class: "text-xl font-semibold mt-5 mb-2.5 text-gray-900 dark:text-white",
                        {children.into_iter()}
                    }
                },
                pulldown_cmark::HeadingLevel::H3 => rsx! {
                    h3 { class: "text-lg font-semibold mt-4 mb-2 text-gray-800 dark:text-gray-100",
                        {children.into_iter()}
                    }
                },
                pulldown_cmark::HeadingLevel::H4 => rsx! {
                    h4 { class: "text-base font-semibold mt-3 mb-1.5 text-gray-800 dark:text-gray-100",
                        {children.into_iter()}
                    }
                },
                _ => rsx! {
                    h5 { class: "text-sm font-semibold mt-2 mb-1 text-gray-700 dark:text-gray-200",
                        {children.into_iter()}
                    }
                },
            };
            (element, end_idx + 1)
        }
        Tag::BlockQuote(_) => {
            let children = render_events(inner_events);
            (
                rsx! {
                    blockquote { class: "pl-4 my-4 border-l-4 border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 italic",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::CodeBlock(kind) => {
            let lang = match kind {
                CodeBlockKind::Fenced(lang) => lang.to_string(),
                CodeBlockKind::Indented => String::new(),
            };
            let code = extract_text(inner_events);
            let lang_class = if !lang.is_empty() {
                format!("language-{}", lang)
            } else {
                String::new()
            };
            (
                rsx! {
                    pre { class: "my-4 p-4 bg-gray-900 rounded-lg overflow-x-auto",
                        code { class: "text-sm text-gray-100 font-mono {lang_class}",
                            "{code}"
                        }
                    }
                },
                end_idx + 1,
            )
        }
        Tag::List(None) => {
            let children = render_list_items(inner_events, false);
            (
                rsx! {
                    ul { class: "my-3 ml-4 space-y-1 list-disc list-outside text-gray-800 dark:text-gray-200",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::List(Some(start)) => {
            let children = render_list_items(inner_events, true);
            let start_num = *start as i64;
            (
                rsx! {
                    ol { class: "my-3 ml-4 space-y-1 list-decimal list-outside text-gray-800 dark:text-gray-200",
                        start: start_num,
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Item => {
            let children = render_events(inner_events);
            (
                rsx! {
                    li { class: "ml-2",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Strong => {
            let children = render_events(inner_events);
            (
                rsx! {
                    strong { class: "font-semibold text-gray-900 dark:text-white",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Emphasis => {
            let children = render_events(inner_events);
            (
                rsx! {
                    em { class: "italic",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Strikethrough => {
            let children = render_events(inner_events);
            (
                rsx! {
                    s { class: "line-through text-gray-500",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Link { dest_url, title, .. } => {
            let children = render_events(inner_events);
            let href = dest_url.to_string();
            let title_attr = title.to_string();
            (
                rsx! {
                    a {
                        class: "text-blue-600 dark:text-blue-400 hover:underline",
                        href: "{href}",
                        title: "{title_attr}",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Image { dest_url, title, .. } => {
            let src = dest_url.to_string();
            let alt = extract_text(inner_events);
            let title_attr = title.to_string();
            (
                rsx! {
                    img {
                        class: "max-w-full h-auto rounded-lg my-4",
                        src: "{src}",
                        alt: "{alt}",
                        title: "{title_attr}"
                    }
                },
                end_idx + 1,
            )
        }
        Tag::Table(_) => {
            let children = render_table_content(inner_events);
            (
                rsx! {
                    div { class: "overflow-x-auto my-4",
                        table { class: "min-w-full border-collapse border border-gray-300 dark:border-gray-600",
                            {children.into_iter()}
                        }
                    }
                },
                end_idx + 1,
            )
        }
        Tag::TableHead => {
            let children = render_events(inner_events);
            (
                rsx! {
                    thead { class: "bg-gray-100 dark:bg-gray-700",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::TableRow => {
            let children = render_events(inner_events);
            (
                rsx! {
                    tr { class: "border-b border-gray-300 dark:border-gray-600",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        Tag::TableCell => {
            let children = render_events(inner_events);
            (
                rsx! {
                    td { class: "px-4 py-2 border border-gray-300 dark:border-gray-600",
                        {children.into_iter()}
                    }
                },
                end_idx + 1,
            )
        }
        _ => {
            let children = render_events(inner_events);
            (
                rsx! {
                    span { {children.into_iter()} }
                },
                end_idx + 1,
            )
        }
    }
}

/// Find the index of the matching end tag
fn find_end_tag(events: &[Event]) -> usize {
    let mut depth = 0;
    for (i, event) in events.iter().enumerate() {
        match event {
            Event::Start(_) => depth += 1,
            Event::End(_) => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    events.len() - 1
}

/// Extract plain text from events
fn extract_text(events: &[Event]) -> String {
    let mut text = String::new();
    for event in events {
        match event {
            Event::Text(t) => text.push_str(t),
            Event::Code(c) => text.push_str(c),
            Event::SoftBreak | Event::HardBreak => text.push('\n'),
            _ => {}
        }
    }
    text
}

/// Render list items from events
fn render_list_items(events: &[Event], _ordered: bool) -> Vec<Element> {
    let mut items: Vec<Element> = Vec::new();
    let mut idx = 0;

    while idx < events.len() {
        match &events[idx] {
            Event::Start(Tag::Item) => {
                let end_idx = find_item_end(&events[idx..]);
                let item_events = &events[idx + 1..idx + end_idx];
                let children = render_events(item_events);
                items.push(rsx! {
                    li { class: "ml-2",
                        {children.into_iter()}
                    }
                });
                idx += end_idx + 1;
            }
            _ => {
                idx += 1;
            }
        }
    }

    items
}

/// Find the end of a list item
fn find_item_end(events: &[Event]) -> usize {
    let mut depth = 0;
    for (i, event) in events.iter().enumerate() {
        match event {
            Event::Start(Tag::Item) => depth += 1,
            Event::End(TagEnd::Item) => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    events.len() - 1
}

/// Render table content
fn render_table_content(events: &[Event]) -> Vec<Element> {
    let mut elements: Vec<Element> = Vec::new();
    let mut idx = 0;

    while idx < events.len() {
        match &events[idx] {
            Event::Start(Tag::TableHead) => {
                let end_idx = find_table_section_end(&events[idx..], TagEnd::TableHead);
                let rows = render_table_rows(&events[idx + 1..idx + end_idx], true);
                elements.push(rsx! {
                    thead { class: "bg-gray-100 dark:bg-gray-700",
                        {rows.into_iter()}
                    }
                });
                idx += end_idx + 1;
            }
            Event::Start(Tag::TableRow) => {
                // Body rows
                let end_idx = find_table_section_end(&events[idx..], TagEnd::TableRow);
                let cells = render_table_cells(&events[idx + 1..idx + end_idx], false);
                elements.push(rsx! {
                    tr { class: "border-b border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-750",
                        {cells.into_iter()}
                    }
                });
                idx += end_idx + 1;
            }
            _ => {
                idx += 1;
            }
        }
    }

    elements
}

/// Render table rows
fn render_table_rows(events: &[Event], is_header: bool) -> Vec<Element> {
    let mut rows: Vec<Element> = Vec::new();
    let mut idx = 0;

    while idx < events.len() {
        match &events[idx] {
            Event::Start(Tag::TableRow) => {
                let end_idx = find_table_section_end(&events[idx..], TagEnd::TableRow);
                let cells = render_table_cells(&events[idx + 1..idx + end_idx], is_header);
                rows.push(rsx! {
                    tr { class: "border-b border-gray-300 dark:border-gray-600",
                        {cells.into_iter()}
                    }
                });
                idx += end_idx + 1;
            }
            _ => {
                idx += 1;
            }
        }
    }

    rows
}

/// Render table cells
fn render_table_cells(events: &[Event], is_header: bool) -> Vec<Element> {
    let mut cells: Vec<Element> = Vec::new();
    let mut idx = 0;

    while idx < events.len() {
        match &events[idx] {
            Event::Start(Tag::TableCell) => {
                let end_idx = find_table_section_end(&events[idx..], TagEnd::TableCell);
                let content = render_events(&events[idx + 1..idx + end_idx]);
                if is_header {
                    cells.push(rsx! {
                        th { class: "px-4 py-2 text-left font-semibold border border-gray-300 dark:border-gray-600",
                            {content.into_iter()}
                        }
                    });
                } else {
                    cells.push(rsx! {
                        td { class: "px-4 py-2 border border-gray-300 dark:border-gray-600",
                            {content.into_iter()}
                        }
                    });
                }
                idx += end_idx + 1;
            }
            _ => {
                idx += 1;
            }
        }
    }

    cells
}

/// Find end of table section
fn find_table_section_end(events: &[Event], end_tag: TagEnd) -> usize {
    for (i, event) in events.iter().enumerate() {
        if let Event::End(tag) = event {
            if std::mem::discriminant(tag) == std::mem::discriminant(&end_tag) {
                return i;
            }
        }
    }
    events.len() - 1
}
