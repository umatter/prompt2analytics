# Fonts

`../fonts.css` embeds two variable fonts as data URLs, so the web and desktop
builds load them without a font host (the web build's CSP allows `font-src data:`):

- **Archivo** (wdth 75-125, wght 400-800): interface text. Copyright The Archivo
  Project Authors, SIL Open Font License 1.1 (`OFL-Archivo.txt`).
- **Martian Mono** (wdth 75-112.5, wght 100-800): code, tool calls, data. Copyright
  The Martian Mono Project Authors, SIL Open Font License 1.1 (`OFL-MartianMono.txt`).

Both come from the google/fonts repository, subset with fontTools to Latin
(U+0020-017F), punctuation, arrows, box drawing, block elements and braille.
`archivo.woff` and `martian-mono.woff` are the subset files `fonts.css` is built from.

`arimo-regular.ttf` is the text font of figures exported as PDF (`src/figure_pdf.rs` embeds it
in the file):

- **Arimo** Regular. Copyright 2010 Google Inc., SIL Open Font License 1.1
  (`OFL-Arimo.txt`). Version 1.33, subset with fontTools, without hinting, to Latin, Greek,
  basic Cyrillic, sub- and superscripts, punctuation, arrows and common mathematical signs.
