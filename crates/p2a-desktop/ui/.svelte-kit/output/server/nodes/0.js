

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export const universal = {
  "ssr": false,
  "prerender": true
};
export const universal_id = "src/routes/+layout.ts";
export const imports = ["_app/immutable/nodes/0.Bykn2OTQ.js","_app/immutable/chunks/eaGfzuTu.js","_app/immutable/chunks/D0JSkPKu.js","_app/immutable/chunks/DhWLDm4_.js"];
export const stylesheets = ["_app/immutable/assets/0.DE_h7Ey9.css"];
export const fonts = [];
