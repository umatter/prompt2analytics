// Markdown rendering with syntax highlighting

import { marked } from 'marked';
import hljs from 'highlight.js';

// Configure marked with syntax highlighting
marked.setOptions({
	gfm: true,
	breaks: true
});

// Custom renderer for code blocks with syntax highlighting
const renderer = new marked.Renderer();

renderer.code = function ({ text, lang }: { text: string; lang?: string }) {
	const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext';
	const highlighted = hljs.highlight(text, { language }).value;
	return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`;
};

renderer.codespan = function ({ text }: { text: string }) {
	return `<code class="inline-code">${text}</code>`;
};

// Parse markdown to HTML
export function renderMarkdown(content: string): string {
	if (!content) return '';

	try {
		return marked.parse(content, { renderer }) as string;
	} catch (e) {
		console.error('Markdown parsing error:', e);
		return escapeHtml(content);
	}
}

// Escape HTML for safe display
function escapeHtml(text: string): string {
	const div = document.createElement('div');
	div.textContent = text;
	return div.innerHTML;
}

// Check if content likely contains markdown
export function hasMarkdown(content: string): boolean {
	// Check for common markdown patterns
	const patterns = [
		/^#{1,6}\s/m, // Headers
		/\*\*[^*]+\*\*/, // Bold
		/\*[^*]+\*/, // Italic
		/`[^`]+`/, // Inline code
		/```[\s\S]*?```/, // Code blocks
		/^\s*[-*+]\s/m, // Unordered lists
		/^\s*\d+\.\s/m, // Ordered lists
		/\[.+\]\(.+\)/, // Links
		/^\s*>/m // Blockquotes
	];

	return patterns.some((pattern) => pattern.test(content));
}
