<script lang="ts">
	import { renderMarkdown } from '$lib/utils/markdown';
	import type { ToolCall, LlmToolResult } from '$lib/types/llm';

	interface Props {
		content: string;
		toolCalls?: ToolCall[];
		toolResults?: LlmToolResult[];
		images?: string[];
		isStreaming?: boolean;
	}

	let { content, toolCalls, toolResults, images, isStreaming = false }: Props = $props();

	// Render markdown content
	let renderedContent = $derived(renderMarkdown(content));
</script>

<div class="message-content">
	{#if content}
		<div class="markdown-content">
			{@html renderedContent}
		</div>
	{/if}

	{#if isStreaming && !content}
		<span class="cursor">|</span>
	{/if}

	{#if toolCalls && toolCalls.length > 0}
		<div class="tool-calls">
			<div class="tool-calls-header">Tool Calls</div>
			{#each toolCalls as tc}
				<div class="tool-call">
					<span class="tool-name">{tc.name}</span>
					<pre class="tool-args"><code>{JSON.stringify(tc.arguments, null, 2)}</code></pre>
				</div>
			{/each}
		</div>
	{/if}

	{#if toolResults && toolResults.length > 0}
		<div class="tool-results">
			{#each toolResults as tr}
				<div class="tool-result" class:error={tr.is_error}>
					<pre><code>{tr.content}</code></pre>
				</div>
			{/each}
		</div>
	{/if}

	{#if images && images.length > 0}
		<div class="images">
			{#each images as img}
				<img src="data:image/png;base64,{img}" alt="Chart output" class="message-image" />
			{/each}
		</div>
	{/if}
</div>

<style>
	.message-content {
		line-height: 1.6;
	}

	/* Markdown content styles */
	.markdown-content :global(h1),
	.markdown-content :global(h2),
	.markdown-content :global(h3),
	.markdown-content :global(h4),
	.markdown-content :global(h5),
	.markdown-content :global(h6) {
		margin-top: var(--spacing-md);
		margin-bottom: var(--spacing-sm);
		font-weight: 600;
	}

	.markdown-content :global(h1) {
		font-size: 1.4rem;
	}
	.markdown-content :global(h2) {
		font-size: 1.25rem;
	}
	.markdown-content :global(h3) {
		font-size: 1.1rem;
	}

	.markdown-content :global(p) {
		margin-bottom: var(--spacing-sm);
	}

	.markdown-content :global(p:last-child) {
		margin-bottom: 0;
	}

	.markdown-content :global(ul),
	.markdown-content :global(ol) {
		margin-left: var(--spacing-lg);
		margin-bottom: var(--spacing-sm);
	}

	.markdown-content :global(li) {
		margin-bottom: var(--spacing-xs);
	}

	.markdown-content :global(pre) {
		margin: var(--spacing-sm) 0;
		padding: var(--spacing-md);
		background-color: var(--color-bg);
		border-radius: var(--radius-md);
		overflow-x: auto;
	}

	.markdown-content :global(code) {
		font-family: var(--font-mono);
		font-size: 0.9em;
	}

	.markdown-content :global(.inline-code) {
		background-color: var(--color-bg);
		padding: 0.1em 0.4em;
		border-radius: var(--radius-sm);
	}

	.markdown-content :global(blockquote) {
		margin: var(--spacing-sm) 0;
		padding-left: var(--spacing-md);
		border-left: 3px solid var(--color-primary);
		color: var(--color-text-secondary);
	}

	.markdown-content :global(table) {
		width: 100%;
		margin: var(--spacing-sm) 0;
		border-collapse: collapse;
	}

	.markdown-content :global(th),
	.markdown-content :global(td) {
		padding: var(--spacing-xs) var(--spacing-sm);
		border: 1px solid var(--color-border);
		text-align: left;
	}

	.markdown-content :global(th) {
		background-color: var(--color-bg-secondary);
	}

	.markdown-content :global(a) {
		color: var(--color-primary);
		text-decoration: none;
	}

	.markdown-content :global(a:hover) {
		text-decoration: underline;
	}

	.markdown-content :global(hr) {
		margin: var(--spacing-md) 0;
		border: none;
		border-top: 1px solid var(--color-border);
	}

	/* Syntax highlighting theme (GitHub Dark style) */
	.markdown-content :global(.hljs) {
		color: #c9d1d9;
		background: var(--color-bg);
	}

	.markdown-content :global(.hljs-comment),
	.markdown-content :global(.hljs-quote) {
		color: #8b949e;
		font-style: italic;
	}

	.markdown-content :global(.hljs-keyword),
	.markdown-content :global(.hljs-selector-tag) {
		color: #ff7b72;
	}

	.markdown-content :global(.hljs-string),
	.markdown-content :global(.hljs-addition) {
		color: #a5d6ff;
	}

	.markdown-content :global(.hljs-number),
	.markdown-content :global(.hljs-literal) {
		color: #79c0ff;
	}

	.markdown-content :global(.hljs-built_in),
	.markdown-content :global(.hljs-type) {
		color: #ffa657;
	}

	.markdown-content :global(.hljs-function),
	.markdown-content :global(.hljs-title) {
		color: #d2a8ff;
	}

	.markdown-content :global(.hljs-variable),
	.markdown-content :global(.hljs-attr) {
		color: #79c0ff;
	}

	.markdown-content :global(.hljs-deletion) {
		color: #ffa198;
	}

	/* Streaming cursor */
	.cursor {
		display: inline-block;
		animation: blink 1s step-end infinite;
		color: var(--color-primary);
	}

	@keyframes blink {
		50% {
			opacity: 0;
		}
	}

	/* Tool calls display */
	.tool-calls {
		margin-top: var(--spacing-md);
		padding: var(--spacing-sm);
		background-color: var(--color-bg);
		border-radius: var(--radius-md);
		border-left: 3px solid var(--color-warning);
	}

	.tool-calls-header {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--color-warning);
		margin-bottom: var(--spacing-sm);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.tool-call {
		margin-bottom: var(--spacing-sm);
	}

	.tool-call:last-child {
		margin-bottom: 0;
	}

	.tool-name {
		display: inline-block;
		font-weight: 600;
		color: var(--color-primary);
		background-color: var(--color-bg-secondary);
		padding: var(--spacing-xs) var(--spacing-sm);
		border-radius: var(--radius-sm);
		font-family: var(--font-mono);
		font-size: 0.85rem;
		margin-bottom: var(--spacing-xs);
	}

	.tool-args {
		margin: var(--spacing-xs) 0 0 0;
		padding: var(--spacing-sm);
		background-color: var(--color-bg-secondary);
		border-radius: var(--radius-sm);
		font-size: 0.8rem;
		overflow-x: auto;
	}

	.tool-args code {
		color: var(--color-text-secondary);
	}

	/* Tool results */
	.tool-results {
		margin-top: var(--spacing-sm);
	}

	.tool-result {
		padding: var(--spacing-sm);
		background-color: var(--color-bg);
		border-radius: var(--radius-sm);
		border-left: 3px solid var(--color-success);
		margin-bottom: var(--spacing-xs);
	}

	.tool-result.error {
		border-left-color: var(--color-error);
	}

	.tool-result pre {
		margin: 0;
		padding: 0;
		background: transparent;
		font-size: 0.85rem;
		white-space: pre-wrap;
		word-break: break-word;
	}

	/* Images */
	.images {
		margin-top: var(--spacing-md);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.message-image {
		max-width: 100%;
		border-radius: var(--radius-md);
		border: 1px solid var(--color-border);
	}
</style>
