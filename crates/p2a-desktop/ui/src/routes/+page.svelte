<script lang="ts">
	import { chatState } from '$lib/state/chat.svelte';
	import { datasetsState } from '$lib/state/datasets.svelte';
	import { resultsState } from '$lib/state/results.svelte';
	import { invokeTool, parseCommand, pickFile, loadDataset, listDatasets } from '$lib/api/tauri';
	import type { ToolResult } from '$lib/types';

	let inputElement: HTMLTextAreaElement;

	// Handle sending a message
	async function handleSend() {
		const input = chatState.input.trim();
		if (!input || chatState.isProcessing) return;

		chatState.addUserMessage(input);
		chatState.clearInput();
		chatState.setProcessing(true);

		try {
			const result = await processInput(input);
			chatState.addAssistantMessage(result.content, result.images.map((i) => i.base64));

			// Add to results panel if there's content
			if (result.content || result.images.length > 0) {
				const parsed = parseCommand(input);
				resultsState.addResult(
					parsed?.toolName || 'response',
					result.content,
					result.images.map((i) => i.base64)
				);
			}

			// Refresh datasets list if we loaded something
			if (input.startsWith('load_dataset')) {
				await refreshDatasets();
			}
		} catch (error) {
			chatState.addErrorMessage(`Error: ${error}`);
		} finally {
			chatState.setProcessing(false);
		}
	}

	// Process input - parse as command or show help
	async function processInput(input: string): Promise<ToolResult> {
		const parsed = parseCommand(input);

		if (parsed) {
			return await invokeTool(parsed.toolName, parsed.args);
		}

		// If not a command, show help
		return {
			success: true,
			content: `Enter commands in format: tool_name key=value key2=value2

Examples:
  load_dataset path=/path/to/file.csv
  describe_dataset dataset=mydata
  regression_ols dataset=mydata y=price x=["sqft","beds"]
  viz_histogram dataset=mydata column=price`,
			images: [],
			error: undefined
		};
	}

	// Handle file import button
	async function handleImport() {
		try {
			const path = await pickFile();
			if (path) {
				chatState.setInput(`load_dataset path=${path}`);
				await handleSend();
			}
		} catch (error) {
			chatState.addErrorMessage(`Failed to open file dialog: ${error}`);
		}
	}

	// Refresh datasets list
	async function refreshDatasets() {
		try {
			const datasets = await listDatasets();
			datasetsState.setDatasets(datasets);
		} catch (error) {
			console.error('Failed to refresh datasets:', error);
		}
	}

	// Handle keyboard shortcuts
	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	}

	// Format timestamp
	function formatTime(date: Date): string {
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}
</script>

<div class="layout">
	<!-- Chat Panel (Left) -->
	<div class="panel chat-panel">
		<div class="panel-header">
			<h2>Chat</h2>
			<button class="secondary" onclick={handleImport}>Import File</button>
		</div>
		<div class="panel-content messages">
			{#each chatState.messages as message (message.id)}
				<div class="message {message.role}">
					<div class="message-header">
						<span class="role">{message.role}</span>
						<span class="time">{formatTime(message.timestamp)}</span>
					</div>
					<div class="message-content">
						<pre>{message.content}</pre>
						{#if message.images}
							{#each message.images as img}
								<img src="data:image/png;base64,{img}" alt="Chart" class="message-image" />
							{/each}
						{/if}
					</div>
				</div>
			{/each}
		</div>
		<div class="input-area">
			<textarea
				bind:this={inputElement}
				bind:value={chatState.input}
				onkeydown={handleKeydown}
				placeholder="Enter command (e.g., load_dataset path=/data/file.csv)"
				disabled={chatState.isProcessing}
				rows="3"
			></textarea>
			<button onclick={handleSend} disabled={chatState.isProcessing || !chatState.input.trim()}>
				{chatState.isProcessing ? 'Processing...' : 'Send'}
			</button>
		</div>
	</div>

	<!-- Data Viewer Panel (Center) -->
	<div class="panel data-panel">
		<div class="panel-header">
			<h2>Data</h2>
			{#if datasetsState.datasets.length > 0}
				<select
					value={datasetsState.activeDataset}
					onchange={(e) => datasetsState.setActiveDataset(e.currentTarget.value)}
				>
					<option value="">Select dataset...</option>
					{#each datasetsState.datasets as ds}
						<option value={ds.name}>{ds.name} ({ds.rows} x {ds.columns})</option>
					{/each}
				</select>
			{/if}
		</div>
		<div class="panel-content">
			{#if datasetsState.datasets.length === 0}
				<div class="empty-state">
					<p>No datasets loaded</p>
					<p class="text-muted">Use "Import File" or the load_dataset command</p>
				</div>
			{:else if !datasetsState.activeDataset}
				<div class="empty-state">
					<p>Select a dataset to view</p>
				</div>
			{:else if datasetsState.preview}
				<div class="data-table-container">
					<table>
						<thead>
							<tr>
								{#each datasetsState.preview.columns as col}
									<th onclick={() => datasetsState.toggleSort(col)}>
										{col}
										{#if datasetsState.sortColumn === col}
											<span class="sort-indicator">
												{datasetsState.sortDirection === 'asc' ? '↑' : '↓'}
											</span>
										{/if}
									</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each datasetsState.preview.rows as row, i}
								<tr class:even={i % 2 === 0}>
									{#each datasetsState.preview.columns as col}
										<td>{row[col] ?? ''}</td>
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
	</div>

	<!-- Results Panel (Right) -->
	<div class="panel results-panel">
		<div class="panel-header">
			<h2>Results</h2>
			{#if resultsState.results.length > 0}
				<button class="secondary" onclick={() => resultsState.clearResults()}>Clear</button>
			{/if}
		</div>
		<div class="panel-content">
			{#if resultsState.results.length === 0}
				<div class="empty-state">
					<p>No results yet</p>
					<p class="text-muted">Run analyses to see results here</p>
				</div>
			{:else}
				{#each resultsState.results as result (result.id)}
					<div
						class="result-item"
						class:expanded={resultsState.expandedResult === result.id}
					>
						<button
							class="result-header"
							onclick={() => resultsState.toggleExpanded(result.id)}
						>
							<span class="tool-name">{result.tool}</span>
							<span class="time">{formatTime(result.timestamp)}</span>
						</button>
						{#if resultsState.expandedResult === result.id}
							<div class="result-content">
								<pre>{result.content}</pre>
								{#each result.images as img}
									<img src="data:image/png;base64,{img}" alt="{result.tool} output" />
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			{/if}
		</div>
	</div>
</div>

<style>
	.layout {
		display: grid;
		grid-template-columns: 1fr 1.5fr 1fr;
		gap: var(--spacing-md);
		height: 100%;
		padding: var(--spacing-md);
	}

	.panel {
		display: flex;
		flex-direction: column;
		min-height: 0;
	}

	.panel-content {
		flex: 1;
		overflow: auto;
	}

	/* Chat Panel */
	.messages {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-md);
	}

	.message {
		padding: var(--spacing-md);
		border-radius: var(--radius-md);
		background-color: var(--color-surface);
	}

	.message.user {
		background-color: var(--color-bg-tertiary);
	}

	.message.error {
		background-color: rgba(239, 68, 68, 0.1);
		border-left: 3px solid var(--color-error);
	}

	.message-header {
		display: flex;
		justify-content: space-between;
		margin-bottom: var(--spacing-sm);
		font-size: 0.8rem;
	}

	.message-header .role {
		font-weight: 600;
		text-transform: capitalize;
	}

	.message-header .time {
		color: var(--color-text-muted);
	}

	.message-content pre {
		white-space: pre-wrap;
		word-break: break-word;
		background: transparent;
		padding: 0;
	}

	.message-image {
		max-width: 100%;
		margin-top: var(--spacing-md);
		border-radius: var(--radius-md);
	}

	.input-area {
		padding: var(--spacing-md);
		border-top: 1px solid var(--color-border);
		display: flex;
		gap: var(--spacing-sm);
	}

	.input-area textarea {
		flex: 1;
		resize: none;
	}

	.input-area button {
		align-self: flex-end;
	}

	/* Data Panel */
	.data-table-container {
		overflow: auto;
	}

	.data-panel select {
		padding: var(--spacing-xs) var(--spacing-sm);
		border-radius: var(--radius-sm);
		background-color: var(--color-surface);
		color: var(--color-text);
		border: 1px solid var(--color-border);
	}

	th {
		cursor: pointer;
		user-select: none;
	}

	th:hover {
		background-color: var(--color-surface-hover);
	}

	.sort-indicator {
		margin-left: var(--spacing-xs);
	}

	/* Results Panel */
	.result-item {
		margin-bottom: var(--spacing-sm);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.result-header {
		width: 100%;
		display: flex;
		justify-content: space-between;
		padding: var(--spacing-sm) var(--spacing-md);
		background-color: var(--color-surface);
		border: none;
		border-radius: 0;
		text-align: left;
	}

	.result-header:hover {
		background-color: var(--color-surface-hover);
	}

	.tool-name {
		font-weight: 600;
		color: var(--color-primary);
	}

	.result-content {
		padding: var(--spacing-md);
		background-color: var(--color-bg);
		border-top: 1px solid var(--color-border);
	}

	.result-content pre {
		margin-bottom: var(--spacing-md);
		white-space: pre-wrap;
		word-break: break-word;
	}

	.result-content img {
		max-width: 100%;
		border-radius: var(--radius-md);
	}

	/* Empty states */
	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		text-align: center;
		color: var(--color-text-secondary);
	}

	.empty-state p {
		margin: var(--spacing-xs) 0;
	}

	/* Responsive */
	@media (max-width: 1200px) {
		.layout {
			grid-template-columns: 1fr 1fr;
			grid-template-rows: 1fr 1fr;
		}

		.chat-panel {
			grid-column: 1 / 2;
			grid-row: 1 / 3;
		}

		.data-panel {
			grid-column: 2 / 3;
			grid-row: 1 / 2;
		}

		.results-panel {
			grid-column: 2 / 3;
			grid-row: 2 / 3;
		}
	}

	@media (max-width: 800px) {
		.layout {
			grid-template-columns: 1fr;
			grid-template-rows: 1fr 200px 200px;
		}

		.chat-panel,
		.data-panel,
		.results-panel {
			grid-column: 1;
		}

		.chat-panel {
			grid-row: 1;
		}

		.data-panel {
			grid-row: 2;
		}

		.results-panel {
			grid-row: 3;
		}
	}
</style>
