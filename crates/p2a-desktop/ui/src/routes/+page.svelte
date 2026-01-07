<script lang="ts">
	import { onMount, onDestroy, tick } from 'svelte';
	import { chatState } from '$lib/state/chat.svelte';
	import { datasetsState } from '$lib/state/datasets.svelte';
	import { resultsState } from '$lib/state/results.svelte';
	import { invokeTool, parseCommand, pickFile, loadDataset, listDatasets } from '$lib/api/tauri';
	import {
		sendMessage,
		sendMessageStream,
		listenToStream,
		listConversations,
		getConversation,
		deleteConversation,
		renameConversation,
		exportConversation
	} from '$lib/api/llm';
	import type { ToolResult } from '$lib/types';
	import type { UnlistenFn } from '@tauri-apps/api/event';
	import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';
	import MessageContent from '$lib/components/MessageContent.svelte';

	let inputElement: HTMLTextAreaElement;
	let messagesContainer: HTMLDivElement;
	let unlistenStream: UnlistenFn | null = null;

	// Conversation management state
	let conversationSearch = $state('');
	let renamingConversation = $state<string | null>(null);
	let renameValue = $state('');

	// Filter conversations by search term
	let filteredConversations = $derived(
		chatState.conversations.filter((conv) =>
			conv.title.toLowerCase().includes(conversationSearch.toLowerCase())
		)
	);

	// Auto-scroll to bottom when messages change
	$effect(() => {
		// Track messages array changes
		const _messages = chatState.messages;
		scrollToBottom();
	});

	async function scrollToBottom() {
		await tick();
		if (messagesContainer) {
			messagesContainer.scrollTop = messagesContainer.scrollHeight;
		}
	}

	onMount(async () => {
		// Load conversations on mount
		try {
			const conversations = await listConversations();
			chatState.setConversations(conversations);
		} catch (e) {
			console.error('Failed to load conversations:', e);
		}

		// Refresh datasets
		await refreshDatasets();
	});

	onDestroy(() => {
		// Cleanup stream listener
		if (unlistenStream) {
			unlistenStream();
		}
	});

	// Handle sending a message
	async function handleSend() {
		const input = chatState.input.trim();
		if (!input || chatState.isProcessing) return;

		chatState.addUserMessage(input);
		chatState.clearInput();
		chatState.setProcessing(true);

		try {
			if (chatState.isLlmMode) {
				await handleLlmMessage(input);
			} else {
				await handleCommandMessage(input);
			}
		} catch (error) {
			chatState.addErrorMessage(`Error: ${error}`);
		} finally {
			chatState.setProcessing(false);
		}
	}

	// Handle LLM chat mode
	async function handleLlmMessage(input: string) {
		// Set up stream listener
		unlistenStream = await listenToStream((chunk) => {
			chatState.handleStreamChunk(chunk);
		});

		// Start streaming message
		chatState.startStreamingMessage();

		try {
			const response = await sendMessageStream(input, chatState.currentConversationId ?? undefined);

			// Update conversation ID if this was a new conversation
			if (!chatState.currentConversationId) {
				chatState.setConversationId(response.conversation_id);
				// Refresh conversations list
				const conversations = await listConversations();
				chatState.setConversations(conversations);
			}

			// Finalize the streaming message with final content
			chatState.finalizeStreamingMessage(
				response.message.content,
				response.message.tool_calls ?? undefined
			);

			// Refresh datasets in case tools loaded new data
			await refreshDatasets();
		} finally {
			// Cleanup listener
			if (unlistenStream) {
				unlistenStream();
				unlistenStream = null;
			}
		}
	}

	// Handle command mode
	async function handleCommandMessage(input: string) {
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
				if (chatState.isLlmMode) {
					chatState.setInput(`Please load and describe the dataset at: ${path}`);
				} else {
					chatState.setInput(`load_dataset path=${path}`);
				}
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

	// Handle new conversation
	function handleNewConversation() {
		chatState.startNewConversation();
	}

	// Handle loading a conversation
	async function handleLoadConversation(id: string) {
		try {
			chatState.setConversationId(id);
			chatState.clearMessages();

			const messages = await getConversation(id);
			for (const msg of messages) {
				chatState.addMessage(
					msg.role === 'tool' ? 'system' : msg.role,
					msg.content,
					undefined,
					msg.tool_calls,
					msg.tool_results
				);
			}
		} catch (error) {
			chatState.addErrorMessage(`Failed to load conversation: ${error}`);
		}
	}

	// Handle deleting a conversation
	async function handleDeleteConversation(id: string) {
		try {
			await deleteConversation(id);
			chatState.removeConversation(id);
		} catch (error) {
			chatState.addErrorMessage(`Failed to delete conversation: ${error}`);
		}
	}

	// Start renaming a conversation
	function startRenameConversation(id: string, currentTitle: string) {
		renamingConversation = id;
		renameValue = currentTitle;
	}

	// Cancel renaming
	function cancelRename() {
		renamingConversation = null;
		renameValue = '';
	}

	// Submit rename
	async function submitRename() {
		if (!renamingConversation || !renameValue.trim()) return;
		try {
			await renameConversation(renamingConversation, renameValue.trim());
			chatState.updateConversationTitle(renamingConversation, renameValue.trim());
			cancelRename();
		} catch (error) {
			chatState.addErrorMessage(`Failed to rename conversation: ${error}`);
		}
	}

	// Handle rename keydown
	function handleRenameKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			submitRename();
		} else if (e.key === 'Escape') {
			cancelRename();
		}
	}

	// Export a conversation
	async function handleExportConversation(id: string) {
		try {
			const exported = await exportConversation(id);
			// Create downloadable JSON
			const blob = new Blob([JSON.stringify(exported, null, 2)], { type: 'application/json' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `${exported.title.replace(/[^a-z0-9]/gi, '_')}_${exported.id.slice(0, 8)}.json`;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);
		} catch (error) {
			chatState.addErrorMessage(`Failed to export conversation: ${error}`);
		}
	}
</script>

<div class="layout">
	<!-- Chat Panel (Left) -->
	<div class="panel chat-panel">
		<div class="panel-header">
			<div class="header-left">
				<h2>Chat</h2>
				<label class="mode-toggle">
					<input
						type="checkbox"
						checked={chatState.isLlmMode}
						onchange={() => chatState.toggleLlmMode()}
					/>
					<span class="toggle-label">{chatState.isLlmMode ? 'AI Mode' : 'Command Mode'}</span>
				</label>
			</div>
			<div class="header-actions">
				<button class="secondary" onclick={handleImport}>Import</button>
				<a href="/settings" class="settings-link" title="Settings">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="20"
						height="20"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
					>
						<circle cx="12" cy="12" r="3"></circle>
						<path
							d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
						></path>
					</svg>
				</a>
			</div>
		</div>

		<!-- Conversation Sidebar (LLM mode only) -->
		{#if chatState.isLlmMode && chatState.conversations.length > 0}
			<div class="conversations-sidebar">
				<div class="sidebar-header">
					<button class="new-chat-btn" onclick={handleNewConversation}> + New Chat </button>
					<input
						type="text"
						class="conversation-search"
						placeholder="Search..."
						bind:value={conversationSearch}
					/>
				</div>
				<div class="conversation-list">
					{#each filteredConversations.slice(0, 10) as conv}
						<div
							class="conversation-item"
							class:active={chatState.currentConversationId === conv.id}
						>
							{#if renamingConversation === conv.id}
								<input
									type="text"
									class="rename-input"
									bind:value={renameValue}
									onkeydown={handleRenameKeydown}
									onblur={cancelRename}
								/>
							{:else}
								<button
									class="conversation-btn"
									onclick={() => handleLoadConversation(conv.id)}
									title={conv.title}
								>
									{conv.title.slice(0, 25)}{conv.title.length > 25 ? '...' : ''}
								</button>
								<div class="conversation-actions">
									<button
										class="action-btn"
										onclick={() => startRenameConversation(conv.id, conv.title)}
										title="Rename"
									>
										<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
											<path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
										</svg>
									</button>
									<button
										class="action-btn"
										onclick={() => handleExportConversation(conv.id)}
										title="Export"
									>
										<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
											<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
											<polyline points="7 10 12 15 17 10"></polyline>
											<line x1="12" y1="15" x2="12" y2="3"></line>
										</svg>
									</button>
									<button
										class="action-btn delete"
										onclick={() => handleDeleteConversation(conv.id)}
										title="Delete"
									>
										&times;
									</button>
								</div>
							{/if}
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<div class="panel-content messages" bind:this={messagesContainer}>
			{#each chatState.messages as message (message.id)}
				<div class="message {message.role}" class:streaming={message.isStreaming}>
					<div class="message-header">
						<span class="role">{message.role}</span>
						<span class="time">{formatTime(message.timestamp)}</span>
					</div>
					<MessageContent
						content={message.content}
						toolCalls={message.toolCalls}
						toolResults={message.toolResults}
						images={message.images}
						isStreaming={message.isStreaming}
					/>
				</div>
			{/each}
			{#if chatState.isProcessing && chatState.messages.length === 0}
				<div class="message system loading-message">
					<LoadingSpinner message="Connecting to LLM..." />
				</div>
			{/if}
		</div>

		<div class="input-area">
			<textarea
				bind:this={inputElement}
				bind:value={chatState.input}
				onkeydown={handleKeydown}
				placeholder={chatState.isLlmMode
					? 'Ask a question about your data...'
					: 'Enter command (e.g., load_dataset path=/data/file.csv)'}
				disabled={chatState.isProcessing}
				rows="3"
			></textarea>
			<button onclick={handleSend} disabled={chatState.isProcessing || !chatState.input.trim()}>
				{#if chatState.isProcessing}
					<LoadingSpinner size="sm" />
				{:else}
					Send
				{/if}
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
					<p class="text-muted">Use "Import" or ask the AI to load a dataset</p>
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

	/* Header styles */
	.header-left {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
	}

	.header-actions {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.mode-toggle {
		display: flex;
		align-items: center;
		gap: var(--spacing-xs);
		cursor: pointer;
		font-size: 0.85rem;
	}

	.mode-toggle input {
		width: 16px;
		height: 16px;
	}

	.toggle-label {
		color: var(--color-text-secondary);
	}

	.settings-link {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		border-radius: var(--radius-md);
		color: var(--color-text-secondary);
		transition: all 0.2s ease;
	}

	.settings-link:hover {
		background-color: var(--color-surface-hover);
		color: var(--color-text);
	}

	/* Conversations sidebar */
	.conversations-sidebar {
		padding: var(--spacing-sm);
		border-bottom: 1px solid var(--color-border);
		background-color: var(--color-bg-tertiary);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
	}

	.sidebar-header {
		display: flex;
		gap: var(--spacing-xs);
		align-items: center;
	}

	.new-chat-btn {
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 0.8rem;
		background-color: var(--color-primary);
		flex-shrink: 0;
	}

	.conversation-search {
		flex: 1;
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 0.8rem;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		background-color: var(--color-surface);
		color: var(--color-text);
		min-width: 0;
	}

	.conversation-search:focus {
		outline: none;
		border-color: var(--color-primary);
	}

	.conversation-list {
		display: flex;
		flex-wrap: wrap;
		gap: var(--spacing-xs);
	}

	.conversation-item {
		display: flex;
		align-items: center;
		background-color: var(--color-surface);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}

	.conversation-item.active {
		background-color: var(--color-primary);
	}

	.conversation-item.active .conversation-btn {
		color: white;
	}

	.conversation-btn {
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 0.8rem;
		background: transparent;
		color: var(--color-text);
		border-radius: 0;
		max-width: 100px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.conversation-btn:hover {
		background-color: var(--color-surface-hover);
	}

	.conversation-actions {
		display: flex;
		align-items: center;
	}

	.action-btn {
		padding: var(--spacing-xs);
		font-size: 0.8rem;
		background: transparent;
		color: var(--color-text-muted);
		border-radius: 0;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.action-btn:hover {
		background-color: var(--color-surface-hover);
		color: var(--color-text);
	}

	.action-btn.delete:hover {
		background-color: var(--color-error);
		color: white;
	}

	.rename-input {
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 0.8rem;
		border: 1px solid var(--color-primary);
		border-radius: var(--radius-sm);
		background-color: var(--color-bg);
		color: var(--color-text);
		width: 120px;
	}

	.rename-input:focus {
		outline: none;
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

	.message.streaming {
		border-left: 3px solid var(--color-primary);
	}

	.message.loading-message {
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 60px;
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

	.result-header .tool-name {
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
