<script lang="ts">
	import { onMount } from 'svelte';
	import { settingsState } from '$lib/state/settings.svelte';
	import {
		PROVIDER_NAMES,
		PROVIDER_REQUIRES_API_KEY,
		PROVIDER_DEFAULT_MODELS,
		type ProviderType
	} from '$lib/types/llm';
	import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';

	const providerTypes: ProviderType[] = ['ollama', 'anthropic', 'openai'];

	onMount(async () => {
		await settingsState.loadSettings();
	});

	// Handle provider change
	function handleProviderChange(e: Event) {
		const target = e.currentTarget as HTMLSelectElement;
		settingsState.setProviderType(target.value as ProviderType);
	}

	// Handle model change
	function handleModelChange(e: Event) {
		const target = e.currentTarget as HTMLSelectElement;
		settingsState.setModel(target.value);
	}

	// Handle API key change
	function handleApiKeyChange(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		settingsState.setApiKey(target.value || null);
	}

	// Handle base URL change
	function handleBaseUrlChange(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		settingsState.setBaseUrl(target.value || null);
	}

	// Handle temperature change
	function handleTemperatureChange(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		const value = parseFloat(target.value);
		settingsState.setTemperature(isNaN(value) ? null : value);
	}

	// Handle max tokens change
	function handleMaxTokensChange(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		const value = parseInt(target.value);
		settingsState.setMaxTokens(isNaN(value) ? null : value);
	}

	// Get available models - use fetched models if available, otherwise defaults
	function getAvailableModels(): string[] {
		if (settingsState.availableModels.length > 0) {
			return settingsState.availableModels;
		}
		return PROVIDER_DEFAULT_MODELS[settingsState.config.provider_type];
	}

	// Check if current provider requires API key
	function requiresApiKey(): boolean {
		return PROVIDER_REQUIRES_API_KEY[settingsState.config.provider_type];
	}
</script>

<div class="settings-page">
	<header class="settings-header">
		<a href="/" class="back-link">&larr; Back to Chat</a>
		<h1>LLM Settings</h1>
	</header>

	{#if settingsState.isLoading}
		<div class="loading">
			<LoadingSpinner size="lg" message="Loading settings..." />
		</div>
	{:else}
		<div class="settings-content">
			<!-- Provider Selection -->
			<section class="settings-section">
				<h2>LLM Provider</h2>
				<p class="section-description">
					Choose your preferred LLM provider. Ollama runs locally, while Anthropic and OpenAI
					require API keys.
				</p>

				<div class="form-group">
					<label for="provider">Provider</label>
					<select
						id="provider"
						value={settingsState.config.provider_type}
						onchange={handleProviderChange}
					>
						{#each providerTypes as pt}
							<option value={pt}>{PROVIDER_NAMES[pt]}</option>
						{/each}
					</select>
				</div>

				<!-- Provider Status -->
				<div class="provider-status">
					{#if settingsState.isCheckingProvider}
						<span class="status checking">Checking provider...</span>
					{:else if settingsState.providerAvailable === true}
						<span class="status available">Provider available</span>
					{:else if settingsState.providerAvailable === false}
						<span class="status unavailable">Provider unavailable</span>
					{/if}
					<button class="secondary small" onclick={() => settingsState.checkProviderAvailability()}>
						Check Status
					</button>
				</div>
			</section>

			<!-- Model Selection -->
			<section class="settings-section">
				<h2>Model</h2>
				<div class="form-group">
					<label for="model">Model</label>
					<div class="model-row">
						<select id="model" value={settingsState.config.model} onchange={handleModelChange} disabled={settingsState.isFetchingModels}>
							{#each getAvailableModels() as model}
								<option value={model}>{model}</option>
							{/each}
						</select>
						<button
							class="secondary small"
							onclick={() => settingsState.refreshModels()}
							disabled={settingsState.isFetchingModels || settingsState.isSaving}
							title="Refresh available models from provider"
						>
							{#if settingsState.isFetchingModels}
								<LoadingSpinner size="sm" />
							{:else}
								Refresh
							{/if}
						</button>
					</div>
					<p class="help-text">
						{#if settingsState.config.provider_type === 'ollama'}
							Make sure the model is pulled locally with: ollama pull {settingsState.config.model}
						{:else}
							Select a model available in your account.
						{/if}
					</p>
				</div>
			</section>

			<!-- API Key (for cloud providers) -->
			{#if requiresApiKey()}
				<section class="settings-section">
					<h2>Authentication</h2>
					<div class="form-group">
						<label for="api-key">API Key <span class="required">*</span></label>
						<input
							id="api-key"
							type="password"
							value={settingsState.config.api_key ?? ''}
							oninput={handleApiKeyChange}
							placeholder="Enter your API key"
							class:input-error={settingsState.validationErrors.some(e => e.includes('API key'))}
						/>
						{#if settingsState.validationErrors.some(e => e.includes('API key'))}
							<p class="validation-error">API key is required for this provider</p>
						{/if}
						<p class="help-text">
							{#if settingsState.config.provider_type === 'anthropic'}
								Get your API key from <a
									href="https://console.anthropic.com/"
									target="_blank"
									rel="noopener">console.anthropic.com</a
								>
							{:else if settingsState.config.provider_type === 'openai'}
								Get your API key from <a
									href="https://platform.openai.com/api-keys"
									target="_blank"
									rel="noopener">platform.openai.com</a
								>
							{/if}
						</p>
					</div>
				</section>
			{/if}

			<!-- Advanced Settings -->
			<section class="settings-section">
				<h2>Advanced Settings</h2>

				{#if settingsState.config.provider_type === 'ollama'}
					<div class="form-group">
						<label for="base-url">Base URL (optional)</label>
						<input
							id="base-url"
							type="text"
							value={settingsState.config.base_url ?? ''}
							oninput={handleBaseUrlChange}
							placeholder="http://localhost:11434"
						/>
						<p class="help-text">Leave empty to use default (localhost:11434)</p>
					</div>
				{/if}

				<div class="form-row">
					<div class="form-group">
						<label for="temperature">Temperature</label>
						<input
							id="temperature"
							type="number"
							min="0"
							max="2"
							step="0.1"
							value={settingsState.config.temperature ?? 0.7}
							oninput={handleTemperatureChange}
						/>
						<p class="help-text">0 = deterministic, 2 = creative</p>
					</div>

					<div class="form-group">
						<label for="max-tokens">Max Tokens</label>
						<input
							id="max-tokens"
							type="number"
							min="100"
							max="32000"
							step="100"
							value={settingsState.config.max_tokens ?? 4096}
							oninput={handleMaxTokensChange}
						/>
						<p class="help-text">Maximum response length</p>
					</div>
				</div>
			</section>

			<!-- Error/Success Messages -->
			{#if settingsState.error}
				<div class="message error-message">
					{settingsState.error}
					<button class="dismiss" onclick={() => settingsState.clearError()}>Dismiss</button>
				</div>
			{/if}

			{#if settingsState.successMessage}
				<div class="message success-message">
					{settingsState.successMessage}
				</div>
			{/if}

			<!-- Action Buttons -->
			<div class="actions">
				<button
					class="secondary"
					onclick={() => settingsState.testConnection()}
					disabled={settingsState.isTestingConnection || settingsState.isSaving}
				>
					{#if settingsState.isTestingConnection}
						<LoadingSpinner size="sm" message="Testing..." />
					{:else}
						Test Connection
					{/if}
				</button>
				<button onclick={() => settingsState.saveSettings()} disabled={settingsState.isSaving || settingsState.isTestingConnection}>
					{#if settingsState.isSaving}
						<LoadingSpinner size="sm" message="Saving..." />
					{:else}
						Save Settings
					{/if}
				</button>
			</div>
		</div>
	{/if}
</div>

<style>
	.settings-page {
		max-width: 700px;
		margin: 0 auto;
		padding: var(--spacing-lg);
	}

	.settings-header {
		margin-bottom: var(--spacing-xl);
	}

	.back-link {
		display: inline-block;
		margin-bottom: var(--spacing-md);
		color: var(--color-primary);
		text-decoration: none;
	}

	.back-link:hover {
		text-decoration: underline;
	}

	h1 {
		margin: 0;
		font-size: 1.75rem;
	}

	.loading {
		text-align: center;
		padding: var(--spacing-xl);
		color: var(--color-text-secondary);
	}

	.settings-content {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xl);
	}

	.settings-section {
		background-color: var(--color-surface);
		padding: var(--spacing-lg);
		border-radius: var(--radius-lg);
		border: 1px solid var(--color-border);
	}

	.settings-section h2 {
		margin: 0 0 var(--spacing-sm) 0;
		font-size: 1.1rem;
	}

	.section-description {
		margin: 0 0 var(--spacing-lg) 0;
		color: var(--color-text-secondary);
		font-size: 0.9rem;
	}

	.form-group {
		margin-bottom: var(--spacing-md);
	}

	.form-group:last-child {
		margin-bottom: 0;
	}

	.form-group label {
		display: block;
		margin-bottom: var(--spacing-xs);
		font-weight: 500;
	}

	.form-group input,
	.form-group select {
		width: 100%;
		padding: var(--spacing-sm) var(--spacing-md);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		background-color: var(--color-bg);
		color: var(--color-text);
		font-size: 1rem;
	}

	.form-group input:focus,
	.form-group select:focus {
		outline: none;
		border-color: var(--color-primary);
		box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
	}

	.form-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-md);
	}

	.model-row {
		display: flex;
		gap: var(--spacing-sm);
	}

	.model-row select {
		flex: 1;
	}

	.help-text {
		margin: var(--spacing-xs) 0 0 0;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.required {
		color: var(--color-error);
	}

	.input-error {
		border-color: var(--color-error) !important;
	}

	.validation-error {
		margin: var(--spacing-xs) 0 0 0;
		font-size: 0.85rem;
		color: var(--color-error);
	}

	.help-text a {
		color: var(--color-primary);
	}

	.provider-status {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		margin-top: var(--spacing-md);
		padding: var(--spacing-sm) var(--spacing-md);
		background-color: var(--color-bg);
		border-radius: var(--radius-md);
	}

	.status {
		flex: 1;
		font-size: 0.9rem;
	}

	.status.checking {
		color: var(--color-text-secondary);
	}

	.status.available {
		color: var(--color-success);
	}

	.status.unavailable {
		color: var(--color-error);
	}

	button.small {
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 0.85rem;
	}

	.message {
		padding: var(--spacing-md);
		border-radius: var(--radius-md);
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.error-message {
		background-color: rgba(239, 68, 68, 0.1);
		border: 1px solid var(--color-error);
		color: var(--color-error);
	}

	.success-message {
		background-color: rgba(34, 197, 94, 0.1);
		border: 1px solid var(--color-success);
		color: var(--color-success);
	}

	.dismiss {
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 0.85rem;
		background: transparent;
		border: 1px solid currentColor;
		color: inherit;
	}

	.actions {
		display: flex;
		justify-content: flex-end;
		gap: var(--spacing-md);
	}

	.actions button {
		min-width: 150px;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-xs);
	}
</style>
