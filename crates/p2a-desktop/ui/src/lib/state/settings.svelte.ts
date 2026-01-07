// Settings state management using Svelte 5 runes

import type { ProviderConfig, ProviderType } from '$lib/types/llm';
import { DEFAULT_PROVIDER_CONFIG, PROVIDER_REQUIRES_API_KEY } from '$lib/types/llm';
import {
	getLlmSettings,
	updateLlmSettings,
	checkProvider,
	listProviderModels
} from '$lib/api/llm';

class SettingsState {
	// Provider configuration
	config = $state<ProviderConfig>({ ...DEFAULT_PROVIDER_CONFIG });

	// UI state
	isLoading = $state(false);
	isSaving = $state(false);
	isCheckingProvider = $state(false);
	isFetchingModels = $state(false);
	isTestingConnection = $state(false);
	providerAvailable = $state<boolean | null>(null);
	availableModels = $state<string[]>([]);
	error = $state<string | null>(null);
	successMessage = $state<string | null>(null);
	validationErrors = $state<string[]>([]);

	// ========== Validation ==========

	/** Validate the current configuration */
	validate(): boolean {
		this.validationErrors = [];

		// Check API key requirement
		if (PROVIDER_REQUIRES_API_KEY[this.config.provider_type]) {
			if (!this.config.api_key || this.config.api_key.trim() === '') {
				this.validationErrors.push('API key is required for this provider');
			}
		}

		// Check model is selected
		if (!this.config.model || this.config.model.trim() === '') {
			this.validationErrors.push('Please select a model');
		}

		// Validate temperature range
		if (this.config.temperature !== null) {
			if (this.config.temperature < 0 || this.config.temperature > 2) {
				this.validationErrors.push('Temperature must be between 0 and 2');
			}
		}

		// Validate max tokens
		if (this.config.max_tokens !== null) {
			if (this.config.max_tokens < 1 || this.config.max_tokens > 100000) {
				this.validationErrors.push('Max tokens must be between 1 and 100000');
			}
		}

		return this.validationErrors.length === 0;
	}

	// ========== Load/Save Methods ==========

	/** Load settings from backend */
	async loadSettings() {
		this.isLoading = true;
		this.error = null;
		try {
			const settings = await getLlmSettings();
			this.config = settings;
			// Check provider availability after loading
			await this.checkProviderAvailability();
		} catch (e) {
			this.error = `Failed to load settings: ${e}`;
		} finally {
			this.isLoading = false;
		}
	}

	/** Save settings to backend with validation */
	async saveSettings(): Promise<boolean> {
		// Validate first
		if (!this.validate()) {
			this.error = this.validationErrors.join('. ');
			return false;
		}

		this.isSaving = true;
		this.error = null;
		this.successMessage = null;
		try {
			await updateLlmSettings(this.config);
			this.successMessage = 'Settings saved successfully';
			// Clear success message after 3 seconds
			setTimeout(() => {
				this.successMessage = null;
			}, 3000);
			return true;
		} catch (e) {
			this.error = `Failed to save settings: ${e}`;
			return false;
		} finally {
			this.isSaving = false;
		}
	}

	// ========== Provider Methods ==========

	/** Check if current provider is available */
	async checkProviderAvailability() {
		this.isCheckingProvider = true;
		this.providerAvailable = null;
		try {
			this.providerAvailable = await checkProvider();
			if (this.providerAvailable) {
				// Fetch available models
				await this.fetchModels();
			}
		} catch (e) {
			this.providerAvailable = false;
			this.error = `Provider check failed: ${e}`;
		} finally {
			this.isCheckingProvider = false;
		}
	}

	/** Test connection by saving settings first, then checking provider */
	async testConnection() {
		this.isTestingConnection = true;
		this.error = null;
		this.successMessage = null;

		try {
			// Save settings first
			const saved = await this.saveSettings();
			if (!saved) {
				return;
			}

			// Then check provider
			await this.checkProviderAvailability();

			if (this.providerAvailable) {
				this.successMessage = 'Connection successful! Provider is available.';
			} else {
				this.error = 'Connection failed. Provider is not available.';
			}
		} finally {
			this.isTestingConnection = false;
		}
	}

	/** Fetch available models for current provider */
	async fetchModels() {
		this.isFetchingModels = true;
		try {
			this.availableModels = await listProviderModels();
		} catch (e) {
			console.error('Failed to fetch models:', e);
			this.availableModels = [];
		} finally {
			this.isFetchingModels = false;
		}
	}

	/** Refresh models list (save first, then fetch) */
	async refreshModels() {
		// Save current settings first so provider has correct config
		const saved = await this.saveSettings();
		if (saved) {
			await this.fetchModels();
		}
	}

	// ========== Config Update Methods ==========

	setProviderType(providerType: ProviderType) {
		this.config.provider_type = providerType;
		// Reset model to default for the new provider
		this.config.model = this.getDefaultModel(providerType);
		// Clear API key if switching to Ollama
		if (providerType === 'ollama') {
			this.config.api_key = null;
			this.config.base_url = null;
		}
		// Reset provider availability
		this.providerAvailable = null;
		this.availableModels = [];
		// Clear validation errors
		this.validationErrors = [];
	}

	setModel(model: string) {
		this.config.model = model;
	}

	setApiKey(apiKey: string | null) {
		this.config.api_key = apiKey;
		// Clear validation errors when user types
		this.validationErrors = this.validationErrors.filter((e) => !e.includes('API key'));
	}

	setBaseUrl(baseUrl: string | null) {
		this.config.base_url = baseUrl;
	}

	setTemperature(temperature: number | null) {
		this.config.temperature = temperature;
	}

	setMaxTokens(maxTokens: number | null) {
		this.config.max_tokens = maxTokens;
	}

	// ========== Helper Methods ==========

	getDefaultModel(providerType: ProviderType): string {
		switch (providerType) {
			case 'ollama':
				return 'llama3.2';
			case 'anthropic':
				return 'claude-sonnet-4-20250514';
			case 'openai':
				return 'gpt-4o';
			default:
				return 'llama3.2';
		}
	}

	clearError() {
		this.error = null;
		this.validationErrors = [];
	}

	clearSuccessMessage() {
		this.successMessage = null;
	}

	/** Check if form has validation errors */
	hasValidationErrors(): boolean {
		return this.validationErrors.length > 0;
	}
}

export const settingsState = new SettingsState();
