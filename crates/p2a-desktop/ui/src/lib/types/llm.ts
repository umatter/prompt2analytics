// LLM-related type definitions for prompt2analytics desktop app

/** Supported LLM provider types */
export type ProviderType = 'ollama' | 'anthropic' | 'openai';

/** Configuration for an LLM provider */
export interface ProviderConfig {
	provider_type: ProviderType;
	api_key: string | null;
	base_url: string | null;
	model: string;
	temperature: number | null;
	max_tokens: number | null;
}

/** Role of a message in the conversation */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** A tool call requested by the LLM */
export interface ToolCall {
	id: string;
	name: string;
	arguments: unknown;
}

/** Result of executing a tool */
export interface LlmToolResult {
	tool_call_id: string;
	content: string;
	is_error: boolean;
}

/** A message in the LLM conversation */
export interface LlmMessage {
	role: MessageRole;
	content: string;
	tool_calls?: ToolCall[];
	tool_results?: LlmToolResult[];
}

/** A stored message with metadata */
export interface StoredMessage {
	id: string;
	conversation_id: string;
	role: MessageRole;
	content: string;
	tool_calls?: ToolCall[];
	tool_results?: LlmToolResult[];
	timestamp: string;
}

/** A conversation with metadata */
export interface Conversation {
	id: string;
	title: string;
	created_at: string;
	updated_at: string;
	provider: ProviderType;
	model: string;
}

/** Response from sending a message */
export interface SendMessageResponse {
	conversation_id: string;
	message: LlmMessage;
}

/** Streaming response chunk types */
export type StreamChunk =
	| { type: 'text'; content: string }
	| { type: 'tool_call'; tool_call: ToolCall }
	| { type: 'tool_result'; tool_result: LlmToolResult }
	| { type: 'done' }
	| { type: 'error'; message: string };

/** Default provider configuration */
export const DEFAULT_PROVIDER_CONFIG: ProviderConfig = {
	provider_type: 'ollama',
	api_key: null,
	base_url: null,
	model: 'llama3.2',
	temperature: 0.7,
	max_tokens: 4096
};

/** Available models for each provider (defaults, actual list fetched from API) */
export const PROVIDER_DEFAULT_MODELS: Record<ProviderType, string[]> = {
	ollama: ['llama3.2', 'llama3.1', 'mistral', 'codellama', 'phi3'],
	anthropic: [
		'claude-sonnet-4-20250514',
		'claude-3-5-sonnet-20241022',
		'claude-3-5-haiku-20241022',
		'claude-3-opus-20240229'
	],
	openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']
};

/** Provider display names */
export const PROVIDER_NAMES: Record<ProviderType, string> = {
	ollama: 'Ollama (Local)',
	anthropic: 'Anthropic Claude',
	openai: 'OpenAI GPT'
};

/** Whether a provider requires an API key */
export const PROVIDER_REQUIRES_API_KEY: Record<ProviderType, boolean> = {
	ollama: false,
	anthropic: true,
	openai: true
};
