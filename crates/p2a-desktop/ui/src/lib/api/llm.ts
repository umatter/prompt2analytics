// LLM API wrappers for Tauri commands

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type {
	Conversation,
	LlmMessage,
	ProviderConfig,
	SendMessageResponse,
	StreamChunk
} from '$lib/types/llm';

/**
 * Send a message to the LLM and get a complete response.
 * @param message The user's message
 * @param conversationId Optional existing conversation ID
 */
export async function sendMessage(
	message: string,
	conversationId?: string
): Promise<SendMessageResponse> {
	return invoke<SendMessageResponse>('send_message', {
		message,
		conversationId
	});
}

/**
 * Send a message with streaming response.
 * Use listenToStream to receive stream events.
 * @param message The user's message
 * @param conversationId Optional existing conversation ID
 */
export async function sendMessageStream(
	message: string,
	conversationId?: string
): Promise<SendMessageResponse> {
	return invoke<SendMessageResponse>('send_message_stream', {
		message,
		conversationId
	});
}

/**
 * Listen to LLM stream events.
 * @param callback Function to handle stream chunks
 * @returns Unlisten function to stop listening
 */
export async function listenToStream(
	callback: (chunk: StreamChunk) => void
): Promise<UnlistenFn> {
	return listen<StreamChunk>('llm-stream', (event) => {
		callback(event.payload);
	});
}

/**
 * List all conversations.
 */
export async function listConversations(): Promise<Conversation[]> {
	return invoke<Conversation[]>('list_conversations');
}

/**
 * Get messages for a conversation.
 * @param conversationId The conversation ID
 */
export async function getConversation(conversationId: string): Promise<LlmMessage[]> {
	return invoke<LlmMessage[]>('get_conversation', { conversationId });
}

/**
 * Delete a conversation.
 * @param conversationId The conversation ID
 */
export async function deleteConversation(conversationId: string): Promise<void> {
	return invoke('delete_conversation', { conversationId });
}

/**
 * Get current LLM settings.
 */
export async function getLlmSettings(): Promise<ProviderConfig> {
	return invoke<ProviderConfig>('get_llm_settings');
}

/**
 * Update LLM settings.
 * @param config The new configuration
 */
export async function updateLlmSettings(config: ProviderConfig): Promise<void> {
	return invoke('update_llm_settings', { config });
}

/**
 * Check if the current provider is available.
 */
export async function checkProvider(): Promise<boolean> {
	return invoke<boolean>('check_provider');
}

/**
 * List available models for the current provider.
 */
export async function listProviderModels(): Promise<string[]> {
	return invoke<string[]>('list_provider_models');
}

/**
 * Get list of supported provider types.
 */
export async function listProviderTypes(): Promise<string[]> {
	return invoke<string[]>('list_provider_types');
}

/**
 * Rename a conversation.
 * @param conversationId The conversation ID
 * @param newTitle The new title
 */
export async function renameConversation(conversationId: string, newTitle: string): Promise<void> {
	return invoke('rename_conversation', { conversationId, newTitle });
}

/** Exported conversation data */
export interface ExportedConversation {
	id: string;
	title: string;
	created_at: string;
	updated_at: string;
	provider: string;
	model: string;
	messages: LlmMessage[];
}

/**
 * Export a conversation with all metadata and messages.
 * @param conversationId The conversation ID
 */
export async function exportConversation(conversationId: string): Promise<ExportedConversation> {
	return invoke<ExportedConversation>('export_conversation', { conversationId });
}
